import copy
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange

from utils import exists, noop, cycle, num_to_groups

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))



# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        data,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None,
        max_num_mask_frames=4, 
        mask_range=None, 
        null_cond_prob=0,
        exclude_conditional=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.data = data
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.max_num_mask_frames = max_num_mask_frames, 
        self.mask_range = mask_range, 
        self.null_cond_prob = null_cond_prob,
        self.exclude_conditional = exclude_conditional,

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.data).cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        self.max_num_mask_frames, 
                        self.mask_range, 
                        self.null_cond_prob,
                        self.exclude_conditional,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)

                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, 'sample': video_path}
                self.save(milestone)

            log_fn(log)
            self.step += 1

        print('training completed')
