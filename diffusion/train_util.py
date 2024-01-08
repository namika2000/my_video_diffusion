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

from .utils import exists, noop, cycle, num_to_groups

# dataset for mpv4
# @title mpv4
import imageio

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}

def video_to_tensor(video_path, channels=3, transform=T.ToTensor()):
    video = imageio.get_reader(video_path, 'ffmpeg')
    frames = [transform(Image.fromarray(frame)) for frame in video]
    return torch.stack(frames, dim=1)

def identity(t, *args, **kwargs):
    return t

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class VideoDataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=3,
        num_frames=10,
        horizontal_flip=False,
        force_num_frames=True,
        exts=['mp4'],
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        print("path: ", path," index: ", index)
        tensor = video_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)

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
        # data,
        folder,
        experiment,
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
        num_sample_rows = 4,
        max_grad_norm = None,
        max_num_mask_frames=4, 
        mask_range=None, 
        seq_len=20,
        null_cond_prob=0.25,
        exclude_conditional=True
    ):
        super().__init__()
        self.model = diffusion_model
        # self.data = data
        self.folder = folder
        self.experiment = experiment
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.max_num_mask_frames = max_num_mask_frames, 
        if mask_range is None:
          mask_range = [0, seq_len]
        else:
          mask_range = [int(i) for i in mask_range if i != ","]
        self.mask_range = mask_range, 
        self.null_cond_prob = null_cond_prob,
        self.exclude_conditional = exclude_conditional,

        self.ds = VideoDataset(folder, diffusion_model.image_size, channels = diffusion_model.channels, num_frames = diffusion_model.num_frames)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        results_folder = "results/experiment/" + str(self.experiment)
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

    def save(self, milestone, log):
          # 保存するモデルのファイルパスを指定
        model_path = str(self.results_folder / f'model_{milestone}.pt')
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        # モデルの状態辞書を保存
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

        # ハイパーパラメータを保存
        hyperparameters = {
            'train_num_steps': self.train_num_steps,
            'gradient_accumulate_every': self.gradient_accumulate_every,
            'folder': self.folder,
            'experiment': self.experiment,
            'ema_decay': self.ema_decay,
            'num_frames': self.num_frames,
            'train_batch_size': self.train_batch_size,
            'train_lr': self.train_lr,
            'amp': self.amp,
            'step_start_ema': self.step_start_ema,
            'update_ema_every': self.update_ema_every,
            'save_and_sample_every': self.save_and_sample_every,
            'num_sample_rows': self.num_sample_rows,
            'max_grad_norm': self.max_grad_norm,
            'mask_range': self.mask_range,
            'max_num_mask_frames': self.max_num_mask_frames,
            'exclude_conditional': self.exclude_conditional,
            'seq_len': self.seq_len,
            # 他のハイパーパラメータもここに追加
        }
        hyperparameters_path = str(self.results_folder / 'hyperparameters.json')
        with open(hyperparameters_path, 'w') as json_file:
            json.dump(hyperparameters, json_file, indent=2)

        print(f'Model saved at {model_path}')
        print(f'Hyperparameters saved at {hyperparameters_path}')
        return log

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
        log_fn = noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cpu()

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        self.max_num_mask_frames, 
                        self.mask_range, 
                        self.null_cond_prob,
                        self.exclude_conditional,
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
                self.save(milestone, log)

            log_fn(log)
            self.step += 1

        print('training completed')
