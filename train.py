import torch
from diffusion.unet import Unet3D
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.train_util import Trainer

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

# diffusion = GaussianDiffusion(
#     model,
#     image_size = 64,
#     num_frames = 10,
#     timesteps = 1000,   # number of steps
#     loss_type = 'l1'    # L1 or L2
# ).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 10,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

trainer = Trainer(
    diffusion,
    './kinetics-dataset',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 32,
    train_lr = 1e-4,
    save_and_sample_every = 1000,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()