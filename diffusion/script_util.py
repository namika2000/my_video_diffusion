import argparse
import inspect

from . import gaussian_diffusion as gd
from .unet import Unet3D

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    attn_heads = 8,
    attn_dim_head = 32,
    init_kernel_size = 7,
    resnet_groups = 8,
    image_size = 64,
    num_frames = 10,
    timesteps=1000
    )


def create_model_and_diffusion(
    dim,
    dim_mults,
    attn_heads,
    attn_dim_head,
    init_kernel_size,
    resnet_groups,
    image_size,
    num_frames,
    timesteps
):
    model = create_model(
    dim,
    dim_mults,
    attn_heads,
    attn_dim_head,
    init_kernel_size,
    resnet_groups
)
    diffusion = create_gaussian_diffusion(
    model,
    image_size,
    num_frames,
    timesteps,
)
    return model, diffusion


def create_model(
    dim,
    dim_mults,
    attn_heads,
    attn_dim_head,
    init_kernel_size,
    resnet_groups
):
    return UNet3D(dim,
    dim_mults=dim_mults,
    attn_heads = attn_heads,
    attn_dim_head = attn_dim_head,
    init_kernel_size = init_kernel_size,
    resnet_groups = resnet_groups)
  
def create_gaussian_diffusion(
    denoise_fn,
    *,
    image_size,
    num_frames,
    timesteps,
):
    return gd.GaussianDiffusion(
    denoise_fn,
    image_size,
    num_frames,
    timesteps = timesteps,
)


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")