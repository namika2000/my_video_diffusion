"""
Codebase for "Improved Denoising Diffusion Probabilistic Models".

Raw copy of [1] at commit 32b43b6c677df7642c5408c8ef4a09272787eb50 from Feb 22, 2021.

[1] https://github.com/openai/improved-diffusion/tree/main/improved_diffusion
"""
from .gaussian_diffusion import GaussianDiffusion
from .unet import Unet3D
