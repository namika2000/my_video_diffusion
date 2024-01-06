"""
Train a diffusion model on images.
"""

import argparse
import os, sys
sys.path.insert(1, os.getcwd()) 
# import numpy as np
import torch
from torch.utils import data
from diffusion import logger
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion.train_util import Trainer

def main():

    parser, defaults = create_argparser()
    args = parser.parse_args()
    parameters = args_to_dict(args, defaults.keys())
    # th.manual_seed(args.seed)
    # np.random.seed(args.seed)

    logger.configure()
    for key, item in parameters.items():
        logger.logkv(key, item)
    logger.dumpkvs()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    logger.log("creating data loader...")

    ds = Dataset(args.folder, diffusion.image_size, channels = diffusion.channels, num_frames = diffusion.num_frames)
    print(f'found {len(self.ds)} videos as gif files at {folder}')
    assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'
    ds = cycle(data.DataLoader(ds, batch_size = args.train_batch_size, shuffle=True, pin_memory=True))
        
    if args.mask_range is None:
        mask_range = [0, args.seq_len]
    else:
        mask_range = [int(i) for i in args.mask_range if i != ","]
    logger.log("training...")

    trainer = Trainer(
    diffusion.cuda(),
    data=ds,
    train_batch_size=args.train_batch_size,
    train_lr=args.train_lr,
    train_num_steps=args.train_num_steps,  
    save_and_sample_every=args.save_and_sample_every,
    max_num_mask_frames=args.max_num_mask_frames, 
    mask_range=mask_range, 
    null_cond_prob=args.null_cond_prob,
    exclude_conditional=args.exclude_conditional
    )
    
    trainer.train()

# コマンドライン引数からパラメータの設定を行うための準備
def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=64,
        microbatch=32,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip=1,
        seed=123,
        anneal_type=None,
        steps_drop=0.0,
        drop=0.0,
        decay=0.0,
        seq_len=20,
        max_num_mask_frames=4,
        mask_range=None,
        uncondition_rate=0.0,
        exclude_conditional=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser, defaults


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    end = time.time()
    print(f"training time: {end - start}")