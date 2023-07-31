# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import src.dnnlib as dnnlib
from torch_utils import distributed as dist

from src.training_loops import denoise_training_loop, guide_training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--config_path',          help='Path to the config file', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--exp_name_suffix', help='Experiment name suffix, will be put at the end of the log file', type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)
@click.option('--train_mode',      help='guidance training or denoising training', metavar='guide|denoise', type=click.Choice(['guide', 'denoise']), default='denoise', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--sig_data',      help='sigma of the data, used in precond and loss', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)
@click.option('--p_mean',        help='mean of the log_sigma for training', metavar='FLOAT', type=float, default=0.0, show_default=True)
@click.option('--p_std',         help='sigma of the log_sigma for training', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--lambda_dir',    help='the weight for the edge direction loss', metavar='FLOAT', type=float, default=1e-3, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=0), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('--guide_ckpt',    help='ckpt of the guidance network to be loaded', type=str)
@click.option('--pretrained_model_ckpt',    help='ckpt of the pretrained model to init the denoising network', type=str, default=None)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)


def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.config_path = opts.config_path
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=False, num_workers=opts.workers)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=opts.lr, weight_decay=1e-4, betas=[0.9,0.999], eps=1e-8)

    # Network architecture.
    c.network_kwargs.update(model_type='maptr')
    
    if opts.pretrained_model_ckpt is not None:
        c.network_kwargs['pretrained_model_ckpt'] = opts.pretrained_model_ckpt

    # Preconditioning & loss function.
    assert opts.precond == 'edm'
    c.network_kwargs.class_name = 'src.models.networks.EDMPrecond'

    if opts.train_mode == 'denoise':
        c.loss_kwargs.class_name = 'src.losses.loss_denoise.EDMLoss'
        c.loss_kwargs.lambda_dir = opts.lambda_dir
    else:
        c.loss_kwargs.class_name = 'src.losses.loss_guide.GuideLoss'
    c.loss_kwargs.P_mean = opts.p_mean
    c.loss_kwargs.P_std = opts.p_std
    c.loss_kwargs.sigma_data = opts.sig_data

    # Network options.
    c.network_kwargs.update(use_fp16=opts.fp16)
    c.network_kwargs.update(sigma_data=opts.sig_data)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pth', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot.pth')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # set up the guidance network ckpt path
    if opts.guide_ckpt is not None:
        c.guide_ckpt = opts.guide_ckpt

    # Description string.
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'nuscenes-{opts.train_mode}-{opts.precond:s}-gpus{dist.get_world_size():d}' \
            f'-batch{c.batch_size:d}-Pmean_{c.loss_kwargs.P_mean}-Pstd_{c.loss_kwargs.P_std}-sigd_{c.loss_kwargs.sigma_data}' \
            f'-{dtype_str:s}-{opts.exp_name_suffix}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    
    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        c.run_dir = os.path.join(opts.outdir, f'{desc}')

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Config path:             {c.config_path}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    if opts.train_mode == 'denoise':
        denoise_training_loop.training_loop(**c)
    else:
        guide_training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
