# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import psutil
import numpy as np
import torch
import src.dnnlib as dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv import Config, DictAction
from projects.mmdet3d_plugin.datasets.nuscenes_map_dataset import polygon_collate
import importlib

from src.models.polygon_models.polygon_meta import PolyMetaModel

from torch.nn.utils import clip_grad_norm_
from math import cos, pi

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    config_path         = '',       # MapTR Config file
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 18000,    # Training duration, measured in thousands of training images.
    lr_rampup_kimg      = 5,       # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    guide_ckpt          = None,
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    grad_clip           = 2.0,
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # load the config file
    cfg = Config.fromfile(config_path)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            assert hasattr(cfg, 'plugin_dir')
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

    # Load dataset.
    dist.print0('Loading dataset...')

    cfg.data.train['drop_instance'] = True # randomly drop poly-instance to mimic the missing of instances at test time
    cfg.data.train['load_image'] = True
    dataset_obj = build_dataset(cfg.data.train) 
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, \
                    batch_size=batch_gpu, collate_fn=polygon_collate, \
                    **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()

    # load the image backbone and transformer encoder from the pre-trained MapTR to accelerate convergence
    assert 'pretrained_model_ckpt' in network_kwargs
    if dist.get_rank() != 0:
        torch.distributed.barrier() # rank 0 goes first to load the ckpt
    pretrained_ckpt = torch.load(network_kwargs['pretrained_model_ckpt'], 
                    map_location='cuda:{}'.format(dist.get_rank()))
    keys_to_load = ['img_backbone', 'img_neck', 'pts_bbox_head.transformer.encoder', 
        'pts_bbox_head.transformer.level_embeds', 'pts_bbox_head.transformer.cams_embeds',
        'pts_bbox_head.transformer.can_bus_mlp']
    loaded_param_name = model.load_from_ckpt(pretrained_ckpt['state_dict'], keys_to_load)
    del pretrained_ckpt
    if dist.get_rank() == 0:
        torch.distributed.barrier() # other ranks follow

    network_kwargs['model'] = model
    net = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    net.train().to(device)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    dist.print0('Total number of trainable parameters: {}'.format(n_parameters))

    # Setup optimizer.
    dist.print0('Setting up optimizer...')

    loss_kwargs['pc_range'] = cfg.point_cloud_range
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss

    # setup lr
    loaded_params, from_scratch_params = [], []
    for name, param in net.named_parameters():
        if name[6:] in loaded_param_name and 'img_backbone' in name:
        #if 'img_backbone' in name: # only the image backbone is pre-loaded (ImageNet pre-trained) 
            loaded_params.append(param)
        else:
            from_scratch_params.append(param)

    params_setups = [
        {'params': loaded_params, 'lr': optimizer_kwargs['lr'] * 0.1},
        {'params': from_scratch_params, 'lr': optimizer_kwargs['lr']},
    ]

    optimizer = dnnlib.util.construct_class_by_name(params_setups, **optimizer_kwargs) # subclass of torch.optim.Optimizer

    optimizer_init_lr_dict = {}    
    for group_idx, param in enumerate(optimizer.param_groups):
        optimizer_init_lr_dict[group_idx] = param['lr']

    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)

    pe_dim = net.model.pts_bbox_head.positional_encoding.num_feats
    embed_dim = net.model.pts_bbox_head.transformer.embed_dims
    net_guide = PolyMetaModel(input_dim=pe_dim, embed_dim=embed_dim)
    net_guide.eval().requires_grad_(False).to(device)
    guide_ckpt = torch.load(guide_ckpt, map_location='cuda:{}'.format(dist.get_rank()))
    net_guide.load_state_dict(guide_ckpt['net'])
    dist.print0('Loading the guidance network')

    # Resume training from previous snapshot.
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location='cuda:{}'.format(dist.get_rank()))
        net.load_state_dict(data['net'])
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory
    else:
        if resume_pkl is not None:
            dist.print0(f'Loading network weights from "{resume_pkl}"...')
            if dist.get_rank() != 0:
                torch.distributed.barrier() # rank 0 goes first
            data = torch.load(resume_pkl, map_location='cuda:{}'.format(dist.get_rank()))
            if dist.get_rank() == 0:
                torch.distributed.barrier() # other ranks follow
            net.load_state_dict(data['net'])
            del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                data_batch = next(dataset_iterator)
                # abuse the "images" here for the data
                images = data_batch['gt_bboxes_3d'].to(device)
                attn_mask = data_batch['pts_mask'].to(device)
                model_kwargs = {
                    'img': data_batch['img'].to(device),
                    'poly_class': data_batch['gt_labels_3d'].to(device),
                    'poly_mask': attn_mask.to(device),
                    'img_metas': data_batch['img_metas']
                }

                loss, last_layer_loss, loss_coords, loss_dir, last_loss_coords, last_loss_dir  = \
                    loss_fn(net=ddp, net_guide=net_guide, images=images, **model_kwargs)
                training_stats.report('Loss/loss', loss)
                training_stats.report('Loss/loss_last_layer', last_layer_loss)
                training_stats.report('Loss/loss_coords', loss_coords)
                training_stats.report('Loss/loss_dir', loss_dir)
                training_stats.report('Loss/last_coords', last_loss_coords)
                training_stats.report('Loss/last_dir', last_loss_dir)

                loss = loss.sum().mul(loss_scaling / batch_gpu_total)

                # FPN has some ununsed BN parameters, 
                # This is a workaround for DDP's error regarding those unused parameters
                pseudo_losses = [p.sum() * 0 for p in ddp.parameters()]
                pseudo_losses = torch.stack(pseudo_losses).mean()
                loss += pseudo_losses    
                loss.backward()
        
        # Update weights.
        
        # learning rate warmup and cosine lr decay after warmup
        if cur_nimg <= lr_rampup_kimg * 1000:
            for g_idx, g in enumerate(optimizer.param_groups):
                g['lr'] = optimizer_init_lr_dict[g_idx] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        else:
            for g_idx, g in enumerate(optimizer.param_groups):
                base_lr = optimizer_init_lr_dict[g_idx]
                target_lr = base_lr * 1e-2
                rate = cur_nimg / (total_kimg * 1000)
                updated_lr = annealing_cos(base_lr, target_lr, rate)
                g['lr'] = updated_lr

        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # add gradient clipping to prevent unstable large gradients        
        clip_grad_norm_(net.parameters(), grad_clip)

        optimizer.step()

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        cur_lr = optimizer.param_groups[0]['lr']
                
        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        fields += [f"lr {training_stats.report0('lr', cur_lr)}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(net=net, loss_fn=loss_fn, \
                        dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                # Save torch state_dict instead of the persistent models w/ pickle...
                save_dict = {
                    'net': net.state_dict(),
                    'cur_nimg': cur_nimg,
                    'cur_tick': cur_tick,
                }
                save_path = os.path.join(run_dir, f'network-snapshot.pth')
                torch.save(save_dict, save_path)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            save_dict = {
                'net':net.state_dict(), 
                'optimizer_state': optimizer.state_dict(),
                'cur_nimg': cur_nimg,
                'cur_tick': cur_tick,
            }
            torch.save(save_dict, os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pth'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

def annealing_cos(start: float,
                  end: float,
                  factor: float,
                  weight: float = 1.) -> float:
    """Calculate annealing cos learning rate.
    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out
