"""training loop for the guidance network"""
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


#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    config_path         = '',       # MapTR config path
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
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # load the config file
    cfg = Config.fromfile(config_path)
    cfg.gpu_ids = range(1)
    cfg.seed = 1234

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

    cfg.data.train['drop_instance'] = False
    cfg.data.train['load_image'] = False # do not load image data for guidance training, to save data loading time
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
    network_kwargs['model'] = model

    net = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(False).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')

    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss

    pe_dim = net.model.pts_bbox_head.positional_encoding.num_feats
    embed_dim = net.model.pts_bbox_head.transformer.embed_dims
    net_guide = PolyMetaModel(input_dim=pe_dim, embed_dim=embed_dim)
    net_guide.train().requires_grad_(True).to(device)
    optimizer = torch.optim.AdamW(net_guide.parameters(), lr=optimizer_kwargs['lr'], weight_decay=1e-4)
    ddp_guide = torch.nn.parallel.DistributedDataParallel(net_guide, device_ids=[device], broadcast_buffers=False)

    # Directory to save the visualization results
    viz_dir = './viz_guide_hdmap'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # Resume training from previous snapshot.
    if resume_pkl:
        if resume_state_dump is not None:
            dist.print0(f'Loading training state from "{resume_state_dump}"...')
            data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
            net_guide.load_state_dict(data['net'])
            optimizer.load_state_dict(data['optimizer_state'])
            del data # conserve memory
        else:
            dist.print0(f'Loading network weights from "{resume_pkl}"...')
            if dist.get_rank() != 0:
                torch.distributed.barrier() # rank 0 goes first
            data = torch.load(resume_pkl)
            if dist.get_rank() == 0:
                torch.distributed.barrier() # other ranks follow
            net_guide.load_state_dict(data['net'])
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
    lr_decay = False

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_guide, (round_idx == num_accumulation_rounds - 1)):
                data_batch = next(dataset_iterator)
                # abuse the variable name "images" here for the data
                images = data_batch['gt_bboxes_3d'].to(device)
                attn_mask = data_batch['pts_mask'].to(device)

                model_kwargs = {
                    'poly_class': data_batch['gt_labels_3d'].to(device),
                    'poly_mask': attn_mask.to(device),
                }

                loss, perm_loss, reg_loss, sigma_loss, broken_status, \
                            mu_guide, sigma_guide = loss_fn(net=net, net_guide=ddp_guide, images=images, **model_kwargs)
                
                training_stats.report('Loss/loss', loss)
                training_stats.report('Loss/loss_perm', perm_loss)
                training_stats.report('Loss/loss_reg', reg_loss)
                training_stats.report('Loss/loss_sigma', sigma_loss)

                effective_poly_mask = (attn_mask==0).any(-1)
                effective_sigma = sigma_guide[effective_poly_mask==1]
                training_stats.report('Sigma pred', effective_sigma)
                
                broken_rate = broken_status.sum() / broken_status.shape[0]
                training_stats.report('Broken rate', broken_rate)

                loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                # FPN has some ununsed BN parameters, 
                # This is a workaround for DDP's error regarding those unused parameters
                pseudo_losses = [p.sum() * 0 for p in ddp_guide.parameters()]
                pseudo_losses = torch.stack(pseudo_losses).mean()
                loss += pseudo_losses    
                loss.backward()
        
        # learning rate warmup
        if cur_nimg <= lr_rampup_kimg * 1000:
            for g in optimizer.param_groups:
                g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)

        # learning rate decay
        if not lr_decay and cur_nimg >= int(total_kimg * 1000 * 0.8):
            lr_decay = True
            dist.print0('Learning rate decayed by 10')
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.1

        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
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
            data = dict(net=net_guide, loss_fn=loss_fn, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                # Save torch state_dict instead of the persistent models w/ pickle...
                save_dict = {
                    'net': net_guide.state_dict(),
                    'cur_nimg': cur_nimg,
                    'cur_tick': cur_tick,
                }
                save_path = os.path.join(run_dir, f'network-snapshot.pth')
                torch.save(save_dict, save_path)
            del data # conserve memory

            # visualize the predicted guidances
            visualize_guides(images, mu_guide, sigma_guide, attn_mask, model_kwargs['poly_class'], viz_dir)

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            save_dict = {
                'net':net_guide.state_dict(), 
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


##----------------------------------------------------------------------------
# visualize the guidance
##----------------------------------------------------------------------------

def visualize_guides(x0, mu, sigma, mask, labels, viz_dir):
    x0 = x0.cpu().numpy()
    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()
    mask = mask.cpu().numpy()
    labels = labels.cpu().numpy()
    for sample_i in range(x0.shape[0]):
        sample_mask = mask[sample_i]
        num_poly = (sample_mask==0).any(-1).sum()
        gt_poly = x0[sample_i, :num_poly]
        pred_guides = mu[sample_i, :num_poly]
        pred_sigma = sigma[sample_i]
        sample_labels = labels[sample_i, :num_poly]

        gt_path = os.path.join(viz_dir, '{}_gt.png'.format(sample_i))
        guide_path = os.path.join(viz_dir, '{}_guide.png'.format(sample_i))

        # viz G.T.
        gt_image = plot_map(gt_poly, sample_labels, 
                    pc_range, colors_plt, car_img)

        # viz pred
        guide_image = plot_map(pred_guides, sample_labels, 
                    pc_range, colors_plt, car_img)

        effective_sigma = pred_sigma[:num_poly]
        cv2.putText(guide_image, r'sig_ma={:.2f},sig_mi={:.2f}'.format(effective_sigma.max(), effective_sigma
        .min()), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (255, 255, 0), 2, cv2.LINE_AA)

        imageio.imsave(gt_path, gt_image)
        imageio.imsave(guide_path, guide_image)


from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imageio
import io

# get pc_range
pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
# get car icon
car_img = Image.open('./assets/imgs/lidar_car.png')
# get color map: divider->r, ped->b, boundary->g
colors_plt = ['orange', 'b', 'g']

def plot_map(viz_pts, labels, pc_range, colors_plt, car_img):
    plt.figure(figsize=(2, 4))
    plt.xlim(pc_range[0], pc_range[3])
    plt.ylim(pc_range[1], pc_range[4])
    plt.axis('off')
    for gt_bbox_3d, gt_label_3d in zip(viz_pts, labels):
        pts = gt_bbox_3d
        pts[:, 0] *= pc_range[3]
        pts[:, 1] *= pc_range[4]
        x = np.array([pt[0] for pt in pts])
        y = np.array([pt[1] for pt in pts])
        
        plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
        plt.scatter(x[1:-1], y[1:-1], color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
        plt.scatter(x[0:1], y[0:1], color='red',s=2,alpha=0.8,zorder=-1)
        plt.scatter(x[-1:], y[-1:], color='black',s=2,alpha=0.8,zorder=-1)
    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches='tight', format='png', dpi=200)
    plt.close()
    viz_image = np.array(Image.open(img_buf))
    return viz_image
