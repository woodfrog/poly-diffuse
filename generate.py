"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".
Adapted for PolyDiffuse
"""

import os
import re
import click
import tqdm
import numpy as np
import torch
from PIL import Image
import src.dnnlib as dnnlib
from torch_utils import distributed as dist

from src.dnnlib.hdmap_utils import (
    plot_map,
    visualize_all_iter,
    preprocess_init_result,
    compute_samplewise_metric,
)

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv import Config, DictAction
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.nuscenes_map_dataset import polygon_collate
import importlib

import copy
import json
import shutil
import cv2

from src.models.polygon_models.polygon_meta import PolyMetaModel

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

@torch.no_grad()
def edm_sampler(net, latents, mu_guide, model_kwargs, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, second_order=True,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if num_steps == 1:
        t_hat = net.round_sigma(sigma_max).cuda()
        denoised = net(latents, t_hat, mu_guide, **model_kwargs).to(torch.float64)
        denoised = denoised[-1]
        return denoised, [denoised, ]

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    t_steps[-1] = t_steps[-2] * 0.5

    # Main sampling loop.
    #x_next = latents.to(torch.float64) * t_steps[0]
    x_next = latents.to(torch.float64)
    inter_results = []
    inter_results.append(x_next.cpu())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        if i == 0:  # cache image features in the first step
            model_kwargs['cache_image_feat'] = True
            model_kwargs['use_cached_feat'] = False
        else: # use the cached image features
            model_kwargs['cache_image_feat'] = False
            model_kwargs['use_cached_feat'] = True
            
        # Euler step.
        denoised = net(x_hat, t_hat, mu_guide, **model_kwargs).to(torch.float64)
        denoised = denoised[-1]
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if second_order and i < num_steps - 1:
            model_kwargs['cache_image_feat'] = False
            model_kwargs['use_cached_feat'] = True
            denoised = net(x_next, t_next, mu_guide, **model_kwargs).to(torch.float64)
            denoised = denoised[-1]
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        inter_results.append(x_next.cpu())

    # clear the cache after finishing processing this sample
    net.model.clear_cache()
    return x_next, inter_results

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--config_path',                  help='Path to the config file', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--port',                    help='Manully set the port number', metavar='STR',                     type=str, default='12345', show_default=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers',                 help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=0), default=1, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--second_order',            help='using second-order solver or not', metavar='BOOL',                 type=bool, default=True, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--guide_ckpt',              help='Load the guidance model', type=str)
@click.option('--init_path',               help='Path of the results from the init proposal generator', type=str)
@click.option('--viz_results',             help='Visualize all steps with a gif', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--viz_gif',                 help='Visualize all steps with a gif', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--proposal_type',           help='The proposal initialization results to use ', type=click.Choice(['rough_annot', 'maptr']))


def main(network_pkl, config_path, port, outdir, seeds, max_batch_size, workers, guide_ckpt, proposal_type, init_path, viz_results, viz_gif, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init(manual_port=port)
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    viz_dir = os.path.join(outdir, 'viz')
    save_dir = os.path.join(outdir, 'npy')
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    
    cfg = Config.fromfile(config_path)
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
    
    # get pc_range
    pc_range = cfg.point_cloud_range
    # get car icon
    car_img = Image.open('./assets/imgs/lidar_car.png')
    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['orange', 'b', 'g']

    network_kwargs = dnnlib.EasyDict()
    network_kwargs.update(model_type='maptr',)
    network_kwargs.class_name = 'src.models.networks.EDMPrecond'
    network_kwargs.update(use_fp16=False)
    network_kwargs.update(sigma_data=1.0)

    ##-------------------------------------------------
    # build the dataset and MapTR model (adapted from MapTR codebase)
    cfg.data.test['load_init'] = True
    cfg.data.test['init_results_path'] = init_path
    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=max_batch_size,
                collate_fn=polygon_collate, num_workers=workers)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ##-------------------------------------------------

    network_kwargs['model'] = model
    net = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    net = net.to(device)
    net.eval()

    pe_dim = net.model.pts_bbox_head.positional_encoding.num_feats
    embed_dim = net.model.pts_bbox_head.transformer.embed_dims
    net_guide = PolyMetaModel(input_dim=pe_dim, embed_dim=embed_dim)
    net_guide.eval()
    guide_ckpt = torch.load(guide_ckpt, map_location='cpu')
    net_guide.load_state_dict(guide_ckpt['net'])
    net_guide.to(device)
    dist.print0('Loading the guidance network')

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    ckpt = torch.load(network_pkl)
    net.load_state_dict(ckpt['net'])
    print(f'Loaded network ckpt from {network_pkl}')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    # Loop over batches.
    dist.print0(f'Generating {len(data_loader)} images to "{outdir}"...')
    idx = 0

    # copy the init results, update the results with poly-diffuse outputs for evaluation
    all_results = copy.deepcopy(dataset.init_results)

    if viz_results:
        eval_records = list()

    num_steps = sampler_kwargs['num_steps']
    for data_batch in tqdm.tqdm(data_loader, disable=(dist.get_rank() != 0)):
        idx += 1
        torch.distributed.barrier()
        
        gt_pts = data_batch['gt_bboxes_3d'].to(device)
        rough_annot = data_batch['rough_annot'].to(device)
        attn_mask = data_batch['pts_mask'].to(device)
        model_kwargs = {
            'img': data_batch['img'].to(device),
            'poly_class': data_batch['gt_labels_3d'].to(device),
            'poly_mask': attn_mask.to(device),
            'img_metas': data_batch['img_metas']
        }

        gt_labels_3d = data_batch['gt_labels_3d']
        sigma_max = sampler_kwargs['sigma_max']

        c_in = 1 / np.sqrt(net.sigma_data ** 2 + sigma_max ** 2)
        if proposal_type == 'maptr': # Init with MapTR results
            init_result = copy.deepcopy(data_batch['init_result'][0]['vectors'])

            init_pred_pts, init_pred_labels, init_pred_conf, processed_items, remaining_items = \
                    preprocess_init_result(init_result, pc_range=pc_range, conf_thresh=0.5)
        
            if init_pred_pts is None: 
            # init result is empty, skip as no result to refine
                if viz_results:
                    eval_entry = {
                        'num_pred': pred_pts.shape[1],
                        'num_gt': gt_pts.shape[1],
                        'init': None,
                        'deformed': None,
                    }
                    eval_records.append(eval_entry)
                continue

            init_pts = torch.from_numpy(init_pred_pts).to(device).unsqueeze(0)
            init_attn_mask = torch.zeros_like(init_pts)[:, :, :, 0].bool()
            init_poly_class = torch.from_numpy(init_pred_labels).to(device).unsqueeze(0).long()
            model_kwargs['poly_mask'] = init_attn_mask
            model_kwargs['poly_class'] = init_poly_class

            guide_mean, guide_sigma = net_guide(init_pts, 
                            init_attn_mask.clone(), model_kwargs['poly_class'])
            guide_mean = guide_mean[:, :, None, :].repeat(1, 1, init_pts.shape[2], 1)

            # Start with an interpolation of proposal results and the guidance mean
            # depending on the c_in (a small value)
            c_in = torch.tensor(c_in).to(guide_mean.device)
            guide_mean = c_in * init_pts + (1 - c_in) * guide_mean

            pred_labels = init_poly_class
            pred_confs = init_pred_conf 
        else:  
            # Init with rough annot (derived from G.T.) for mu_meta (for mimicking human inputs)
            guide_mean, guide_sigma = net_guide(rough_annot, 
                            attn_mask.clone(), model_kwargs['poly_class'])
            
            guide_mean = guide_mean[:, :, None, :].repeat(1, 1, gt_pts.shape[2], 1)

            c_in = torch.tensor(c_in).to(guide_mean.device)
            guide_mean = c_in * rough_annot + (1 - c_in) * guide_mean

            pred_labels = data_batch['gt_labels_3d']
            pred_confs = torch.ones(gt_pts.shape[1]) * 0.99

        latents = guide_mean

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        pred_polygons, inter_results = edm_sampler(net, latents, guide_mean, model_kwargs, **sampler_kwargs)
        pred_pts = pred_polygons.cpu()

        # Compute the per-sample metric
        if viz_results:
            prec, recall, cls_results, matching, matching_scores, matching_angles = \
                compute_samplewise_metric(pred_pts[0], pred_labels[0], pred_confs, gt_pts[0], gt_labels_3d[0], pc_range, threshold=0.5)
            eval_entry = {
                'deformed': {'prec': prec, 'recall': recall, 'num_hit': len(matching),
                            'cls_results': cls_results},
                'num_pred': pred_pts.shape[1],
                'num_gt': gt_pts.shape[1],
            }

            if proposal_type == 'maptr':
                prec_init, recall_init, cls_results_init, matching_init, matching_scores_init, \
                    matching_angles_init = compute_samplewise_metric(init_pred_pts, init_pred_labels, init_pred_conf, gt_pts[0], gt_labels_3d[0], pc_range, threshold=0.5)
                eval_entry['init'] = {'prec':prec_init, 'recall':recall_init, 
                    'num_hit': len(matching_init), 'cls_results': cls_results_init}

            eval_records.append(eval_entry)

            # viz G.T. map
            gt_fixedpts_map_path = os.path.join(viz_dir, '{}_GT_fixednum_pts_MAP.png'.format(idx))
            plot_map(gt_pts[0].cpu().numpy(), gt_labels_3d[0],
                        gt_fixedpts_map_path, pc_range, colors_plt, car_img, viz_end=False)

            # viz pred map
            if proposal_type == 'maptr':
                pred_map_path = os.path.join(viz_dir, '{}_init_maptr_polydiffuse_deformed_step_{}.png'.format(idx, num_steps))
            else:
                pred_map_path = os.path.join(viz_dir, '{}_init_rough_annot_polydiffuse_deformed_step_{}.png'.format(idx, num_steps))

            plot_map(pred_pts[0].cpu().numpy(), pred_labels[0],
                        pred_map_path, pc_range, colors_plt, car_img, 
                        matching=matching, matching_scores=matching_scores,
                        matching_angles=matching_angles, prec=prec, recall=recall,
                        viz_score=False, viz_end=False)
                
        if proposal_type == 'maptr':
            if viz_results:
                # viz init result from MapTR (Res50, 110epc)
                pred_map_path = os.path.join(viz_dir, '{}_maptr_pred.png'.format(idx))
                plot_map(init_pred_pts, init_pred_labels, 
                            pred_map_path, pc_range, colors_plt, car_img, matching=matching_init,
                            matching_scores=matching_scores_init, matching_angles=matching_angles_init,
                            prec=prec_init, recall=recall_init,
                            viz_score=False, viz_end=False)
            
            # Update the json results for final quantitative evaluations
            for item_idx, item in enumerate(processed_items):
                new_pts = pred_pts[0][item_idx]
                new_pts[:, 0] *= pc_range[3]
                new_pts[:, 1] *= pc_range[4]
                new_pts = new_pts.tolist()
                item['pts'] = new_pts
            all_items = processed_items + remaining_items
            all_results[idx-1]['vectors'] = all_items
        else:
            if viz_results:
                pred_map_path = os.path.join(viz_dir, '{}_rough_annot.png'.format(idx))
                plot_map(rough_annot[0].cpu().numpy(), gt_labels_3d[0], 
                            pred_map_path, pc_range, colors_plt, car_img, viz_score=False, viz_end=False)

            rough_annot_items = []
            for pred_idx, pred in enumerate(pred_pts[0]):
                new_pts = pred.clone()
                new_pts[:, 0] *= pc_range[3]
                new_pts[:, 1] *= pc_range[4]
                new_pts = new_pts.tolist()
                cls_label = pred_labels[0, pred_idx]
                if cls_label == 0:
                    label_name = 'divider'
                elif cls_label == 1:
                    label_name = 'ped_crossing'
                else:
                    label_name = 'boundary'
                item = dict()
                item['pts'] = new_pts
                item['cls_name'] = label_name
                item['pts_num'] = 20
                item['type'] = cls_label.tolist()
                item['confidence_level'] = 0.99
                rough_annot_items.append(item)
            all_results[idx-1]['vectors'] = rough_annot_items

        if viz_results and viz_gif: # visualize all intermediate steps of the inference
            all_preds = [item[0].cpu().numpy() for item in inter_results]
            all_iter_path = os.path.join(viz_dir, '{}_polydiffuse_all_iter.gif'.format(idx))
            visualize_all_iter(all_preds, pred_labels[0], 
                        all_iter_path, pc_range, colors_plt, car_img)
    
    # Write the poly-diffsue deformed results for final quantitative evaluation
    updated_results = {
        'meta': dataset.init_meta,
        'results': all_results,
    }
    results_path = os.path.join(save_dir, 'deformed_results.json')
    with open(results_path, 'w') as f_out:
        json.dump(updated_results, f_out)
    
    if viz_results:
        eval_results_path = os.path.join(save_dir, 'samplewise_eval_results.json')
        with open(eval_results_path, 'w') as f_out:
            json.dump(eval_records, f_out)
        
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')


#----------------------------------------------------

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]

def save_surr_images(filenames, out_dir, idx):
    img_path_dict = {}
    # save cam img for sample
    for filepath in filenames:
        filename = os.path.basename(filepath)
        filename_splits = filename.split('__')
        img_name = filename_splits[1] + '.jpg'
        img_path = os.path.join(out_dir,img_name)
        # img_path_list.append(img_path)
        shutil.copyfile(filepath,img_path)
        img_path_dict[filename_splits[1]] = img_path
    # surrounding view
    row_1_list = []
    for cam in CAMS[:3]:
        cam_img_name = cam + '.jpg'
        cam_img = cv2.imread(os.path.join(out_dir, cam_img_name))
        row_1_list.append(cam_img)
    row_2_list = []
    for cam in CAMS[3:6]:
        cam_img_name = cam + '.jpg'
        cam_img = cv2.imread(os.path.join(out_dir, cam_img_name))
        row_2_list.append(cam_img)
    row_1_img=cv2.hconcat(row_1_list)
    row_2_img=cv2.hconcat(row_2_list)
    cams_img = cv2.vconcat([row_1_img,row_2_img])
    cams_img_path = os.path.join(out_dir, '{}_surroud_view.jpg'.format(idx))
    cv2.imwrite(cams_img_path, cams_img,[cv2.IMWRITE_JPEG_QUALITY, 70])
    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
