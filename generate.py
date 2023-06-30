""" Inference of PolyDiffuse, adapted from the sampling code of the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist

import cv2
from src.polygon_utils import (
    visualize_results, 
    process_polygons,
    visualize_inter_results
)

from src.models.polygon_models.polygon_meta import PolyMetaModel


#----------------------------------------------------------------------------
# Likelihood evaluation for the EDM sampler...

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, mu_proposal, model_kwargs, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum((x - fn(x, t, mu_proposal, **model_kwargs)) / t * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        #x.requires_grad_(False)
        # TODO: apply the mask on the variables
        var_mask = (model_kwargs['attn_mask'] == 0)
        var_mask = var_mask[:, :, :, None]
        return torch.sum(grad_fn_eps * eps * var_mask, dim=tuple(range(1, len(x.shape))))

    return div_fn


def edm_likelihood(net, data, mu_guide, sigma_guide, model_kwargs,
                   randn_like=torch.randn_like,
                   hutchinson_type='Gaussian',
                   num_steps=18,
                   sigma_min=0.002,
                   sigma_max=80,
                   rho=7, second_order=True):

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    div_fn = get_div_fn(net)
    with torch.no_grad():
        if hutchinson_type == 'Gaussian':
            eps = randn_like(data)
        elif hutchinson_type == 'Rademacher':
            eps = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
        else:
            raise NotImplementedError

    step_indices = torch.arange(num_steps,
                                dtype=torch.float64,
                                device=data.device)
    t_steps = (sigma_max**(1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min**(1 / rho) - sigma_max**(1 / rho)))**rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps),
         torch.zeros_like(t_steps[:1])])
    t_steps_reverse = t_steps.flip(0)

    delta_logp = 0
    x_next = data

    t_steps_reverse[0] = t_steps_reverse[1] * 0.5

    # Go reversely
    for i, (t_hat, t_next) in enumerate(zip(t_steps_reverse[:-1],
                                     t_steps_reverse[1:])):
        # Euler step
        x_hat = x_next
        denoised = net(x_hat, t_hat, mu_guide, **model_kwargs).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        delta_logp_d_cur = div_fn(x_hat, t_hat, mu_guide, model_kwargs, eps)

        # Apply 2nd order correction.
        if second_order and i < num_steps - 1:
            denoised = net(x_next, t_next, mu_guide, **model_kwargs).to(torch.float64)
            d_prime = (x_next - denoised) / t_next

            delta_logp_d_prime = div_fn(x_next, t_next, mu_guide, model_kwargs, eps)

            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            delta_logp += (t_next - t_hat) * (0.5 * delta_logp_d_cur +
                                              0.5 * delta_logp_d_prime)
        else:
            delta_logp += (t_next - t_hat) * delta_logp_d_cur

    mask = model_kwargs['attn_mask'][0]
    elements_mask = (~mask).any(dim=-1)
    prior_logp = 0
    for idx in range(mask.shape[0]):
        if elements_mask[idx]:
            gauss_mean = mu_guide[0, idx, 0]
            gauss_sigma = sigma_guide[0, idx, 0]
            for coord_idx in range(2):
                norm_distribution = torch.distributions.Normal(gauss_mean[coord_idx].item(), gauss_sigma.item())
                data_to_eval = x_next[0, idx][mask[idx] == 0][:, coord_idx]
                prior_logp_element = torch.sum(norm_distribution.log_prob(data_to_eval),
                                    dim=tuple(torch.arange(1, len(data_to_eval.shape))))
                prior_logp += prior_logp_element
    
    total_logp = prior_logp + delta_logp
    num_vars = (mask==0).sum() * 2
    avg_logp = total_logp / num_vars
    nats = -avg_logp
    return nats

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

@torch.no_grad()
def edm_sampler(
    net, latents, mu_guide, model_kwargs, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, second_order=True
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    if num_steps == 1:
        t_hat = net.round_sigma(sigma_max).cuda()
        denoised = net(latents, t_hat, mu_guide, **model_kwargs).to(torch.float64)
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

        # Euler step.
        denoised = net(x_hat, t_hat, mu_guide, **model_kwargs).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if second_order and i < num_steps - 1:
            denoised = net(x_next, t_next, mu_guide, **model_kwargs).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        inter_results.append(x_next.cpu())

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
@click.option('--data_dir', 'data_dir',    help='Base directory of the data', metavar='PATH|URL',                      type=str, required=True)
@click.option('--init_dir', 'init_dir',    help='Directory of the initial results from propsoal generator', metavar='PATH|URL',      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--port',                    help='Manully set the port number', metavar='STR',                     type=str, default='12345', show_default=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--second_order',            help='using second-order solver or not', metavar='BOOL',                    type=bool, default=True, show_default=True)
@click.option('--compute_likelihood',      help='eval the likelihood or not', metavar='BOOL',                    type=bool, default=False, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--guide_ckpt',              help='Load the proposaal model', type=str)
@click.option('--proposal_type',           help='The type of the proposal generator', metavar='roomformer|rough_annot',   type=click.Choice(['roomformer', 'rough_annot']))


def main(network_pkl, port, data_dir, init_dir, outdir, seeds, max_batch_size, guide_ckpt, compute_likelihood, proposal_type, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
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

    dataset_kwargs = dnnlib.EasyDict(class_name='src.datasets.polygon_datasets.S3DPolygonDataset', 
                    data_dir=data_dir, init_dir=init_dir, split='test', rand_aug=False)

    network_kwargs = dnnlib.EasyDict()
    network_kwargs.update(model_type='PolyModel',)
    network_kwargs.update(num_poly=20, num_vert=40, input_dim=128, hidden_dim=256, num_feature_levels=4)
    network_kwargs.class_name = 'src.models.networks.EDMPrecond'
    network_kwargs.update(use_fp16=False)
    network_kwargs.update(sigma_data=1.0)

    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    data_loader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=max_batch_size)
    net = dnnlib.util.construct_class_by_name(**network_kwargs) # subclass of torch.nn.Module
    net = net.to(device)
    net.eval()

    net_guide = PolyMetaModel(input_dim=network_kwargs['input_dim'], embed_dim=network_kwargs['hidden_dim'])
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
    
    sigma_max = sampler_kwargs['sigma_max']
    c_in = 1 / np.sqrt(net.sigma_data ** 2 + sigma_max ** 2)

    for data in tqdm.tqdm(data_loader, disable=(dist.get_rank() != 0)):
        idx += 1
        torch.distributed.barrier()
        dist.print0(f'Generating No.{idx} example')
        gt_sample = data['polygon_verts'].to(device)
        roomformer_results = data['roomformer_results'].to(device)
        mimic_proposal_results = data['mimic_proposal_results'].to(device)

        if proposal_type == 'roomformer':
            # Results from an existing method (Roomformer) as the input to the guidance network 
            proposal_results = roomformer_results
        elif proposal_type == 'rough_annot':
            # Results from the mimic rough annotations as the input to the guidance network 
            proposal_results = mimic_proposal_results
        else:
            raise ValueError(f'Invalid proposal type {proposal_type}')

        rnd = StackedRandomGenerator(device, [42,] * proposal_results.shape[0])

        # Feeding the results from the proposal generator to the guidance network
        attn_mask = proposal_results[:, :, :, -1] != 1
        proposal_coords = proposal_results[:, :, :, :2]
        guide_mean, guide_sigma = net_guide(proposal_coords, attn_mask)
        guide_mean = guide_mean[:, :, None, :].repeat(1, 1, proposal_results.shape[2], 1)

        poly_mask = data['polygon_mask']
        model_kwargs = {
            'attn_mask': attn_mask,
            'image': data['image'].to(device),
            'poly_mask': poly_mask.to(device),
        }

        ## Option1: Init with random samples from per-element Gaussian
        #latents = guide_mean + guide_sigma[..., None] * torch.randn_like(guide_mean)
        ## Option2: Directly init with per-element Gaussian mean
        latents = guide_mean

        latents = latents.to(device)
        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        pred_polygons, inter_results = edm_sampler(net, latents, guide_mean, model_kwargs, randn_like=rnd.randn_like, **sampler_kwargs)

        if compute_likelihood:
            likelihood = edm_likelihood(net,
                                    pred_polygons,
                                    guide_mean,
                                    guide_sigma,
                                    model_kwargs,
                                    randn_like=rnd.randn_like,
                                    hutchinson_type='Gaussian',
                                    num_steps=sampler_kwargs['num_steps'],
                                    sigma_max=sampler_kwargs['sigma_max'],
                                    sigma_min=sampler_kwargs['sigma_min'],
                                    rho=sampler_kwargs['rho'],
                                    second_order=sampler_kwargs['second_order'])
            print('The sample likelihood is: {:.4f} bits/dim'.format(likelihood.item()))
            pred_polygons = pred_polygons.detach()

        pred_polygons, pred_poly_ids = process_polygons(pred_polygons[0], attn_mask[0])
        
        # process all inter-step results
        inter_polygons = [process_polygons(item[0], attn_mask[0], omit_small=False)[0] for item in inter_results]

        gt_polygons, _ = process_polygons(gt_sample[0, :, :, :2], (gt_sample[0, :, :, -1]==-1))

        density_path = data['density_path'][0]
        normal_path = data['normal_path'][0]
        scene = data['density_path'][0].split('/')[-2].split('_')[-1]
        density = cv2.imread(density_path)
        normal = cv2.imread(normal_path)
        viz_image = np.maximum(density, normal)

        # Visualize the results
        viz_path = os.path.join(viz_dir, 'scene_{}.png'.format(scene))
        visualize_results(pred_polygons, pred_poly_ids, gt_polygons, viz_image, viz_path)
        visualize_inter_results(inter_results, attn_mask[0], scene, viz_dir)

        save_path = os.path.join(save_dir, '{}.npy'.format(scene))
        save_path_inter = os.path.join(save_dir, '{}_inter.npy'.format(scene))
        np.save(save_path, np.array(pred_polygons, dtype=object))
        np.save(save_path_inter, np.array(inter_polygons, dtype=object))

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
