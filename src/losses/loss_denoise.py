import torch
from torch_utils import persistence
from einops import repeat, rearrange


#----------------------------------------------------------------------------
# Adapted from the loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, pc_range=None, lambda_dir=2e-3):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.lambda_dir = lambda_dir
        print(f'Loss: P_mean={P_mean}, P_std={P_std}, sigma_data={sigma_data}, lambda_dir={lambda_dir}')
        self.dir_interval = 1
        self.pc_range = pc_range

    def __call__(self, net, net_guide, images, **model_kwargs):
        rnd_normal = torch.randn([images.shape[0], 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = 1 # make the model direct estimate x0
        y = images

        attn_mask = model_kwargs['poly_mask'].clone()
        poly_class = model_kwargs['poly_class'].clone()
        mu_guide, sigma_guide = net_guide(images, attn_mask, poly_class)

        mu_guide = mu_guide[:, :, None, :]
        n = torch.randn_like(y) * sigma[..., None] * sigma_guide[:, :, None, :]
        noised_y = y + n
        # make sure that the indicator variable is set up correctly
        D_yn = net(noised_y, sigma, mu_guide, **model_kwargs)

        # only apply the loss to the valid vertices
        loss_mask = (model_kwargs['poly_mask'] == 0)

        # The l1-based manhattan-loss, following the original MapTR
        l1_loss_coords = (D_yn - y.unsqueeze(0)).abs().sum(-1)
        loss_coords = l1_loss_coords

        # The edge direction loss from the original MapTR
        denormed_pts_preds = denormalize_2d_pts(D_yn, self.pc_range)
        denormed_targets = denormalize_2d_pts(y, self.pc_range)
        denormed_targets = repeat(denormed_targets, 'b n1 n2 c -> n0 b n1 n2 c', n0=D_yn.shape[0])
        pts_preds_dir = denormed_pts_preds[...,self.dir_interval:,:] - denormed_pts_preds[...,:-self.dir_interval,:]
        pts_targets_dir = denormed_targets[..., self.dir_interval:,:] - denormed_targets[...,:-self.dir_interval,:]
        pts_preds_dir = rearrange(pts_preds_dir, 'n0 b n1 n2 c -> (n0 b n1) n2 c')
        pts_targets_dir = rearrange(pts_targets_dir, 'n0 b n1 n2 c -> (n0 b n1) n2 c')
        loss_dir = pts_dir_cos_loss(pts_preds_dir, pts_targets_dir)
        loss_dir = rearrange(loss_dir, '(n0 b n1) n2 -> n0 b n1 n2', n0=D_yn.shape[0], b=images.shape[0])
        loss_dir = torch.cat([loss_dir, loss_dir[..., -1:]], dim=-1)

        loss_dir = self.lambda_dir * loss_dir
        loss_all = loss_coords + loss_dir
        loss = weight * loss_all

        # get the last-layer losses for logging
        loss_coords_last = loss_coords[-1]
        loss_dir_last = loss_dir[-1]
        last_layer_loss = loss[-1]
        last_layer_loss = last_layer_loss[loss_mask]
        full_mask = repeat(loss_mask, 'b n1 n2 -> n0 b n1 n2', n0=D_yn.shape[0])
        loss = loss[full_mask]
        loss_coords = loss_coords[full_mask]
        loss_dir = loss_dir[full_mask]
        last_layer_coords_loss = loss_coords_last[loss_mask]
        last_layer_dir_loss = loss_dir_last[loss_mask]

        return loss, last_layer_loss, loss_coords, loss_dir, last_layer_coords_loss, last_layer_dir_loss
        
#----------------------------------------------------------------------------

def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] * pc_range[3]
    new_pts[...,1:2] = pts[...,1:2] * pc_range[4]
    return new_pts


def pts_dir_cos_loss(pred, target):
    """ Dir cosine similiarity loss
    pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
    target (torch.Tensor): shape [num_samples, num_dir, num_coords]
    """
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0,1), target.flatten(0,1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss
