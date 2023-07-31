"""
The loss function for training the guidance networks of the Guided Set Diffuison Models
"""

import torch
from torch_utils import persistence
import torch.nn as nn


@persistence.persistent_class
class GuideLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        print(f'Loss: P_mean={P_mean}, P_std={P_std}, sigma_data={sigma_data}')
        self.poly_matching_loss = PolygonMatchingLoss()

    def __call__(self, net, net_guide, images, **model_kwargs):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        y = images

        attn_mask = model_kwargs['poly_mask'].clone()
        poly_class = model_kwargs['poly_class'].clone()
        mu_guide, sigma_guide = net_guide(images, attn_mask, poly_class)

        mu_guide = mu_guide[:, :, None, :]
        n = torch.randn_like(y) * sigma * sigma_guide[:, :, None, :]
        noised_y = y + n
        # make sure that the indicator variable is set up correctly
        x_t = net.get_model_input(noised_y, sigma, mu_guide)

        loss, perm_loss, reg_loss, sigma_loss, broken_status = self.poly_matching_loss(y, x_t, 
                                                                        mu_guide, sigma_guide, 
                                                                        model_kwargs['poly_mask'], 
                                                                        model_kwargs['poly_class'])
        return loss, perm_loss, reg_loss, sigma_loss, broken_status, mu_guide, sigma_guide


#----------------------------------------------------------------------------


class PolygonMatchingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(PolygonMatchingLoss, self).__init__()
        self.margin = margin
        self.lambda_perm = 1.0
        self.lambda_reg = 0.1
        self.lambda_sigma = 0.05

    def forward(self, all_gt_polys, all_pred_polys, mu_pred, sigma_pred, all_poly_mask, all_poly_class):
        bs = all_poly_mask.shape[0]
        all_num_gt_polys = (all_poly_mask==0).any(-1).sum(-1)

        perm_losses = []
        reg_losses = []
        sigma_losses = []
        all_broken_status = []
        
        for b_i in range(bs):
            num_polys = all_num_gt_polys[b_i]
            gt_polys = all_gt_polys[b_i, :num_polys]
            pred_polys = all_pred_polys[b_i, :num_polys]
            pred_mu = mu_pred[b_i, :num_polys]
            pred_sigma = sigma_pred[b_i, :num_polys]
            poly_mask = all_poly_mask[b_i, :num_polys]
            poly_class = all_poly_class[b_i, :num_polys]
            
            # The permutation loss
            costs = []
            for gt_i in range(num_polys):
                cost_i = self._polygon_dist(gt_polys[gt_i], poly_class[gt_i], pred_polys, poly_mask, gt_i)
                costs.append(cost_i)
            costs = torch.stack(costs, dim=0)
            perm_loss, broken = self.compute_triplet_loss(costs)
            perm_losses.append(perm_loss)
            all_broken_status.append(broken)

            sigma_loss = 1 / pred_sigma
            sigma_loss = sigma_loss.mean()

            reg_loss = (pred_mu ** 2).sum(-1)
            reg_loss = reg_loss.mean()

            reg_losses.append(reg_loss)
            sigma_losses.append(sigma_loss)

        perm_losses = torch.stack(perm_losses, dim=0)
        reg_losses = torch.stack(reg_losses, dim=0)        
        sigma_losses = torch.stack(sigma_losses, dim=0)        
        all_broken_status = torch.stack(all_broken_status, dim=0)

        batch_losses = self.lambda_reg * reg_losses + self.lambda_sigma * sigma_losses \
                    + perm_losses * self.lambda_perm

        return batch_losses, perm_losses, reg_losses, sigma_losses, all_broken_status
    
    def compute_triplet_loss(self, costs):
        diagonal = costs.diag().view(costs.size(0), 1)
        d1 = diagonal.expand_as(costs)
        d2 = diagonal.t().expand_as(costs)

        cost_row = (self.margin - costs + d1).clamp(min=0)
        cost_col = (self.margin - costs + d2).clamp(min=0)
        I = (torch.eye(costs.size(0)) > .5).to(costs.device)
        cost_row = cost_row.masked_fill_(I, 0)
        cost_col = cost_col.masked_fill_(I, 0)

        loss_row = cost_row.max(1)[0]
        loss_col = cost_col.max(0)[0]
        triplet_loss = loss_row.sum() + loss_col.sum()
        all_broken_row = (costs < d1).masked_fill_(I, 0)
        broken = (all_broken_row & all_broken_row.T).any()
        return triplet_loss, broken

    def _polygon_dist(self, target, target_class, preds, vert_mask, gt_index):
        num_valid_vert_target = (vert_mask[gt_index] == 0).sum()
        if target_class == 1: # is polygon (ped crossing)
            rotated_targets = self._rotate_target(target, num_valid_vert_target)
        else:
            rotated_targets = self._rotate_target_line(target, num_valid_vert_target)

        coord_costs = []
        for pred_id in range(preds.shape[0]):
            num_valid_vert_pred = (vert_mask[pred_id] == 0).sum()
            num_valid = torch.min(num_valid_vert_pred, num_valid_vert_target)
            preds_valid = preds[pred_id, :num_valid, :]
            target_valid = rotated_targets[:, :num_valid, :]
            coord_l1 = torch.abs(target_valid - preds_valid[None, :, :])
            coord_cost = coord_l1.mean(-1).mean(-1)
            coord_costs.append(coord_cost)
        coord_costs = torch.stack(coord_costs, dim=1)
        min_cost_per_poly, target_indices = coord_costs.min(dim=0)
        return min_cost_per_poly

    @staticmethod
    def _rotate_target(target, num_valid_vert):
        all_rotated_targets = []
        for start_idx in range(num_valid_vert):
            rotated_target = target[:num_valid_vert].roll(-start_idx, 0)
            all_rotated_targets.append(rotated_target)
        rotated_targets = torch.stack(all_rotated_targets, dim=0)
        zero_paddings = torch.zeros(rotated_targets.shape[0], target.shape[0] - num_valid_vert, 
                                    rotated_targets.shape[2]).to(rotated_targets.device)
        rotated_targets = torch.cat([rotated_targets, zero_paddings], dim=1)
        return rotated_targets
    
    @staticmethod
    def _rotate_target_line(target, num_valid_vert):
        reversed_target = torch.flip(target, dims=[0,])
        rotated_targets = torch.stack([target, reversed_target], dim=0)
        return rotated_targets
