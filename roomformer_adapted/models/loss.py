import torch
from torch import nn
import torch.nn.functional as F
from utils.geometry_utils import edge_acc
from scipy.optimize import linear_sum_assignment
import numpy as np

import os
import cv2
import imageio


class CornerCriterion(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.loss_rate = 9

    def forward(self, outputs_s1, targets, gauss_targets, epoch=0):
        # Compute the acc first, use the acc to guide the setup of loss weight
        preds_s1 = (outputs_s1 >= 0.5).float()
        pos_target_ids = torch.where(targets == 1)
        correct = (preds_s1[pos_target_ids] == targets[pos_target_ids]).float().sum()
        recall_s1 = correct / len(pos_target_ids[0])

        rate = self.loss_rate

        loss_weight = (gauss_targets > 0.5).float() * rate + 1
        loss_s1 = F.binary_cross_entropy(outputs_s1, gauss_targets, weight=loss_weight, reduction='none')
        loss_s1 = loss_s1.sum(-1).sum(-1).mean()

        return loss_s1, recall_s1


class EdgeCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.33, 1.0]).cuda(), reduction='none')

    def forward(self, logits_s1, logits_s2_hybrid, logits_s2_rel, s2_ids, s2_edge_mask, edge_labels, edge_lengths,
                edge_mask, s2_gt_values):
        # loss for edge filtering
        s1_losses = self.edge_loss(logits_s1, edge_labels)
        s1_losses[torch.where(edge_mask == True)] = 0
        s1_losses = s1_losses[torch.where(s1_losses > 0)].sum() / edge_mask.shape[0]
        gt_values = torch.ones_like(edge_mask).long() * 2
        s1_acc = edge_acc(logits_s1, edge_labels, edge_lengths, gt_values)

        # loss for stage-2
        s2_labels = torch.gather(edge_labels, 1, s2_ids)

        # the image-aware decoder
        s2_losses_hybrid = self.edge_loss(logits_s2_hybrid, s2_labels)
        s2_losses_hybrid[torch.where((s2_edge_mask == True) | (s2_gt_values != 2))] = 0
        # aggregate the loss into the final scalar
        s2_losses_hybrid = s2_losses_hybrid[torch.where(s2_losses_hybrid > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        # compute edge-level acc
        s2_acc_hybrid = edge_acc(logits_s2_hybrid, s2_labels, s2_edge_lengths, s2_gt_values)

        # the geom-only decoder
        s2_losses_rel = self.edge_loss(logits_s2_rel, s2_labels)
        s2_losses_rel[torch.where((s2_edge_mask == True) | (s2_gt_values != 2))] = 0
        # aggregate the loss into the final scalar
        s2_losses_rel = s2_losses_rel[torch.where(s2_losses_rel > 0)].sum() / s2_edge_mask.shape[0]
        s2_edge_lengths = (s2_edge_mask == 0).sum(dim=-1)
        # compute edge-level f1-score
        s2_acc_rel = edge_acc(logits_s2_rel, s2_labels, s2_edge_lengths, s2_gt_values)

        return s1_losses, s1_acc, s2_losses_hybrid, s2_acc_hybrid, s2_losses_rel, s2_acc_rel


class RoomCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.matching_lambda_coord = 2.5
        self.cost_lambda_coord = 2.5
        #self.exs_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.33, 1.0]).cuda(), reduction='none')
        self.exs_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).cuda(), reduction='none')
    
    def forward(self, gt_polys, merged_pred_polys, pred_coords, pred_logits, num_gt_polys):
        matches = self.polygon_matching(gt_polys, merged_pred_polys, num_gt_polys)
        permuted_gt_polys, match_mask = self.permute_targets_with_matches(gt_polys, matches)

        # To check the matching (debugging only)
        #self._check_matching(matches, pred_coords, permuted_gt_polys)
        #print(pred_coords.mean(-1).mean(-1))

        gt_exs = permuted_gt_polys[:, :, :, -1].long()
        num_verts = gt_exs.sum(-1)
        pred_logits = pred_logits.permute(0, 3, 1, 2)
        exs_loss = self.exs_loss(pred_logits, gt_exs)

        # scale the per-polygon loss based on the number of vertices
        poly_weights = torch.ones_like(num_verts).double()
        pos_poly_ids = torch.where(num_verts > 0)
        #poly_weights[pos_poly_ids] = num_verts[pos_poly_ids].double() / 4
        #agg_exs_loss = (exs_loss * poly_weights[:, :, None]).mean()
        agg_exs_loss = exs_loss.mean()

        coord_diff = pred_coords - permuted_gt_polys[:, :, :, :2]
        #coord_diff = coord_diff * poly_weights[:, :, None, None]
        coord_loss = torch.abs(coord_diff)[match_mask==1] # l1 coord loss
        #coord_loss = (coord_diff ** 2)[match_mask==1] # l2 coord loss
        agg_coord_loss = coord_loss.sum(-1).mean()

        polygon_loss = agg_exs_loss + self.cost_lambda_coord * agg_coord_loss
        loss = {
            'loss': polygon_loss,
            'exs_loss': agg_exs_loss,
            'coord_loss': agg_coord_loss,
        }

        return matches, loss


    def polygon_matching(self, all_gt_polys, all_pred_polys, all_num_gt_polys):
        bs = all_num_gt_polys.shape[0]
        all_matches = []
        for b_i in range(bs):
            gt_polys = all_gt_polys[b_i]
            pred_polys = all_pred_polys[b_i]
            num_polys = all_num_gt_polys[b_i]
            all_dists = []
            all_target_shifts = []
            for gt_i in range(num_polys):
                dist_i, target_shifts = self._polygon_dist(gt_polys[gt_i], pred_polys)
                all_dists.append(dist_i)
                all_target_shifts.append(target_shifts)
            all_dists = torch.stack(all_dists, dim=0).permute(1, 0)
            all_target_shifts = torch.stack(all_target_shifts, dim=0).permute(1, 0)
            all_dists = all_dists.cpu().detach().numpy()
            all_dists = np.nan_to_num(all_dists, nan=1e5)
            indices = linear_sum_assignment(all_dists)
            indices = [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]
            target_shifts = all_target_shifts[indices[0], indices[1]]
            indices.append(target_shifts)
            all_matches.append(indices)

        return all_matches
            
    def _polygon_dist(self, target, preds):
        rotated_targets, num_valid_vert = self._rotate_target(target)

        cls_cost = torch.abs(rotated_targets[:, None, :, -1] - preds[None, :, :, -1]).sum(-1)

        preds_valid = preds[:, :num_valid_vert, :]
        target_valid = rotated_targets[:, :num_valid_vert, :]
        coord_l1 = torch.abs(target_valid[:, None, :, :2] - preds_valid[None, :, :, :2])
        coord_cost = coord_l1.mean(-1).sum(-1)
        total_cost = cls_cost + self.matching_lambda_coord * coord_cost

        min_cost, target_indices = total_cost.min(dim=0)
        return min_cost, target_indices

    def permute_targets_with_matches(self, targets, matches):
        bs = targets.shape[0]

        all_permute_targets = []
        all_match_mask = []
        for b_i in range(bs):
            targets_i = targets[b_i]
            matches_i = matches[b_i]

            tgt_permuge_ids, match_mask_i = self._process_match_indices(matches_i, targets_i.shape[0], targets_i)
            permute_targets_i = targets_i[tgt_permuge_ids]
            # shift the target correspondingly 
            for idx, shift in zip(matches_i[0], matches_i[2]):
                num_vert = permute_targets_i[idx, :, -1].sum().long()
                shifted_target = torch.roll(permute_targets_i[idx, :num_vert], -shift.item(), 0)
                permute_targets_i[idx, :num_vert] = shifted_target
            all_permute_targets.append(permute_targets_i)
            all_match_mask.append(match_mask_i)

        permute_targets = torch.stack(all_permute_targets, dim=0)        
        match_mask = torch.stack(all_match_mask, dim=0)
        return permute_targets, match_mask

    @staticmethod
    def _process_match_indices(indices, N, targets):
        permute_ids = torch.full([N], -1)
        matched_mask = torch.zeros(targets.shape[:-1])
        for i, j in zip(indices[0], indices[1]):
            permute_ids[i] = j
            num_verts = (targets[j, :, -1] == 1).sum()
            matched_mask[i, :num_verts] = 1
        return permute_ids, matched_mask
    
    @staticmethod
    def _rotate_target(target):
        num_valid_vert = (target[:, -1] == 1).sum()
        all_rotated_targets = []
        for start_idx in range(num_valid_vert):
            rotated_target = target[:num_valid_vert].roll(-start_idx, 0)
            all_rotated_targets.append(rotated_target)
        rotated_targets = torch.stack(all_rotated_targets, dim=0)
        zero_paddings = torch.zeros(rotated_targets.shape[0], target.shape[0] - num_valid_vert, 
                                    rotated_targets.shape[2]).to(rotated_targets.device)
        rotated_targets = torch.cat([rotated_targets, zero_paddings], dim=1)
        return rotated_targets, num_valid_vert
    
    @staticmethod
    def _check_matching(matches, merged_pred_polys, permuted_gt_polys):
        bs = len(matches)
        debug_dir = './debug_matching'
        for idx in range(bs):
            pred_ids = matches[idx][0]
            for pred_id in pred_ids:
                pred_polygon = merged_pred_polys[idx, pred_id]
                gt_polygon = permuted_gt_polys[idx, pred_id]
                viz_img = _draw_two_polygons(gt_polygon, pred_polygon)
                save_path = os.path.join(debug_dir, 'sample_{}_match_{}.png'.format(idx, pred_id))
                imageio.imwrite(save_path, viz_img)
        import pdb; pdb.set_trace()

    

def _draw_two_polygons(gt, pred):
    num_vert = gt[:, -1].sum().long().item()
    gt = gt[:num_vert]
    pred = pred[:num_vert]
    img1 = _draw_polygon(gt, (255, 0, 0))
    img2 = _draw_polygon(pred, (0, 0, 255))
    img = np.concatenate([img1, img2], axis=1)
    return img

def _draw_polygon(poly, color):
    prev = None
    image = np.zeros([256, 256, 3])
    for idx, vert in enumerate(poly):
        c = vert * 255
        cv2.circle(image, (int(c[0]), int(c[1])), 3, (0, 255, 255), 1)
        cv2.putText(image, '{}'.format(idx), (int(c[0]),int(c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        if idx == 0:
            prev = poly[-1] * 255
        #print(c, prev)
        cv2.line(image, (int(prev[0]), int(prev[1])), (int(c[0]), int(c[1])), color, 2)
        prev = c
    return image
    
         
