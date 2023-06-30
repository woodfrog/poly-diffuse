# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        print(f'Loss: P_mean={P_mean}, P_std={P_std}, sigma_data={sigma_data}')

    def __call__(self, net, net_guide, images, **model_kwargs):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        
        mu_guide, sigma_guide = net_guide(images, model_kwargs['attn_mask'])
        mu_guide = mu_guide[:, :, None, :]
        n = torch.randn_like(y) * sigma * sigma_guide[:, :, None, :]
        noised_y = y + n
        # make sure that the indicator variable is set up correctly
        D_yn = net(noised_y, sigma, mu_guide, **model_kwargs)

        # NOTE: only apply the loss to the valid vertices
        loss_mask = (model_kwargs['attn_mask'] == 0) 
        l2_loss_coords = ((D_yn - y) ** 2)[:, :, :, :2]
        loss = weight * l2_loss_coords
        loss = loss[loss_mask]
        return loss

#----------------------------------------------------------------------------
