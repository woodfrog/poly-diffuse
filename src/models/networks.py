# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

from src.models.polygon_models.polygon_net import PolyModel

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = 256
        self.img_channels = 3
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](**model_kwargs)
        print(f'Precond: sigma_max={sigma_max}, sigma_min={sigma_min}, sigma_data={sigma_data}')

    def forward(self, x, sigma, mu_guide, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        model_in = (c_in * x).to(dtype) + (1 - c_in) * mu_guide.to(dtype)
        model_outputs = self.model(model_in, c_noise.flatten(), **model_kwargs)
        F_x = model_outputs['final_results']
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def get_model_input(self, x, sigma, mu_guide, force_fp32=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        model_in = (c_in * x).to(dtype) + (1 - c_in) * mu_guide.to(dtype)
        return model_in


    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
