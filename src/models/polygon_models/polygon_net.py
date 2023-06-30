import torch
import copy
import math

import torch.nn as nn
import torch.nn.functional as F

from .nn_util import (
    positional_encoding_1d,
    linear,
    timestep_embedding,
    NestedTensor, 
    inverse_sigmoid
)

from einops import rearrange, repeat

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .ops.modules import MSDeformAttn
from .mlp import MLP
from .resnet import ResNetBackbone


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros([x.shape[0], x.shape[2], x.shape[3]]).bool().to(x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        #dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PolyModel(nn.Module):
    def __init__(self, num_poly, num_vert, input_dim, hidden_dim, num_feature_levels):
        super(PolyModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_poly = num_poly
        self.num_vert = num_vert

        # Create the ConvNet backbone for extracting image features
        self.backbone = ResNetBackbone()
        backbone_strides = self.backbone.strides
        backbone_num_channels = self.backbone.num_channels

        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.coords_embed = linear(2*input_dim, hidden_dim)
        self.exs_embed = linear(input_dim, hidden_dim)

        time_embed_dim = input_dim * 4
        self.time_embed = nn.Sequential(
            linear(input_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.img_pos = PositionEmbeddingSine(hidden_dim // 2)

        self.vertex_input_fc_noise = nn.Linear(2*input_dim+hidden_dim, hidden_dim)
        self.output_fc = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim // 2, output_dim=2, num_layers=2)

        self.transformer = RoomTransformer(num_poly, num_vert, d_model=hidden_dim, time_emb_channels=time_embed_dim,
                                           nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                                           dim_feedforward=1024, dropout=0.1, return_intermediate_dec=True)
        
        #self.query_embed_noise = nn.Embedding(num_poly*num_vert, hidden_dim)

        self.output_init_logvar = MLP(hidden_dim, hidden_dim//2, 1, 2)
        
        self.class_embed_noise = nn.Linear(hidden_dim, 1)
        self.coord_embed_noise = MLP(hidden_dim, hidden_dim, 2, 3)

        self.output_mlp = MLP(hidden_dim*2, hidden_dim, 2, 3)

        poly_token = positional_encoding_1d(input_dim, num_poly)
        vert_token = positional_encoding_1d(input_dim, num_vert)
        expand_vert_token = repeat(vert_token, 'n c -> n1 n c', n1=num_poly)
        expand_poly_token = repeat(poly_token, 'n c -> n n2 c', n2=num_vert)
        self.per_vertex_encoding = torch.cat([expand_poly_token, expand_vert_token], dim=-1)
        self.poly_exs_token = nn.Embedding(2, hidden_dim)
        
        num_pred_noise = self.transformer.decoder_time.num_layers

        # init the input projections
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # iterative refinement
        self.class_embed_noise = _get_clones(self.class_embed_noise, num_pred_noise)
        self.coord_embed_noise = _get_clones(self.coord_embed_noise, num_pred_noise)

        # hack implementation for iterative bounding box refinement
        self.transformer.decoder_time.coord_embed = self.coord_embed_noise

    @staticmethod
    def get_ms_feat(xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
    def forward(self, polys=None, timesteps=None, attn_mask=None, poly_mask=None,
                image=None):
        bs = image.shape[0]

        srcs, masks, all_pos = self.process_image(image)
        return self._estimate_noise(bs, polys, attn_mask, poly_mask, timesteps, srcs, masks, all_pos)

    def process_image(self, image):
        # Prepare image features
        image_feats, feat_mask, all_image_feats = self.backbone(image)

        # zero out the image information (for analysis only)
        #for key, val in image_feats.items():
        #    image_feats[key] = torch.zeros_like(val)

        features = self.get_ms_feat(image_feats, feat_mask)

        srcs = []
        masks = []
        all_pos = []

        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            srcs.append(self.input_proj[l](src))
            pos = self.img_pos(src).to(src.dtype)
            all_pos.append(pos)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0].to(src.device)
                pos_l = self.img_pos(src).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                all_pos.append(pos_l)
        return srcs, masks, all_pos

    def _estimate_noise(self, bs, polys, attn_mask, poly_mask, timesteps, 
                        srcs, masks, all_pos):
        # Process polygon information
        num_polys = polys.size(1)
        num_verts = polys.size(2)

        vert_coords = polys

        # used for reference points of deformable-attn
        raw_vert_coords = (vert_coords + 1) / 2

        # sinusoidal positional encoding followed by learnable linear layer
        vert_coords = timestep_embedding((vert_coords.flatten() + 1) * 127.5, self.input_dim).view(bs, num_polys, num_verts, 2, -1)
        vert_coords = torch.cat([vert_coords[:, :, :, 0, :], vert_coords[:, :, :, 1, :]], dim=-1)
        vert_coords = self.coords_embed(vert_coords)

        vertex_id_encodings = repeat(self.per_vertex_encoding, 'n1 n2 c -> b n1 n2 c', b=bs).to(vert_coords.device)

        vert_encodings = torch.cat([vert_coords, vertex_id_encodings], dim=-1)
        vert_encodings = self.vertex_input_fc_noise(vert_encodings)
        query_embeds = vertex_id_encodings

        # add the polygen existence encoding
        poly_exs_encodings = self.poly_exs_token(poly_mask.long())
        vert_encodings += poly_exs_encodings[:, :, None, :]

        vert_encodings = rearrange(vert_encodings, 'b n1 n2 c -> b (n1 n2) c')
        vert_coords = rearrange(vert_coords, 'b n1 n2 c -> b (n1 n2) c')
        raw_vert_coords = rearrange(raw_vert_coords, 'b n1 n2 c -> b (n1 n2) c')
        query_embeds = rearrange(query_embeds, 'b n1 n2 c -> b (n1 n2) c')

        attn_mask = rearrange(attn_mask, 'b n1 n2 -> b (n1 n2)')
        t_emb = self.time_embed(timestep_embedding(timesteps, self.input_dim))

        hs, init_reference, inter_references = self.transformer(srcs, masks, all_pos, 
                                                                query_embeds, vert_encodings, 
                                                                raw_vert_coords, attn_mask,
                                                                t_emb)

        # produce the final output coords status
        outputs_class, outputs_coords = self._get_per_layer_results(hs, init_reference, inter_references, 'noise')
       
        final_encodings = torch.cat([hs[-1], vert_coords], dim=-1)
        final_results = self.output_mlp(final_encodings)
        final_results = rearrange(final_results, 'b (n1 n2) c -> b n1 n2 c', n1=self.num_poly)

        outputs_coords = rearrange(outputs_coords, 'n0 b (n1 n2) c -> n0 b n1 n2 c', n1=self.num_poly)
        outputs_class = rearrange(outputs_class, 'n0 b (n1 n2) c -> n0 b n1 n2 c', n1=self.num_poly)

        out = {
            'final_results': final_results,
            'all_coords': outputs_coords, 
            'all_exs': outputs_class
        }
        return out
    
    def _get_per_layer_results(self, hs, init_reference, inter_references, type):
        assert type == 'noise', 'Only the noise mode supported now'
        class_embed_model = self.class_embed_noise
        coord_embed_model = self.coord_embed_noise
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = class_embed_model[lvl](hs[lvl])
            tmp = coord_embed_model[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_class, outputs_coords


class RoomTransformer(nn.Module):
    def __init__(self, num_poly, num_vert, d_model=512, time_emb_channels=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 ):
        super(RoomTransformer, self).__init__()

        self.num_poly = num_poly
        self.num_vert = num_vert
        self.d_model = d_model
        self.time_emb_channels = time_emb_channels

        self.empty_img_token = nn.Embedding(1, 256)

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer_time = DeformableTransformerDecoderLayer(d_model, time_emb_channels, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)

        self.decoder_time = DeformableTransformerDecoder(decoder_layer_time, num_decoder_layers,
                                                               return_intermediate_dec)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)
        #xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        #constant_(self.reference_points.bias.data, 0.)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, vertex_encodings, 
                vertex_coords=None, tgt_mask=None, t_emb=None):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)
    
        # prepare input for decoder
        bs, _, c = memory.shape
        tgt = vertex_encodings
        # directly use the input coords as the initial reference points for df-attention
        reference_points = vertex_coords.clip(0, 1)
        init_reference_out = reference_points

        hs, inter_references = self.decoder_time(tgt, reference_points, memory,
                                                spatial_shapes, level_start_index, valid_ratios, 
                                                t_emb, query_embed, mask_flatten,
                                                key_padding_mask=tgt_mask)

        return hs, init_reference_out, inter_references
