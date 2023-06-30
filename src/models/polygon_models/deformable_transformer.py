import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .ops.modules import MSDeformAttn
from .nn_util import (
    inverse_sigmoid,
    linear,
    timestep_embedding,
)
from .mlp import MLP

from einops import rearrange


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, time_emb_channels=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 n_polys=20):
        super().__init__()
        self.d_model = d_model
        self.n_polys = n_polys
        self.time_emb_channels = time_emb_channels
        self.time_embeded = time_emb_channels > 0

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention (global)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # self attention (per polygon)
        self.self_attn_poly = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_poly = nn.Dropout(dropout)
        self.norm2_poly = nn.LayerNorm(d_model)

        # time step embedding
        if self.time_embeded > 0:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    time_emb_channels,
                    2 * d_model,
                ),
            )

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # coord mapping mlp
        self.coord_mlp = MLP(d_model*2, d_model, d_model, 2)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, 
                src, src_spatial_shapes, level_start_index,
                t_emb=None,
                src_padding_mask=None,
                key_padding_mask=None):
        # Time embedding
        if self.time_embeded:
            emb_out = self.emb_layers(t_emb).type(tgt.dtype)[:, None, :]
            scale, shift = torch.chunk(emb_out, 2, dim=2)
            tgt = tgt * (1 + scale) + shift
        
        # use the latest reference points to update the q and k
        flat_points = rearrange(reference_points[:, :, 0, :], 'b n1 n2 -> (b n1 n2)')
        coords_emb = timestep_embedding(flat_points * 255, self.d_model)
        coords_emb = rearrange(coords_emb, '(b n1 n2) n3 -> b n1 (n2 n3)', 
                            n1=reference_points.shape[1], n2=2)
        coords_emb = self.coord_mlp(coords_emb)
        query_pos = query_pos + coords_emb
        
        q = k = self.with_pos_embed(tgt, query_pos)
        # self attention (global)
        tgt2 = \
        self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=key_padding_mask)[
            0].transpose(0, 1)
        tgt_global = tgt + self.dropout2(tgt2)
        tgt_global = self.norm2(tgt_global)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt_global, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt_global = tgt_global + self.dropout1(tgt2)
        tgt_global = self.norm1(tgt_global)

        # self attention (polygon)
        q_poly = rearrange(q, 'b (n1 n2) c -> (b n1) n2 c', n1=self.n_polys)
        k_poly = rearrange(k, 'b (n1 n2) c -> (b n1) n2 c', n1=self.n_polys)
        tgt_poly = rearrange(tgt, 'b (n1 n2) c -> (b n1) n2 c', n1=self.n_polys)
        key_padding_mask_poly = rearrange(key_padding_mask.clone(), 'b (n1 n2) -> (b n1) n2', n1=self.n_polys)
        key_padding_mask_poly[:, 0] = False
        tgt2 = \
        self.self_attn_poly(q_poly.transpose(0, 1), k_poly.transpose(0, 1), tgt_poly.transpose(0, 1), 
                            key_padding_mask=key_padding_mask_poly)[
            0].transpose(0, 1)
        tgt_poly = tgt_poly + self.dropout2(tgt2)
        tgt_poly = self.norm2(tgt_poly)
        tgt_poly = rearrange(tgt_poly, '(b n1) n2 c -> b (n1 n2) c', n1=self.n_polys)

        # merge global and per-polygon features
        tgt = tgt_poly + tgt_global

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.coord_embed = None

    def forward(self, tgt, reference_points, 
                src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                t_emb=None, query_pos=None, src_padding_mask=None, 
                key_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output, query_pos, reference_points_input, 
                            src, src_spatial_shapes, src_level_start_index,
                            t_emb, src_padding_mask, key_padding_mask)
                            
            if self.coord_embed is not None:
                tmp = self.coord_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class PolyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8, n_polys=20):
        super().__init__()
        self.d_model = d_model
        self.n_polys = n_polys

        # self attention (global)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # self attention (per polygon)
        self.self_attn_poly = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_poly = nn.Dropout(dropout)
        self.norm2_poly = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos,
                key_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        # self attention (global)
        tgt2 = \
        self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=key_padding_mask)[
            0].transpose(0, 1)
        tgt_global = tgt + self.dropout2(tgt2)
        tgt_global = self.norm2(tgt_global)

        # self attention (polygon)
        q_poly = rearrange(q, 'b (n1 n2) c -> (b n1) n2 c', n1=self.n_polys)
        k_poly = rearrange(k, 'b (n1 n2) c -> (b n1) n2 c', n1=self.n_polys)
        tgt_poly = rearrange(tgt, 'b (n1 n2) c -> (b n1) n2 c', n1=self.n_polys)
        key_padding_mask_poly = rearrange(key_padding_mask, 'b (n1 n2) -> (b n1) n2', n1=self.n_polys)
        tgt2 = \
        self.self_attn_poly(q_poly.transpose(0, 1), k_poly.transpose(0, 1), tgt_poly.transpose(0, 1), 
                            key_padding_mask=key_padding_mask_poly)[
            0].transpose(0, 1)
        tgt_poly = tgt_poly + self.dropout2(tgt2)
        tgt_poly = self.norm2(tgt_poly)
        tgt_poly = rearrange(tgt_poly, '(b n1) n2 c -> b (n1 n2) c', n1=self.n_polys)

        # merge global and per-polygon features
        tgt = tgt_poly + tgt_global

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class PolyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, query_pos=None, 
                key_padding_mask=None):
        output = tgt

        for _, layer in enumerate(self.layers):
            output = layer(output, query_pos, key_padding_mask)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")