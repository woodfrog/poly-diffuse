import copy
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class PolyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()
        self.d_model = d_model

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

    def forward(self, tgt, query_pos, num_polys,
                key_padding_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        # self attention (global)
        tgt2 = \
        self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=key_padding_mask)[
            0].transpose(0, 1)
        tgt_global = tgt + self.dropout2(tgt2)
        tgt_global = self.norm2(tgt_global)

        # self attention (polygon)
        q_poly = rearrange(q, 'b (n1 n2) c -> (b n1) n2 c', n1=num_polys)
        k_poly = rearrange(k, 'b (n1 n2) c -> (b n1) n2 c', n1=num_polys)
        tgt_poly = rearrange(tgt, 'b (n1 n2) c -> (b n1) n2 c', n1=num_polys)
        key_padding_mask_poly = rearrange(key_padding_mask, 'b (n1 n2) -> (b n1) n2', n1=num_polys)
        tgt2 = \
        self.self_attn_poly(q_poly.transpose(0, 1), k_poly.transpose(0, 1), tgt_poly.transpose(0, 1), 
                            key_padding_mask=key_padding_mask_poly)[
            0].transpose(0, 1)
        tgt_poly = tgt_poly + self.dropout2(tgt2)
        tgt_poly = self.norm2(tgt_poly)
        tgt_poly = rearrange(tgt_poly, '(b n1) n2 c -> b (n1 n2) c', n1=num_polys)

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

    def forward(self, tgt, num_polys, query_pos=None, 
                key_padding_mask=None):
        output = tgt

        for _, layer in enumerate(self.layers):
            output = layer(output, query_pos, num_polys, key_padding_mask)

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