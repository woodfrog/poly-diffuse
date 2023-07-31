import torch
import torch.nn as nn
from .mlp import MLP

from einops import rearrange, repeat
from .nn_util import (
    positional_encoding_1d,
    timestep_embedding,
    linear,
)

from .deformable_transformer import PolyTransformerDecoder, PolyTransformerDecoderLayer


class PolyMetaModel(nn.Module):
    """The model with a small transformer for the mu_meta and sigma_meta of the
    forward process.
    """
    def __init__(self, input_dim, embed_dim, max_poly=50, num_vert=20):
        super(PolyMetaModel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_poly = max_poly
        self.num_vert = num_vert

        self.coords_embed = linear(2*input_dim, embed_dim)
        self.input_mlp = MLP(input_dim*2+embed_dim*2, embed_dim, embed_dim, 2)

        poly_token = positional_encoding_1d(input_dim, max_poly)
        vert_token = positional_encoding_1d(input_dim, num_vert)
        expand_vert_token = repeat(vert_token, 'n c -> n1 n c', n1=max_poly)
        expand_poly_token = repeat(poly_token, 'n c -> n n2 c', n2=num_vert)
        self.per_vertex_encoding = torch.cat([expand_poly_token, expand_vert_token], dim=-1)
        self.class_embedding = nn.Embedding(3, embed_dim)

        self.global_token_embedding = nn.Embedding(1, embed_dim)

        encoder_layer = PolyTransformerDecoderLayer(d_model=embed_dim, n_heads=8)
        self.encoder = PolyTransformerDecoder(encoder_layer, num_layers=2)
        
        self.output_coord_proj = nn.Linear(embed_dim, 2)
        self.output_sigma_proj = nn.Linear(embed_dim, 1)

        nn.init.constant_(self.output_coord_proj.weight.data, 0)
        nn.init.constant_(self.output_coord_proj.bias.data, 0)
    
    def forward(self, polys, mask, poly_class):
        bs = polys.size(0)
        num_polys = polys.size(1)
        num_verts = polys.size(2)
        vert_encodings = timestep_embedding((polys.flatten() + 1) * 127.5, self.input_dim).view(bs, num_polys, num_verts, 2, -1)
        vert_encodings = torch.cat([vert_encodings[:, :, :, 0, :], vert_encodings[:, :, :, 1, :]], dim=-1)
        vert_encodings = self.coords_embed(vert_encodings)
        
        vertex_id_encodings = repeat(self.per_vertex_encoding, 'n1 n2 c -> b n1 n2 c', b=bs).to(vert_encodings.device)
        vertex_id_encodings = vertex_id_encodings[:, :num_polys, :, :]

        poly_class_emb = self.class_embedding(poly_class)
        poly_class_emb = repeat(poly_class_emb, 'b n1 c -> b n1 n2 c', n2=num_verts)

        vert_encodings = torch.cat([vert_encodings, vertex_id_encodings, poly_class_emb], dim=-1)
        vert_encodings = rearrange(vert_encodings, 'b n1 n2 c -> b (n1 n2) c')
        vert_encodings = self.input_mlp(vert_encodings)
    
        global_token = repeat(self.global_token_embedding.weight[0], 'c -> b n1 n2 c', b=bs, n1=num_polys, n2=1)
        vert_encodings = rearrange(vert_encodings, 'b (n1 n2) c -> b n1 n2 c', n1=num_polys)
        vert_encodings = torch.cat([global_token, vert_encodings], dim=-2)
        vert_encodings = rearrange(vert_encodings, 'b n1 n2 c -> b (n1 n2) c')

        mask_padding = torch.zeros(*global_token.shape[:3]).bool().to(mask.device)
        attn_mask = torch.cat([mask_padding, mask], dim=-1)
        attn_mask = rearrange(attn_mask, 'b n1 n2 -> b (n1 n2)')
        
        # add the global token
        vert_encodings = self.encoder(vert_encodings, num_polys, key_padding_mask=attn_mask)

        vert_encodings = rearrange(vert_encodings, 'b (n1 n2) l -> b n1 n2 l', n1=num_polys)
        #coords_encodings = vert_encodings[:, :, 1:, :]
        poly_global_encodings = vert_encodings[:, :, 0, :]
        
        coords_pred = self.output_coord_proj(poly_global_encodings).sigmoid()
        coords_pred = coords_pred * 2 - 1 # rescale from [0, 1] to [-1, 1] to align with the dm data space
        
        sigma_pred = self.output_sigma_proj(poly_global_encodings)
        sigma_pred = sigma_pred.sigmoid()

        return coords_pred, sigma_pred
