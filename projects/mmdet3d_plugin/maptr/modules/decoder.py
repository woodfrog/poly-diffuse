import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence)

from mmdet.models.utils.transformer import inverse_sigmoid, DetrTransformerDecoderLayer
import warnings

from projects.mmdet3d_plugin.maptr.dense_heads import linear, timestep_embedding, MLP

import copy
from einops import rearrange


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRANSFORMER_LAYER.register_module()
class PolyDetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(PolyDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.num_verts = 20
        assert len(operation_order) == 8
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])


    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        bs = query.shape[1]

        # prepare inputs for the per-poly attention
        query_poly = query.clone()
        query_poly = rearrange(query_poly.permute(1,0,2), 'b (n1 n2) c -> (b n1) n2 c', n2=self.num_verts)
        query_poly = query_poly.permute(1, 0, 2)
        query_pos_poly = rearrange(query_pos.permute(1,0,2), 'b (n1 n2) n3 -> (b n1) n2 n3', n2=self.num_verts)
        query_pos_poly = query_pos_poly.permute(1, 0, 2)
        # NOTE: need to clone the mask here to prevent changing the global mask...
        poly_mask = rearrange(query_key_padding_mask.clone(), 'b (n1 n2) -> (b n1) n2', n2=self.num_verts)
        poly_mask[:, 0] = False # prevent nan
        temp_key_poly = temp_value_poly = query_poly

        for layer in self.operation_order:
            if layer == 'self_attn':
                if attn_index == 1:
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity=query,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    attn_index += 1
                else: # polygon-wise attn
                    assert attn_index == 0
                    query_poly = self.attentions[attn_index](
                        query_poly,
                        temp_key_poly,
                        temp_value_poly,
                        identity=query_poly,
                        query_pos=query_pos_poly,
                        key_pos=query_pos_poly,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=poly_mask,
                        **kwargs)
                    attn_index += 1

            elif layer == 'norm':
                if norm_index == 0:
                    query_poly = self.norms[norm_index](query_poly)
                    query_poly = rearrange(query_poly.permute(1,0,2), '(b n1) n2 c -> b (n1 n2) c', b=bs)
                    query_poly = query_poly.permute(1, 0, 2)
                    query = query_poly
                else:
                    query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity=query,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity=query)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, timestep_embed=None, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.timestep_embed = timestep_embed

        if timestep_embed is not None:
            time_embed_layers = []
            for _ in range(len(self.layers)):
                time_embed_layer = nn.Sequential(
                    nn.SiLU(),
                    linear(
                        timestep_embed,
                        2 * self.embed_dims,
                    ),
                )
                time_embed_layers.append(time_embed_layer)
            self.time_embed_layers = nn.ModuleList(time_embed_layers)

        # mlp for feeding latest ref points coord info back to transformer nodes
        coord_mlp = MLP(self.embed_dims*2, self.embed_dims, self.embed_dims, 2)
        self.all_coord_mlps = _get_clones(coord_mlp, self.num_layers)


    def forward(self,
                query,
                t_emb,
                *args,
                reference_points=None,
                reg_branches=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                query_pos=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        bs = query.shape[1]

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            
            # Time embedding
            if self.timestep_embed:
                emb_out = self.time_embed_layers[lid](t_emb).type(output.dtype)[:, None, :]
                emb_out = emb_out.permute(1, 0, 2)
                scale, shift = torch.chunk(emb_out, 2, dim=2)
                output = output * (1 + scale) + shift
            
            # use the latest reference points to update the query_pos of the nodes
            flat_points = rearrange(reference_points, 'b n1 n2 -> (b n1 n2)')
            coords_emb = timestep_embedding(flat_points * 255, self.embed_dims)
            coords_emb = rearrange(coords_emb, '(b n1 n2) n3 -> b n1 (n2 n3)', 
                                n1=reference_points.shape[1], n2=2)
            coords_emb = self.all_coord_mlps[lid](coords_emb)
            coords_emb = coords_emb.permute(1, 0, 2)
            query_pos = query_pos + coords_emb

            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                query_pos=query_pos,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                # new_reference_points[..., 2:3] = tmp[
                #     ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                # use detach if directly predict x0 using the reg_branches
                reference_points = new_reference_points.detach()

                # NOTE: can't detach here if not doing direct regression
                # under the noise-estimation or the mixture formulation
                #reference_points = new_reference_points

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

