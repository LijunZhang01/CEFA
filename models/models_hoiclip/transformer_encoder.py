import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp
import math

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.enable_cp:
            def _inner_forward(args):
                src_inner, q_inner, k_inner, src_mask_inner, src_key_padding_mask_inner = args
                src_inner = self.self_attn(q_inner, k_inner, value=src_inner, attn_mask=src_mask_inner,
                                      key_padding_mask=src_key_padding_mask_inner)[0]
                return src_inner

            src2 = cp.checkpoint(_inner_forward, (src, q, k, src_mask, src_key_padding_mask))
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class graphtiencoder(TransformerEncoder):
    def __init__(self,
                 hidden_dim=256,
                 feedforward_dim=1024,
                 num_heads=8,
                 dropout=0.1,
                 activation='relu',
                 num_layers=2,
                 normalize_before=False,
                 return_intermediate=False,
                 total_spatial_shapes=None):
        total_spatial_shapes = [] if total_spatial_shapes is None else total_spatial_shapes
        # activation=_get_activation_fn(activation)
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim,
                                                dropout, activation, normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim)
        super(graphtiencoder, self).__init__(encoder_layer, num_layers,encoder_norm)
        self.spatial_shapes = total_spatial_shapes
        self.hidden_dim = hidden_dim
        # self.mask_query = nn.Embedding(1, hidden_dim)
        # self.query_embed_list = nn.Embedding(total_spatial_shapes[0]*total_spatial_shapes[1], hidden_dim * 2)
        self.pos=nn.Embedding(total_spatial_shapes[0],total_spatial_shapes[1])
        # a=1024*3
        # self.output_proj = nn.Linear(hidden_dim, self.spatial_shapes[2])
        # self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tgt,mask_flatten=None):
        bs = tgt.shape[1]
        mae_output = []
        
        # query_pos, tgt = torch.split(self.query_embed_list.weight, self.hidden_dim, dim=1)
        # query_pos = query_pos.unsqueeze(1).expand(-1,bs,-1)
        # tgt = tgt.unsqueeze(1).expand(-1,bs,-1)
        pos=self.pos.weight.unsqueeze(1).expand(-1,bs,-1)
        
        
        hs = super(graphtiencoder, self).forward(
            tgt, src_key_padding_mask=mask_flatten,pos=pos
        )
        # hs = hs.transpose(1,2)
        # print(hs.shape)
        mae_output=hs
        # hs=hs[256:,:,:]
        # output = self.output_proj(hs)
        # # print("output:",output.shape)
        # # print(h,w,c)
        # # print(output.shape)
        # ddd=output.permute(1,2,0)
        # # print(ddd.shape)
        # mae_output.append(ddd)
        return mae_output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])