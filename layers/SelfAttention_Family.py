import torch
from torch import nn
import numpy as np
from math import sqrt

from utils.masking import TriangularCausalMask
from einops import rearrange, repeat


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True,  # mask it or not, encoder=>False / decoder=>True
                 scale=None,            # scaling scores in softmax
                 attention_dropout=0.1, # 
                 output_attention=False):
        super(FullAttention, self).__init__()

        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape  # (Batch, in_num_query, H:n_heads, d_k)
        _, S, _, _ = keys.shape     # (Batch, in_num_key, H, d_k)
        _, _, _, D = values.shape   # (Batch, in_num_val=in_num_key, H, d_v)
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum('blhe,bshe->bhls', queries, keys)

        # masking process for decoder, but we use encoder only.
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)

        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('blhs,bshd->blhd', A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention
        self.n_heads = n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        output, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        output = output.view(B, L, -1)  # (B, in_num_query, h, d_v) -> (B, in_num_query, h * d_v)
        output = self.out_projection(output)    # (B, in_num_query, d_model)

        return output, attn