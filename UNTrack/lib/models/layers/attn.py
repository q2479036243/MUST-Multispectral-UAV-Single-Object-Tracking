import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False, query_len=1, lens_z=432, lens_x=576, add_cls_token=True):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        if add_cls_token:
            lens_x = N - query_len - lens_z     # 由于光谱候选，搜索的长度可能会改变
            q_prompt, q_template, q_search = torch.split(q, [query_len, lens_z, lens_x], dim=2)
            k_prompt, k_template, k_search = torch.split(k, [query_len, lens_z, lens_x], dim=2)
            v_prompt, v_template, v_search = torch.split(v, [query_len, lens_z, lens_x], dim=2)
        else:
            lens_x = N - lens_z
            q_template, q_search = torch.split(q, [lens_z, lens_x], dim=2)
            k_template, k_search = torch.split(k, [lens_z, lens_x], dim=2)
            v_template, v_search = torch.split(v, [lens_z, lens_x], dim=2)
        
        # asymmetric attention
        if add_cls_token:
            ## prompt attention
            attn_prompt = (q_prompt @ k_prompt.transpose(-2, -1)) * self.scale
            attn_prompt = attn_prompt.softmax(dim=-1)
            attn_prompt = self.attn_drop(attn_prompt)
            x_prompt = (attn_prompt @ v_prompt).transpose(1, 2).reshape(B, query_len, C)
        
        ## template attention
        attn_template = (q_template @ k_template.transpose(-2, -1)) * self.scale
        attn_template = attn_template.softmax(dim=-1)
        attn_template = self.attn_drop(attn_template)
        x_template = (attn_template @ v_template).transpose(1, 2).reshape(B, lens_z, C)
        
        ## search attention
        attn_search = (q_search @ k.transpose(-2, -1)) * self.scale
        attn_search = attn_search.softmax(dim=-1)
        attn_search = self.attn_drop(attn_search)
        x_search = (attn_search @ v).transpose(1, 2).reshape(B, lens_x, C)
        
        if add_cls_token:
            x = torch.cat([x_prompt, x_template, x_search], dim=1)
        else:
            x = torch.cat([x_template, x_search], dim=1)
        attn = attn_search
        ####

        # if self.rpe:
        #     relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
        #     attn += relative_position_bias

        # if mask is not None:
        #     attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x