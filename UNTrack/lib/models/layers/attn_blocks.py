import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, 
                          box_mask_z: torch.Tensor):
    """
    光谱候选模块，根据光谱特征剔除搜索中的背景区域
    Args:
        attn (torch.Tensor): [B, num_heads, N_s * L_s, N_t * L_t + N_s * L_s], 非对称注意力的中间变量, 对应了以搜索为Q, 模板+搜索为KV的注意力图
        tokens (torch.Tensor):  [B, N_t * L_t + N_s * L_s, C], 非对称注意力的结果的一部分, 排除了提示对应的tokens, 因为提示不参与光谱候选, 全部保留
        lens_t (int): N_t * L_t, N_t是模板的数量, L_t是一个模板的token长度
        keep_ratio (float): 光谱候选保留的比例, 是一个余弦规律变化的值
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): 一个mask矩阵, 在这里没有使用

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-2]
    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None
    
    attn_t = torch.norm(attn.mean(dim=1), dim=2)        # 先沿head维度求均值，之后计算模长作为选取topk的标准
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # 模板对应的tokens都保留，搜索对应的区域部分剔除
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # 保留tokp部分
    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None, 
                add_cls_token=False, query_len=1, lens_z=432, lens_x=576):
        # 计算非对称注意力，query_len是提示长度，lens_z是模板数量*模板长度，lens_x是搜索数量*搜索长度
        x_attn, attn = self.attn(self.norm1(x), mask, True, query_len=query_len, lens_z=lens_z, lens_x=lens_x, add_cls_token=add_cls_token)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        # 执行光谱候选
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            if add_cls_token:   # 有提示的情况
                tokens = x[:, :query_len, :]
                x, global_index_search, removed_index_search = candidate_elimination(attn[:, :, :, query_len:], 
                                                                                     x[:, query_len:, :],
                                                                                     lens_t, 
                                                                                     keep_ratio_search, global_index_search, 
                                                                                     ce_template_mask)
                x = torch.cat([tokens, x], dim=1)  # (B, query_len+z+x, 768)
            else:               # 没有提示
                x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, query_len=1, lens_z=432, lens_x=576):
        x = x + self.drop_path(self.attn(self.norm1(x), mask, query_len=query_len, lens_z=lens_z, lens_x=lens_x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
