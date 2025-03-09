import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.untrack.vit import vit_base_patch16_224
from lib.models.untrack.vit_ce import vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh

from timm.models.layers import Mlp


class ATTFu(nn.Module):
    def __init__(self, channels, ratio=0.25):
        super(ATTFu, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(2*channels, int(ratio*channels), bias=False),
            nn.ReLU(),
            nn.Linear(int(ratio*channels), 2*channels, bias=False),
            nn.Sigmoid()
        )
        self.mlp = Mlp(in_features=2*channels, hidden_features=4*2*channels)
        
    def forward(self, l_pro, l_tem):
        l_tem = torch.mean(l_tem, dim=1, keepdim=True)      # l_tem是模板，先做全局池化
        l_fu = torch.cat([l_pro, l_tem], dim=2)             # 模板和提示cat在一起
        att = self.fc(l_fu)                                 # 利用fc层实现自注意力
        out = self.mlp(att) + att

        return out[:,:,:self.channels]                      # 只保留提示部分
    

class UNTrack(nn.Module):
    """ This is the base class for UNTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1, num_searches=2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len
        self.num_searches = num_searches
        
        self.prompt = ATTFu(768)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(self.num_searches-1, len(search)):
            # search的提取做了修改，以获取list形式的search
            x_, aux_dict, top_k_indices = self.backbone(z=template.copy(), x=[search[idx] for idx in range(i-self.num_searches+1, i+1)],
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
            # search部分只保留最后一个search的特征图
            x = torch.cat((x_[:, :-1*self.num_searches*self.feat_len_s, :],x_[:,-self.feat_len_s:,:]), dim=1)
            
            feat_last = x   # x.shape torch.Size([8, 1009, 768])
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)    # enc_opt.shape torch.Size([8, 576, 768])
            if self.backbone.add_cls_token:
                t_query = (x[:, :self.token_len]) # (B, N, C)  # self.track_query.shape torch.Size([8, 1, 768])
                z_query = (x[:, self.token_len:-self.feat_len_s])
                self.track_query = self.prompt(t_query, z_query).clone().detach()
                # self.track_query = nn.Parameter(torch.randn(t_query.shape)).to(t_query.device)
                # self.track_query = t_query
                
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_untrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('UNTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = UNTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
        num_searches=cfg.DATA.SEARCH.LENGTH,
    )

    return model
