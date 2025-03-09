# 用来将基于RGB数据的预训练权重的初始层由3维扩展为高位以适应光谱数据
# 根据CIE规定，红(R)700.0nm；绿(G)546.1nm；蓝(B)435.8nm

import os
import numpy as np
import torch
import torch.nn.functional as F
  
pretrain_path = os.path.dirname(os.path.abspath(__file__))
input_dim = 3
output_dim = 8          # MUST数据集是8通道
input_pth = 'mae_pretrain_vit_base.pth'
dels_pth = 'mae_pretrain_vit_base_dels.pth'
chazhi_pth = 'mae_pretrain_vit_base_chazhi.pth'

R_band = 700.0
G_band = 546.1
B_band = 435.8
bands = [422.5, 487.5, 550, 602.5, 660, 725, 785, 887.5]

# 删除第一层，实现第一层随机初始化
file_path = os.path.join(pretrain_path, input_pth)
checkpoint = torch.load(file_path)
del checkpoint['model']['patch_embed.proj.weight']
del checkpoint['model']['patch_embed.proj.bias']
torch.save(checkpoint, os.path.join(pretrain_path, dels_pth))

# 第一层根据光谱插值出来多通道
file_path = os.path.join(pretrain_path, input_pth)
checkpoint = torch.load(file_path)
model_weights = checkpoint['model']['patch_embed.proj.weight']
R_weight = model_weights[:, 2, :, :]
G_weight = model_weights[:, 1, :, :]
B_weight = model_weights[:, 0, :, :]
weight_tensor = []
for band in reversed(bands):        # 要根据数据的读取方式进行修改，确认bands顺序是顺序还是倒叙
    if band <= G_band:
        weight = (B_weight * (G_band - band) + G_weight * (band - B_band)) / (G_band - B_band)
    elif band > G_band and band < R_band:
        weight = (G_weight * (R_band - band) + R_weight * (band - G_band)) / (R_band - G_band)
    else:
        weight = R_weight * 1
    weight = weight.unsqueeze(1)
    weight_tensor.append(weight)
weight_concat = torch.cat(weight_tensor, dim=1)
checkpoint['model']['patch_embed.proj.weight'] = weight_concat
torch.save(checkpoint, os.path.join(pretrain_path, part_pth))

