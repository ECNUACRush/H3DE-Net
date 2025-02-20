# # import configs_TransMorph as configs
# import torch
# from torch import nn as nn
# from torch.nn import functional as F
# import importlib


# import itertools
# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, trunc_normal_, to_3tuple
# from torch.distributions.normal import Normal
# import torch.nn.functional as nnf
# import numpy as np

# from typing import Tuple
# from collections import OrderedDict
# from einops.layers.torch import Rearrange
# from fairscale.nn.checkpoint import checkpoint_wrapper

# from .BiFormer.bra_legacy import BiLevelRoutingAttention
# from .BiFormer._common import Attention, AttentionLePE, DWConv

# import ml_collections
# from torchinfo import summary
# from typing import List, Optional, Tuple, Union


        
# def get_pe_layer(emb_dim, pe_dim=None, name='none'):
#     if name == 'none':
#         return nn.Identity()
#     # if name == 'sum':
#     #     return Summer(PositionalEncodingPermute2D(emb_dim))
#     # elif name == 'npe.sin':
#     #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
#     # elif name == 'npe.coord':
#     #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
#     # elif name == 'hpe.conv':
#     #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
#     # elif name == 'hpe.dsconv':
#     #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
#     # elif name == 'hpe.pointconv':
#     #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
#     else:
#         raise ValueError(f'PE name {name} is not surpported!')

# class Conv3dReLU(nn.Sequential):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             padding=0,
#             stride=1,
#             use_batchnorm=True,
#     ):
#         conv = nn.Conv3d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=False,
#         )
#         relu = nn.LeakyReLU(inplace=True)
#         if not use_batchnorm:
#             nm = nn.InstanceNorm3d(out_channels)
#         else:
#             nm = nn.BatchNorm3d(out_channels)

#         super(Conv3dReLU, self).__init__(conv, nm, relu)

# class DecoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             skip_channels=0,
#             use_batchnorm=True,
#     ):
#         super().__init__()
#         self.conv1 = Conv3dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.conv2 = Conv3dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

#     def forward(self, x, skip=None):
#         x = self.up(x)
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x

# class Block(nn.Module):
#     def __init__(self, dim, drop_path = 0., layer_scale_init_value = -1,
#                  num_heads = 8, n_win = (4,4,2), qk_dim = None, qk_scale = None,
#                  kv_per_win = 4, kv_downsample_ratio = 4, kv_downsample_kernel = None,
#                  kv_downsample_mode = 'ada_avgpool',
#                  topk = 4, param_attention = "qkvo", param_routing = False, diff_routing = False, soft_routing = False,
#                  mlp_ratio = 4, mlp_dwconv = False,
#                  side_dwconv = 5, before_attn_dwconv = 3, pre_norm = True, auto_pad = False):
#         super().__init__()
#         qk_dim = qk_dim or dim

#         # modules
#         if before_attn_dwconv > 0:
#             self.pos_embed = nn.Conv3d(dim, dim, kernel_size = before_attn_dwconv, padding = 1, groups = dim)
#         else:
#             self.pos_embed = lambda x: 0
#         self.norm1 = nn.LayerNorm(dim, eps = 1e-6)  # important to avoid attention collapsing
#         if topk > 0:
#             self.attn = BiLevelRoutingAttention(dim = dim, num_heads = num_heads, n_win = n_win, qk_dim = qk_dim,
#                                                 qk_scale = qk_scale, kv_per_win = kv_per_win,
#                                                 kv_downsample_ratio = kv_downsample_ratio,
#                                                 kv_downsample_kernel = kv_downsample_kernel,
#                                                 kv_downsample_mode = kv_downsample_mode,
#                                                 topk = topk, param_attention = param_attention,
#                                                 param_routing = param_routing,
#                                                 diff_routing = diff_routing, soft_routing = soft_routing,
#                                                 side_dwconv = side_dwconv,
#                                                 auto_pad = auto_pad)
#         elif topk == -1:
#             self.attn = Attention(dim = dim)
#         elif topk == -2:
#             self.attn = AttentionLePE(dim = dim, side_dwconv = side_dwconv)
#         elif topk == 0:
#             self.attn = nn.Sequential(Rearrange('n h w d c -> n c h d w'),  # compatiability
#                                       nn.Conv3d(dim, dim, 1),  # pseudo qkv linear
#                                       nn.Conv3d(dim, dim, 5, padding = 2, groups = dim),  # pseudo attention
#                                       nn.Conv3d(dim, dim, 1),  # pseudo out linear
#                                       Rearrange('n c h w d -> n h w d c')
#                                       )
#         self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
#         self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
#                                  DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
#                                  nn.GELU(),
#                                  nn.Linear(int(mlp_ratio * dim), dim)
#                                  )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         # tricks: layer scale & pre_norm/post_norm
#         if layer_scale_init_value > 0:
#             self.use_layer_scale = True
#             self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad = True)
#             self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad = True)
#         else:
#             self.use_layer_scale = False
#         self.pre_norm = pre_norm

#     def forward(self, x):
#         """
#         x: NCHW tensor
#         """
#         # conv pos embedding
#         x = x + self.pos_embed(x)
#         # permute to NHWC tensor for attention & mlp
#         x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)

#         # attention & mlp
#         if self.pre_norm:
#             if self.use_layer_scale:
#                 x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
#                 x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
#             else:
#                 x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
#                 x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
#         else:  # https://kexue.fm/archives/9009
#             if self.use_layer_scale:
#                 x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
#                 x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
#             else:
#                 x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
#                 x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

#         # permute back
#         x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)
#         return x


# class PBiFormer_Unet(nn.Module):
#     #[32, 64, 128, 256] embed_dim/head_dim=head个数 depth = [2, 2, 4, 2] head_dim = 8
#     def __init__(self, n_class = 14 , n_anchor=4, depth = [2, 2, 4, 2], in_chans = 1, embed_dim = [64, 128, 256, 512],qk_dims = [64, 128, 256, 512],
#                  head_dim = 8, qk_scale = None, representation_size = None,
#                  drop_path_rate = 0., drop_rate = 0.,
#                  use_checkpoint_stages = [],
#                  ########
#                 #  n_win = 5,
#                 #  n_win = 4,
#                  n_win = (4, 4, 2),
#                  side_dwconv = 5,
#                  kv_downsample_mode = 'identity',
#                  kv_per_wins = [-1, -1, -1, -1],
#                  topks = [1, 4, 16, 16],
#                  layer_scale_init_value = -1,
#                  param_routing = False, diff_routing = False, soft_routing = False,
#                  pre_norm = True,
#                  pe = None,
#                  pe_stages = [0],
#                  before_attn_dwconv = 3,
#                  auto_pad = True,
#                  kv_downsample_kernels = [4, 2, 1, 1],
#                  kv_downsample_ratios = [4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
#                  mlp_ratios = [4, 4, 4, 4],
#                  param_attention = 'qkvo',
#                  out_indices = (0, 1, 2, 3),
#                  mlp_dwconv = False):
#         """
#         Args:
#             depth (list): depth of each stage
#             img_size (int, tuple): input image size
#             in_chans (int): number of input channels
#             embed_dim (list): embedding dimension of each stage
#             head_dim (int): head dimension
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             drop_rate (float): dropout rate
#             attn_drop_rate (float): attention dropout rate
#             drop_path_rate (float): stochastic depth rate
#             norm_layer (nn.Module): normalization layer
#             conv_stem (bool): whether use overlapped patch stem
#         """
#         super(PBiFormer_Unet, self).__init__()
#         self.n_anchor = n_anchor
#         self.if_convskip = True
#         self.if_transskip = True
#         self.n_class = n_class
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

#         ############ downsample layers (patch embeddings) ######################
#         self.downsample_layers = nn.ModuleList()
#         # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
#         # down4
#         stem = nn.Sequential(
#             nn.Conv3d(in_chans, embed_dim[0] // 2, kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
#             nn.InstanceNorm3d(embed_dim[0] // 2),
#             nn.GELU(),
#             nn.Conv3d(embed_dim[0] // 2, embed_dim[0], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
#             nn.InstanceNorm3d(embed_dim[0]),
#         )
#         # DOWN2
#         # stem = nn.Sequential(
#         #     nn.Conv3d(in_chans, embed_dim[0], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
#         #     nn.InstanceNorm3d(embed_dim[0]),
#         # )
#         if (pe is not None) and 0 in pe_stages:
#             stem.append(get_pe_layer(emb_dim = embed_dim[0], name = pe))
#         if use_checkpoint_stages:
#             stem = checkpoint_wrapper(stem)
#         self.downsample_layers.append(stem)
#         # patch embedding
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                 nn.Conv3d(embed_dim[i], embed_dim[i + 1], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
#                 nn.InstanceNorm3d(embed_dim[i + 1])
#             )
#             if (pe is not None) and i + 1 in pe_stages:
#                 downsample_layer.append(get_pe_layer(emb_dim = embed_dim[i + 1], name = pe))
#             if use_checkpoint_stages:
#                 downsample_layer = checkpoint_wrapper(downsample_layer)
#             self.downsample_layers.append(downsample_layer)
#         ##########################################################################

#         self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
#         nheads = [dim // head_dim for dim in qk_dims]#[4,8,16,32]
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim = embed_dim[i], drop_path = dp_rates[cur + j],
#                         layer_scale_init_value = layer_scale_init_value,
#                         topk = topks[i],
#                         num_heads = nheads[i],
#                         n_win = n_win,
#                         qk_dim = qk_dims[i],
#                         qk_scale = qk_scale,
#                         kv_per_win = kv_per_wins[i],
#                         kv_downsample_ratio = kv_downsample_ratios[i],
#                         kv_downsample_kernel = kv_downsample_kernels[i],
#                         kv_downsample_mode = kv_downsample_mode,
#                         param_attention = param_attention,
#                         param_routing = param_routing,
#                         diff_routing = diff_routing,
#                         soft_routing = soft_routing,
#                         mlp_ratio = mlp_ratios[i],
#                         mlp_dwconv = mlp_dwconv,
#                         side_dwconv = side_dwconv,
#                         before_attn_dwconv = before_attn_dwconv,
#                         pre_norm = pre_norm,
#                         auto_pad = auto_pad) for j in range(depth[i])],
#             )
#             if i in use_checkpoint_stages:
#                 stage = checkpoint_wrapper(stage)
#             self.stages.append(stage)
#             cur += depth[i]

#         ##########################################################################
#         self.norm = nn.BatchNorm3d(embed_dim[-1])
#         # out_indices = [0, 1, 2, 3]
#         for i_layer in out_indices:
#             layer = nn.LayerNorm(embed_dim[i_layer])
#             layer_name = f'norm{i_layer}'
#             self.add_module(layer_name, layer)
#         # Representation layer
#         if representation_size:
#             self.num_features = representation_size
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(embed_dim, representation_size)),
#                 ('act', nn.Tanh())
#             ]))
#         else:
#             self.pre_logits = nn.Identity()


#         self.head = nn.Linear(embed_dim[-1], n_class) if n_class > 0 else nn.Identity()
#         self.apply(self._init_weights)
#         # down2
#         # self.up0 = DecoderBlock(embed_dim[3], embed_dim[2] , skip_channels = embed_dim[2] if self.if_transskip else 0,
#         #                         use_batchnorm = False)
#         # self.up1 = DecoderBlock(embed_dim[2], embed_dim[1], skip_channels = embed_dim[1]if self.if_transskip else 0,
#         #                         use_batchnorm = False)  # 384, 20, 20, 64
#         # self.up2 = DecoderBlock(embed_dim[1] , embed_dim[0], skip_channels = embed_dim[0] if self.if_transskip else 0,
#         #                         use_batchnorm = False)  # 384, 40, 40, 64
#         # self.up3 = DecoderBlock(embed_dim[0], 16, skip_channels = 16 if self.if_convskip else 0,
#         #                         use_batchnorm = False)  # 384, 80, 80, 128   
             
#         #未改
#         self.up0 = DecoderBlock(embed_dim[3], embed_dim[2] , skip_channels = embed_dim[2] if self.if_transskip else 0,
#                                 use_batchnorm = False)
#         self.up1 = DecoderBlock(embed_dim[2], embed_dim[1], skip_channels = embed_dim[1]if self.if_transskip else 0,
#                                 use_batchnorm = False)  # 384, 20, 20, 64
#         self.up2 = DecoderBlock(embed_dim[1] , embed_dim[0], skip_channels = embed_dim[0] if self.if_transskip else 0,
#                                 use_batchnorm = False)  # 384, 40, 40, 64
#         self.up3 = DecoderBlock(embed_dim[0], embed_dim[0] // 2, skip_channels = embed_dim[0] // 2 if self.if_convskip else 0,
#                                 use_batchnorm = False)  # 384, 80, 80, 128
#         self.up4 = DecoderBlock(embed_dim[0] // 2, 16,
#                                 skip_channels = 16 if self.if_convskip else 0,
#                                 use_batchnorm = False)  # 384, 160, 160, 256
        
#         self.c1 = Conv3dReLU(1, embed_dim[0] // 2, 3, 1, use_batchnorm = False)
#         self.c2 = Conv3dReLU(1, 16, 3, 1, use_batchnorm = False)

#         self.final_conv = nn.Conv3d(16, self.n_class, 1)
#         self.avg_pool = nn.AvgPool3d(3, stride = 2, padding = 1)

#         # self.early_down1 = nn.Conv3d(f_maps[0], f_maps[2], kernel_size=1, stride=4)
#         # self.early_down2 = nn.Conv3d(f_maps[1], f_maps[2], kernel_size=1, stride=2)
#         # self.pre_layer = nn.Conv3d(3*f_maps[2], n_anchor*(3+n_class), kernel_size=1, stride=1)
#         self.early_down1 = nn.Conv3d(16, 64, kernel_size=1, stride=4)
#         self.early_down2 = nn.Conv3d(32, 64, kernel_size=1, stride=2)
#         # self.pre_layer = nn.Conv3d(64+64+64, n_anchor*(3+n_class), kernel_size=1, stride=1)
#         self.pre_layer = nn.Conv3d(64, n_anchor*(3+n_class), kernel_size=1, stride=1)

#         # self.pre_layer = nn.Conv3d(16, n_anchor * (3 + n_class), kernel_size=1, stride=1)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std = .02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def get_classifier(self):
#         return self.head

#     def down_features(self, x):
#         outs = []
#         for i in range(4):
#             # print(x.size(),"111111111111111111")#torch.Size([1, 1, 128, 128, 64]) 
#             x = self.downsample_layers[i](x)
#             # print(x.size(),"22222222222222222")#torch.Size([1, 64, 32, 32, 16]) 
#             x = self.stages[i](x)
#             # print(x.size(),"33333333333333")
#             x = x.permute(0,2,3,4,1)
#             norm_layer = getattr(self, f'norm{i}')
#             x = norm_layer(x)
#             x = x.permute(0,4,1,2,3)
#             # print(x.size(),"4444444444444")
#             outs.append(x)
#         return outs

#     def forward(self, x):
#         moving = x[:, 0:1, :, :]  # [2, 1, 160, 192, 224]
#         # print(x.size(),"--------------0")
#         #down4
#         # x_s0 = x.clone()  # [2, 2, 160, 192, 224]
#         # x_s1 = self.avg_pool(x)  # [2, 2, 80, 96, 112]
#         # f4 = self.c1(x_s1)  # [2, 48, 80, 96, 112]
#         # f5 = self.c2(x_s0)  # [2, 16, 160, 192, 224]

#         #down2
#         # f5 = self.c2(x)  # [2, 16, 160, 192, 224]

#         out_feats = self.down_features(x)
#         f1 = out_feats[-2]  # [2, 384, 10, 12, 14]
#         f2 = out_feats[-3]  # [2, 192, 20, 24, 28]
#         f3 = out_feats[-4]  # [2, 96, 40, 48, 56]

#         x = self.up0(out_feats[-1], f1)  # [2, 384, 10, 12, 14]
#         x = self.up1(x, f2)  # [2, 192, 20, 24, 28]
#         x = self.up2(x, f3)  # [2, 96, 40, 48, 56]

#         # down2
#         # x = self.up3(x, f5)  # [2, 48, 80, 96, 112]
#         # down4
#         # x_s3 = x.clone()
#         # x = self.up3(x, f4)  # [2, 48, 80, 96, 112]
#         # x_s4 = x.clone()
#         # x = self.up4(x, f5)  # [2, 16, 160, 192, 224]
#         # early_out1 = self.early_down1(x)  # 下采样特征 1
#         # early_out2 = self.early_down2(x_s4)  # 下采样特征 2
#         # anchor_out = self.pre_layer(torch.cat([early_out1, early_out2, x_s3], dim=1))  # 拼接特征

#         # Anchor-based 预测
#         # anchor_out = self.pre_layer(x)  # 直接在 x 上应用 pre_layer
#         # early_out1 = self.early_down1(f5)  # 下采样特征 1
#         # early_out2 = self.early_down2(f4)  # 下采样特征 2
#         # anchor_out = self.pre_layer(torch.cat([early_out1, early_out2, x], dim=1))  # 拼接特征


#         anchor_out = self.pre_layer(x)  # 直接在 x 上应用 pre_layer
#         anchor_out = anchor_out.reshape(anchor_out.shape[0], self.n_anchor, 3 + self.n_class,
#                                         anchor_out.shape[2], anchor_out.shape[3], anchor_out.shape[4])
#         anchor_out = anchor_out.permute(0, 3, 4, 5, 1, 2)  # 调整输出格式
#         return anchor_out
#         # out = self.final_conv(x)
#         # return out


# if __name__ == '__main__':
#     # config = CONFIGS['BiFormer']
#     model = PBiFormer_Unet()
#     summary(model, (1,1, 128, 128, 64), depth=3)
#     #out [1, 14, 160, 160, 160]


# import configs_TransMorph as configs
import torch
from torch import nn as nn
from torch.nn import functional as F
import importlib


import itertools
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np

from typing import Tuple
from collections import OrderedDict
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper

from .BiFormer.bra_legacy import BiLevelRoutingAttention
from .BiFormer._common import Attention, AttentionLePE, DWConv

import ml_collections
from torchinfo import summary
from typing import List, Optional, Tuple, Union


        
def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    # if name == 'sum':
    #     return Summer(PositionalEncodingPermute2D(emb_dim))
    # elif name == 'npe.sin':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='sin')
    # elif name == 'npe.coord':
    #     return NeuralPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='coord')
    # elif name == 'hpe.conv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='conv', res_shortcut=True)
    # elif name == 'hpe.dsconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='dsconv', res_shortcut=True)
    # elif name == 'hpe.pointconv':
    #     return HybridPE(emb_dim=emb_dim, pe_dim=pe_dim, mode='pointconv', res_shortcut=True)
    else:
        raise ValueError(f'PE name {name} is not surpported!')

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True, up_factor=2):
#         super().__init__()
#         self.srb = SRB(in_channels, up_factor=up_factor)
#         self.conv1 = Conv3dReLU(
#             in_channels + skip_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )
#         self.conv2 = Conv3dReLU(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             padding=1,
#             use_batchnorm=use_batchnorm,
#         )

#     def forward(self, x, skip=None):

#         x = self.srb(x)

#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)
        
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class PPB(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super(PPB, self).__init__()
        self.pool_scales = pool_scales
        self.ppb_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(scale),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for scale in pool_scales
        ])
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels + len(pool_scales) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        pooled_features = [x]
        for ppb_layer in self.ppb_layers:
            pooled = ppb_layer(x)
            pooled_features.append(F.interpolate(pooled, size=input_size, mode='trilinear', align_corners=False))
        out = torch.cat(pooled_features, dim=1)
        return self.final_conv(out)


from einops import rearrange

class SRB(nn.Module):
    def __init__(self, in_channels, up_factor=2):
        super(SRB, self).__init__()
        self.up_factor = up_factor
        self.conv = nn.Conv3d(in_channels, in_channels * (up_factor ** 3), kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(in_channels * (up_factor ** 3))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # 5D 像素重排
        B, C, D, H, W = x.shape
        x = rearrange(
            x,
            "B (C r1 r2 r3) D H W -> B C (D r1) (H r2) (W r3)",
            r1=self.up_factor,
            r2=self.up_factor,
            r3=self.up_factor,
        )
        return x
class Block(nn.Module):
    def __init__(self, dim, drop_path = 0., layer_scale_init_value = -1,
                 num_heads = 8, n_win = (4,4,2), qk_dim = None, qk_scale = None,
                 kv_per_win = 4, kv_downsample_ratio = 4, kv_downsample_kernel = None,
                 kv_downsample_mode = 'ada_avgpool',
                 topk = 4, param_attention = "qkvo", param_routing = False, diff_routing = False, soft_routing = False,
                 mlp_ratio = 4, mlp_dwconv = False,
                 side_dwconv = 5, before_attn_dwconv = 3, pre_norm = True, auto_pad = False):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv3d(dim, dim, kernel_size = before_attn_dwconv, padding = 1, groups = dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim = dim, num_heads = num_heads, n_win = n_win, qk_dim = qk_dim,
                                                qk_scale = qk_scale, kv_per_win = kv_per_win,
                                                kv_downsample_ratio = kv_downsample_ratio,
                                                kv_downsample_kernel = kv_downsample_kernel,
                                                kv_downsample_mode = kv_downsample_mode,
                                                topk = topk, param_attention = param_attention,
                                                param_routing = param_routing,
                                                diff_routing = diff_routing, soft_routing = soft_routing,
                                                side_dwconv = side_dwconv,
                                                auto_pad = auto_pad)
        elif topk == -1:
            self.attn = Attention(dim = dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim = dim, side_dwconv = side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w d c -> n c h d w'),  # compatiability
                                      nn.Conv3d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv3d(dim, dim, 5, padding = 2, groups = dim),  # pseudo attention
                                      nn.Conv3d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w d -> n h w d c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad = True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad = True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)
        return x


class PBiFormer_Unet(nn.Module):
    #[32, 64, 128, 256] embed_dim/head_dim=head个数 depth = [2, 2, 4, 2] head_dim = 8
    def __init__(self, n_class = 14 , n_anchor=4, depth = [2, 2, 4, 2], in_chans = 1, embed_dim = [64, 128, 256, 512],qk_dims = [64, 128, 256, 512],
                 head_dim = 8, qk_scale = None, representation_size = None,
                 drop_path_rate = 0., drop_rate = 0.,
                 use_checkpoint_stages = [],
                 ########
                #  n_win = 5,
                #  n_win = 4,
                 n_win = (4, 4, 2),
                 side_dwconv = 5,
                 kv_downsample_mode = 'identity',
                 kv_per_wins = [-1, -1, -1, -1],
                 topks = [1, 4, 16, 16],
                 layer_scale_init_value = -1,
                 param_routing = False, diff_routing = False, soft_routing = False,
                 pre_norm = True,
                 pe = None,
                 pe_stages = [0],
                 before_attn_dwconv = 3,
                 auto_pad = True,
                 kv_downsample_kernels = [4, 2, 1, 1],
                 kv_downsample_ratios = [4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios = [4, 4, 4, 4],
                 param_attention = 'qkvo',
                 out_indices = (0, 1, 2, 3),
                 mlp_dwconv = False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super(PBiFormer_Unet, self).__init__()
        self.n_anchor = n_anchor
        self.if_convskip = True
        self.if_transskip = True
        self.n_class = n_class
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        # down4
        stem = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim[0] // 2, kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
            nn.InstanceNorm3d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim[0] // 2, embed_dim[0], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
            nn.InstanceNorm3d(embed_dim[0]),
        )
        # DOWN2
        # stem = nn.Sequential(
        #     nn.Conv3d(in_chans, embed_dim[0], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
        #     nn.InstanceNorm3d(embed_dim[0]),
        # )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim = embed_dim[0], name = pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)
        # patch embedding
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv3d(embed_dim[i], embed_dim[i + 1], kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1)),
                nn.InstanceNorm3d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim = embed_dim[i + 1], name = pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]#[4,8,16,32]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim = embed_dim[i], drop_path = dp_rates[cur + j],
                        layer_scale_init_value = layer_scale_init_value,
                        topk = topks[i],
                        num_heads = nheads[i],
                        n_win = n_win,
                        qk_dim = qk_dims[i],
                        qk_scale = qk_scale,
                        kv_per_win = kv_per_wins[i],
                        kv_downsample_ratio = kv_downsample_ratios[i],
                        kv_downsample_kernel = kv_downsample_kernels[i],
                        kv_downsample_mode = kv_downsample_mode,
                        param_attention = param_attention,
                        param_routing = param_routing,
                        diff_routing = diff_routing,
                        soft_routing = soft_routing,
                        mlp_ratio = mlp_ratios[i],
                        mlp_dwconv = mlp_dwconv,
                        side_dwconv = side_dwconv,
                        before_attn_dwconv = before_attn_dwconv,
                        pre_norm = pre_norm,
                        auto_pad = auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm3d(embed_dim[-1])
        # out_indices = [0, 1, 2, 3]
        for i_layer in out_indices:
            layer = nn.LayerNorm(embed_dim[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()


        self.head = nn.Linear(embed_dim[-1], n_class) if n_class > 0 else nn.Identity()
        self.apply(self._init_weights)
          
        #未改
        self.up0 = DecoderBlock(embed_dim[3], embed_dim[2] , skip_channels = embed_dim[2] if self.if_transskip else 0,
                                use_batchnorm = False)
        self.up1 = DecoderBlock(embed_dim[2], embed_dim[1], skip_channels = embed_dim[1]if self.if_transskip else 0,
                                use_batchnorm = False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim[1] , embed_dim[0], skip_channels = embed_dim[0] if self.if_transskip else 0,
                                use_batchnorm = False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim[0], embed_dim[0] // 2, skip_channels = embed_dim[0] // 2 if self.if_convskip else 0,
                                use_batchnorm = False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim[0] // 2, 16,
                                skip_channels = 16 if self.if_convskip else 0,
                                use_batchnorm = False)  # 384, 160, 160, 256
        
        self.c1 = Conv3dReLU(1, embed_dim[0] // 2, 3, 1, use_batchnorm = False)
        self.c2 = Conv3dReLU(1, 16, 3, 1, use_batchnorm = False)

        self.final_conv = nn.Conv3d(16, self.n_class, 1)
        self.avg_pool = nn.AvgPool3d(3, stride = 2, padding = 1)


        self.early_down1 = nn.Conv3d(16, 64, kernel_size=1, stride=4)
        self.early_down2 = nn.Conv3d(32, 64, kernel_size=1, stride=2)

        self.pre_layer = nn.Conv3d(64, n_anchor*(3+n_class), kernel_size=1, stride=1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def down_features(self, x):
        outs = []
        for i in range(4):
            # print(x.size(),"111111111111111111")#torch.Size([1, 1, 128, 128, 64]) 
            x = self.downsample_layers[i](x)
            # print(x.size(),"22222222222222222")#torch.Size([1, 64, 32, 32, 16]) 
            x = self.stages[i](x)
            # print(x.size(),"33333333333333")
            x = x.permute(0,2,3,4,1)
            norm_layer = getattr(self, f'norm{i}')
            x = norm_layer(x)
            x = x.permute(0,4,1,2,3)
            # print(x.size(),"4444444444444")
            outs.append(x)
        return outs

    def forward(self, x):
        moving = x[:, 0:1, :, :]  # [2, 1, 160, 192, 224]
        # print(x.size(),"--------------0")
        #down4
        x_s0 = x.clone()  # [2, 2, 160, 192, 224]
        x_s1 = self.avg_pool(x)  # [2, 2, 80, 96, 112]
        f4 = self.c1(x_s1)  # [2, 48, 80, 96, 112]
        f5 = self.c2(x_s0)  # [2, 16, 160, 192, 224]

        #down2
        # f5 = self.c2(x)  # [2, 16, 160, 192, 224]

        out_feats = self.down_features(x)
        f1 = out_feats[-2]  # [2, 384, 10, 12, 14]
        f2 = out_feats[-3]  # [2, 192, 20, 24, 28]
        f3 = out_feats[-4]  # [2, 96, 40, 48, 56]

        x = self.up0(out_feats[-1], f1)  # [2, 384, 10, 12, 14]
        x = self.up1(x, f2)  # [2, 192, 20, 24, 28]
        x = self.up2(x, f3)  # [2, 96, 40, 48, 56]


        # Anchor-based 预测
        anchor_out = self.pre_layer(x)  # 直接在 x 上应用 pre_layer


  
        anchor_out = anchor_out.reshape(anchor_out.shape[0], self.n_anchor, 3 + self.n_class,
                                        anchor_out.shape[2], anchor_out.shape[3], anchor_out.shape[4])
        anchor_out = anchor_out.permute(0, 3, 4, 5, 1, 2)  # 调整输出格式
        return anchor_out
        # out = self.final_conv(x)
        # return out


if __name__ == '__main__':
    # config = CONFIGS['BiFormer']
    model = PBiFormer_Unet()
    summary(model, (1,1, 128, 128, 64), depth=3)
    #out [1, 14, 160, 160, 160]












