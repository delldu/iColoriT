# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 04日 星期日 16:41:59 CST
# ***
# ************************************************************************************/
#
import pdb

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from . import data

class Generator(nn.Module):
    """
    # L + ab + mask
    input_nc = 1
    output_nc = 2
    num_in = input_nc + output_nc + 1
    norm_layer = color.get_norm_layer(norm_type="batch")
    """

    def forward(self, rgba):
        # input = torch.cat((input_A, input_B, mask_B), dim=1)
        input = data.rgba2lab(rgba)
        lab_l = input[:, 0:1, :, :]
        # lab_m = input[:, 3:4, :, :]

        out_reg = self.model_out(conv10_2)

        # out_class
        output = data.Lab2rgb(lab_l, out_reg)
        return output.clamp(0.0, 1.0)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.,
                 proj_drop=0., attn_head_dim=None, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim))

        # relative positional bias option
        self.window_size = window_size
        self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        # trunc_normal_(self.rpb_table, std=.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, h*w
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # h*w, h*w
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
        attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, 
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            window_size=window_size)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_shape = (img_size // patch_size, img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.proj -- Conv2d(4, 192, kernel_size=(16, 16), stride=(16, 16))

    def forward(self, x):
        # x.size() -- torch.Size([10, 4, 224, 224])
        x = self.proj(x).flatten(2).transpose(1, 2) # torch.Size([10, 196, 192])
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class CnnHead(nn.Module):
    def __init__(self, embed_dim, num_classes, window_size):
        super().__init__()
        # self = CnnHead(
        #   (head): Conv2d(192, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=reflect)
        # )
        # embed_dim = 192
        # num_classes = 512
        # window_size = 14
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.window_size = window_size

        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
        x = self.head(x)
        x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c')
        return x


class IColoriT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    
    Small configuration:
        embed_dim=384,
        depth=12,
        num_heads=6,
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=4, num_classes=512, 
                embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_classes = num_classes
        assert num_classes == 2 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.in_chans = in_chans


        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # 2

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            window_size=img_size // patch_size)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.head = CnnHead(embed_dim, num_classes, window_size=img_size // patch_size)

        self.tanh = nn.Tanh()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def forward_features(self, x, mask):
    #     # (Pdb) x.size() -- torch.Size([10, 3, 224, 224])
    #     # x.max(), x.min() -- 0.6790, -0.5000

    #     # (Pdb) mask.size() -- torch.Size([10, 12544]), most of mask is True
    #     # 224 * 224 /4 -- 12544

    #     # mask is 1D of 2D if 2D
    #     B, _, H, W = x.shape
    #     assert mask.dim() == 2, f'Check the mask dimension mask.dim() == 2 but {mask.dim()}.'

    #     _, L = mask.shape
    #     # assume square inputs
    #     hint_size = int(math.sqrt(H * W // L)) # -- 2
    #     _device = '.cuda' if x.device.type == 'cuda' else ''

    #     # hint location = 0, non-hint location = 1
    #     mask = torch.reshape(mask, (B, H // hint_size, W // hint_size))
    #     _mask = mask.unsqueeze(1).type(f'torch{_device}.FloatTensor')
    #     _full_mask = F.interpolate(_mask, scale_factor=hint_size)  # Needs to be Float
    #     full_mask = _full_mask.type(f'torch{_device}.BoolTensor')

    #     # mask ab channels
    #     _avg_x = F.interpolate(x, size=(H // hint_size, W // hint_size), mode='bilinear')
    #     _avg_x[:, 1, :, :].masked_fill_(mask.squeeze(1), 0)
    #     _avg_x[:, 2, :, :].masked_fill_(mask.squeeze(1), 0)
    #     x_ab = F.interpolate(_avg_x, scale_factor=hint_size, mode='nearest')[:, 1:, :, :]
    #     x = torch.cat((x[:, 0, :, :].unsqueeze(1), x_ab), dim=1)

    #     if self.in_chans == 4: # True
    #         x = torch.cat((x, 1.0 - _full_mask), dim=1)


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()  # (B, 14*14, 768)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # x.size() -- torch.Size([10, 196, 192])

        return x

    def forward(self, rgba):
        B, C, H, W = rgba.shape
        h = H//self.patch_size
        w = W//self.patch_size

        fake_lab = data.rgba2lab(rgba) # with mask
        lab_l = fake_lab[:, 0:1, :, :]


        # (Pdb) x.size(), mask.size()
        # (torch.Size([10, 3, 224, 224]), torch.Size([10, 12544]))
        fake_lab = self.forward_features(fake_lab) # torch.Size([10, 196, 192])
        fake_lab = self.head(fake_lab)
        fake_lab = self.tanh(fake_lab)
        # x.size() -- torch.Size([10, 196, 512])

        pred_ab = rearrange(fake_lab, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=self.patch_size, p2=self.patch_size)
        pred_ab = rearrange(pred_ab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                        h=h, w=w, p1=self.patch_size, p2=self.patch_size)

        output = data.Lab2rgb(lab_l, pred_ab)
        return output

