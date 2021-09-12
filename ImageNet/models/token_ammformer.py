"""
Take the LordFormer as Transformer
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .transformer_block import Mlp

class AMMAttention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_landmarks=256, window_size=0):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = head_dim ** 0.5
        self.landmarks = num_landmarks
        self.window_size = window_size

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q /= self.scale

        keys_head_dim = k.size(-1)
        segs = N // self.landmarks
        if (N % self.landmarks == 0):
            keys_landmarks = k.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
            values_landmarks = v.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
        else:
            num_k = (segs + 1) * self.landmarks - N
            keys_landmarks_f = k[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            keys_landmarks_l = k[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

            values_landmarks_f = v[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            values_landmarks_l = v[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            values_landmarks = torch.cat((values_landmarks_f, values_landmarks_l), dim = -2)
        attn = q @ keys_landmarks.transpose(-1, -2)
        # FIXME support masking
        if attn_mask == None:
            attn = attn.softmax(dim=-1)
        else:
            attn = torch.nn.functional.softmax(attn - 1e9*(1 - attn_mask[:, None, None, :self.landmarks]), dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ values_landmarks

        x = x.transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = v.squeeze(1) + x 
        return x

class Token_ammformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AMMAttention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
