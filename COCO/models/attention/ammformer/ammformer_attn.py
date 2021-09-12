import math

import torch
import torch.nn.functional as F
from torch import nn

class AMMformerAttention(nn.Module):
    def __init__(self, num_heads, head_dim, dropout=0., num_landmarks=256, window_size=0, is_decoder=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** 0.5

        self.landmarks = num_landmarks
        self.window_size = window_size

        self.is_decoder = is_decoder

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, num_heads, N, C = key.shape
        query /= self.scale

        if self.is_decoder:
            attn = torch.matmul(query, key.transpose(-1, -2))
            if key_padding_mask is not None:
                attn = attn.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf"),
                )
            attn = F.softmax(attn, dim = -1)
            attn = self.attn_drop(attn)
            x = torch.matmul(attn, value)
        else:
            keys_head_dim = key.size(-1)
            segs = N // self.landmarks
            if (N % self.landmarks == 0):
                keys_landmarks = key.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
                values_landmarks = value.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
            else:
                num_k = (segs + 1) * self.landmarks - N
                keys_landmarks_f = key[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
                keys_landmarks_l = key[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
                keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

                values_landmarks_f = value[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
                values_landmarks_l = value[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
                values_landmarks = torch.cat((values_landmarks_f, values_landmarks_l), dim = -2)
            attn = query @ keys_landmarks.transpose(-1, -2)
            if key_padding_mask is not None:
                attn = attn.masked_fill(
                    key_padding_mask[:, None, None, :self.landmarks],
                    float("-inf"),
                )
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ values_landmarks

        return x


    def extra_repr(self):
        return f'num_landmarks={self.landmarks}, window_size={self.window_size}'
