import math
import torch
from torch import nn
import torch.nn.functional as F 

class LinformerAttention(nn.Module):
    def __init__(self, num_heads, head_dim, dropout, max_seq_len=1764):
        super().__init__()
        self.drop_attn = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = max_seq_len

        self.E = nn.Parameter(torch.Tensor(self.num_heads, 256, self.seq_len))
        self.F = nn.Parameter(torch.Tensor(self.num_heads, 256, self.seq_len))
        torch.nn.init.normal_(self.E, std = 0.02)
        torch.nn.init.normal_(self.F, std = 0.02)

    def forward(self, Q, K, V,  attn_mask=None, key_padding_mask=None):
        if key_padding_mask is not None:
            K = K * key_padding_mask[:, None, :, None]
            V = V * key_padding_mask[:, None, :, None]

        K = torch.matmul(self.E, K)
        V = torch.matmul(self.F, V)

        dot = torch.matmul(Q, K.transpose(-2, -1))
        dot = dot / math.sqrt(self.head_dim)

        attn = F.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)
        return X
