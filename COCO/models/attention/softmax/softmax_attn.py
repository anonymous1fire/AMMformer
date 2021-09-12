import math
import torch
from torch import nn
import torch.nn.functional as F 

class SoftmaxAttention(nn.Module):
    def __init__(self, num_heads, head_dim, dropout=0.):
        super().__init__()
        self.drop_attn = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, Q, K, V,  attn_mask = None, key_padding_mask = None):
        bsz, _, tgt_len, _ = Q.shape
        _, _, src_len, _ = K.shape
        Q = Q.contiguous().view(-1, tgt_len, self.head_dim)
        K = K.contiguous().view(-1, src_len, self.head_dim)
        V = V.contiguous().view(-1, src_len, self.head_dim)

        # Q = Q / math.sqrt(self.head_dim)
        dot = torch.bmm(Q, K.transpose(1, 2))
        assert list(dot.size()) == [bsz * self.num_heads, tgt_len, src_len]
        dot = dot / math.sqrt(self.head_dim)
        # dot = dot - 1e6 * (1 - key_padding_mask[:, None, None, :])
        
        if key_padding_mask is not None:
            dot = dot.view(bsz, self.num_heads, tgt_len, src_len)
            dot = dot.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            dot = dot.view(bsz * self.num_heads, tgt_len, src_len)

        attn = F.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.bmm(attn, V)
        
        X = X.view(bsz, self.num_heads, tgt_len, self.head_dim)
        return X
