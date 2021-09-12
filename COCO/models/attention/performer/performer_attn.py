import torch
import torch.nn as nn
import math

class PerformerAttention(nn.Module):
    def __init__(self, num_heads, head_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.drop_attn = nn.Dropout(dropout)
        self.dim = num_heads * head_dim
        
        self.epsilon = 1e-8  # for stable in division
        self.m = int(self.dim * 0.5)
        self.w = torch.randn(self.m, self.dim)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, Q, K, V, attn_mask=None, key_padding_mask=None):
        if key_padding_mask is not None:
            K = K * key_padding_mask[:, :, None]
            V = V * key_padding_mask[:, :, None]
        kp, qp = self.prm_exp(K), self.prm_exp(Q)  # (B, T, m), (B, T, m)
        
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        
        kptv = torch.einsum('bin,bim->bnm', V.float(), kp)  # (B, self.in_dim, m)
        
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.dim) + self.epsilon)  # (B, T, self.in_dim)/Diag
        
        y = y / (D.repeat(1, 1, self.dim) + self.epsilon)
        y = self.drop_attn(y)
        return y

        return self.attn_fn(
            Q / math.sqrt(math.sqrt(self.head_dim)),
            K / math.sqrt(math.sqrt(self.head_dim)) * key_padding_mask[:, None, :, None],
            V * key_padding_mask[:, None, :, None])

    def extra_repr(self):
        return f'rp_dim={self.rp_dim}, kernel_type={self.kernel_type}'
