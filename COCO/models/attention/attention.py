import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ammformer.ammformer_attn import AMMformerAttention
from .softmax.softmax_attn import SoftmaxAttention
from .linformer.linformer_attn import LinformerAttention
from .performer.performer_attn import PerformerAttention
from .nystrom.nystrom_attn import NystromAttention

class Attention(nn.Module):
    def __init__(self, dim, num_head, seq_len, dropout=0.1, attn_type='softmax', args=None, is_decoder=False):
        super().__init__()

        self.grad_checkpointing = args.grad_checkpointing if args is not None else False
        self.seq_len = seq_len

        self.dim = dim
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.attn_type = attn_type
        assert self.head_dim * self.num_head == self.dim, "dim must be divided by num_head"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * self.dim, self.dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.dim))

        if attn_type == 'ammformer':
            self.attn = AMMformerAttention(self.num_head, self.head_dim, dropout, args.landmarks, args.window_size, is_decoder=is_decoder)
        elif attn_type == 'linformer':
            self.attn = LinformerAttention(self.num_head, self.head_dim, dropout, args.max_seq_len)
        elif attn_type == 'performer':
            self.attn = PerformerAttention(self.num_head, self.head_dim, dropout)
        elif attn_type == 'nystrom':
            self.attn = NystromAttention(self.num_head, self.head_dim, dropout, args.landmarks, args.max_seq_len, args.conv_kernel_size)
        elif attn_type == 'softmax_':
            self.attn = SoftmaxAttention(self.num_head, self.head_dim, dropout)
        else:
            raise Exception()

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.ff.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # tranpose HWxNxC to NxHWxC
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        if self.attn_type == 'performer':
            Q = F.linear(query, self.in_proj_weight[:self.dim, :], self.in_proj_bias[:self.dim])
            K = F.linear(key, self.in_proj_weight[self.dim:2*self.dim, :], self.in_proj_bias[self.dim:2*self.dim])
            V = F.linear(value, self.in_proj_weight[2*self.dim:, :], self.in_proj_bias[2*self.dim:])
        else:
            Q = self.split_heads(F.linear(query, self.in_proj_weight[:self.dim, :], self.in_proj_bias[:self.dim]))
            K = self.split_heads(F.linear(key, self.in_proj_weight[self.dim:2*self.dim, :], self.in_proj_bias[self.dim:2*self.dim]))
            V = self.split_heads(F.linear(value, self.in_proj_weight[2*self.dim:, :], self.in_proj_bias[2*self.dim:]))

        # with torch.cuda.amp.autocast(enabled = False):
        if self.grad_checkpointing:
            attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), 
                                  attn_mask, key_padding_mask)
        else:
            attn_out = self.attn(Q, K, V, 
                                 attn_mask, key_padding_mask)

        if self.attn_type == 'performer':
            out = self.ff(attn_out)
        else:
            out = self.ff(self.combine_heads(attn_out))

        out = out.transpose(0, 1)
        return out, None

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.contiguous().view(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.contiguous().view(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
