import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.attention.attention import Attention

seq_len, bsz, dim = 4, 2, 8
nhead = 4

device = torch.device('cuda')
attn1 = nn.MultiheadAttention(dim, nhead, dropout=0.)
attn2 = Attention(dim, nhead, seq_len, dropout=0., attn_type='softmax_', args=None)
norm = nn.LayerNorm(dim)

attn2.W_q.weight.data = attn1.in_proj_weight[: dim]
attn2.W_q.bias.data = attn1.in_proj_bias[: dim]
attn2.W_k.weight.data = attn1.in_proj_weight[dim : 2*dim]
attn2.W_k.bias.data = attn1.in_proj_bias[dim : 2*dim]
attn2.W_v.weight.data = attn1.in_proj_weight[2*dim :]
attn2.W_v.bias.data = attn1.in_proj_bias[2*dim :]
attn2.ff.weight.data = attn1.out_proj.weight
attn2.ff.bias.data = attn1.out_proj.bias

input = torch.ones((seq_len, bsz, dim))
mask = torch.zeros((bsz, seq_len)).to(torch.bool)
for i in range(seq_len):
    for j in range(bsz):
        if (i == seq_len - 1):
            mask[j][i] = True
        for k in range(dim):
            input[i][j][k] = 0.1*i + 0.01*j + 0.001*k

out1 = norm(attn1(input, input, input, key_padding_mask=mask)[0]).permute(1, 0, 2)
out2 = norm(attn2(input, input, input, key_padding_mask=mask)[0]).permute(1, 0, 2)

print ("\nInput")
print (input.shape)
print (input)

print ("\nPyTorch Attention")
print (out1.shape)
print (out1)

print ("\nSoftmax Attention")
print (out2.shape)
print (out2)

print (torch.equal(out1, out2))
for i in range(len(out1)):
    for j in range(len(out1[0])):
        for k in range(len(out1[0][0])):
            print (i, j, k, out1[i][j][k].item(), out2[i][j][k].item())

print (torch.equal(out1, out2))