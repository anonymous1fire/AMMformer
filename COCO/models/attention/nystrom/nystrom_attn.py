import math
import torch
from torch import nn
import torch.nn.functional as F 

class NystromAttention(nn.Module):
    def __init__(self, num_heads, head_dim, dropout=0., num_landmarks=64, max_seq_len=1764, conv_kernel_size=33, inv_coeff_init_option=None):
        super().__init__()
        self.drop_attn = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.landmarks = num_landmarks
        
        if inv_coeff_init_option is not None:
            self.init_option = inv_coeff_init_option
        else:
            self.init_option = "original"

        # self.use_conv = (conv_kernel_size is not None)
        # print (self.use_conv)
        # if self.use_conv:
        #     self.conv = nn.Conv2d(
        #         in_channels = self.num_heads, out_channels = self.num_heads,
        #         kernel_size = (conv_kernel_size, 1), padding = (conv_kernel_size // 2, 0),
        #         bias = False,
        #         groups = self.num_heads)

    def forward(self, Q, K, V, attn_mask=None, key_padding_mask=None):
        key_padding_mask = key_padding_mask.float()
        Q = Q * key_padding_mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        K = K * key_padding_mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        bsz, _, tgt_len, _ = Q.shape
        _, _, src_len, keys_head_dim = K.shape

        segs = src_len // self.landmarks
        if (src_len % self.landmarks == 0):
            Q_landmarks = Q.reshape(bsz, self.num_heads, self.landmarks, src_len // self.landmarks, keys_head_dim).mean(dim = -2)
            K_landmarks = K.reshape(bsz, self.num_heads, self.landmarks, src_len // self.landmarks, keys_head_dim).mean(dim = -2)
        else:
            num_k = (segs + 1) * self.landmarks - src_len
            K_landmarks_f = K[:, :, :num_k * segs, :].reshape(bsz, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            K_landmarks_l = K[:, :, num_k * segs:, :].reshape(bsz, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            K_landmarks = torch.cat((K_landmarks_f, K_landmarks_l), dim = -2)

            Q_landmarks_f = Q[:, :, :num_k * segs, :].reshape(bsz, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            Q_landmarks_l = Q[:, :, num_k * segs:, :].reshape(bsz, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            Q_landmarks = torch.cat((Q_landmarks_f, Q_landmarks_l), dim = -2)

        kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - key_padding_mask[:, None, None, :]), dim = -1)
        a = torch.matmul(kernel_1, self.iterative_inv(kernel_2))
        b = torch.matmul(kernel_3, V)
        X = torch.matmul(a, b)

        # if self.use_conv:
        #     X += self.conv(V * key_padding_mask[:, None, :, None])
        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V


    # ========== For testing flops ==========
    # def forward(self, Q, K, V, attn_mask=None, key_padding_mask=None):
    #     key_padding_mask = key_padding_mask.float()
    #     Q = Q * key_padding_mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
    #     K = K * key_padding_mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

    #     bsz, _, tgt_len, _ = Q.shape
    #     _, _, src_len, keys_head_dim = K.shape

    #     segs = src_len // self.landmarks
    #     if (src_len % self.landmarks == 0):
    #         Q_landmarks = Q.reshape(bsz, self.num_heads, self.landmarks, src_len // self.landmarks, keys_head_dim).mean(dim = -2)
    #         K_landmarks = K.reshape(bsz, self.num_heads, self.landmarks, src_len // self.landmarks, keys_head_dim).mean(dim = -2)
    #     else:
    #         num_k = (segs + 1) * self.landmarks - src_len
    #         K_landmarks_f = K[:, :, :num_k * segs, :].reshape(bsz, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
    #         K_landmarks_l = K[:, :, num_k * segs:, :].reshape(bsz, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
    #         K_landmarks = torch.cat((K_landmarks_f, K_landmarks_l), dim = -2)

    #         Q_landmarks_f = Q[:, :, :num_k * segs, :].reshape(bsz, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
    #         Q_landmarks_l = Q[:, :, num_k * segs:, :].reshape(bsz, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
    #         Q_landmarks = torch.cat((Q_landmarks_f, Q_landmarks_l), dim = -2)

    #     Q = Q.contiguous().view(-1, tgt_len, self.head_dim)
    #     K = K.contiguous().view(-1, src_len, self.head_dim)
    #     V = V.contiguous().view(-1, src_len, self.head_dim)
    #     Q_landmarks = Q_landmarks.contiguous().view(-1, self.landmarks, self.head_dim)
    #     K_landmarks = K_landmarks.contiguous().view(-1, self.landmarks, self.head_dim)

    #     kernel_1 = torch.nn.functional.softmax(torch.bmm(Q, K_landmarks.transpose(-1, -2)), dim = -1)
    #     kernel_2 = torch.nn.functional.softmax(torch.bmm(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
    #     kernel_3 = torch.nn.functional.softmax(torch.bmm(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - key_padding_mask[:, None, :]), dim = -1)
    #     a = torch.bmm(kernel_1, self.iterative_inv(kernel_2))
    #     b = torch.bmm(kernel_3, V)
    #     X = torch.bmm(a, b)

    #     # if self.use_conv:
    #     #     X += self.conv(V * key_padding_mask[:, None, :, None])
    #     X = X.view(bsz, self.num_heads, tgt_len, self.head_dim)
    #     return X

    # def iterative_inv(self, mat, n_iter = 6):
    #     I = torch.eye(mat.size(-1), device = mat.device)
    #     K = mat
        
    #     if self.init_option == "original":
    #         V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
    #     else:
    #         V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
    #     for _ in range(n_iter):
    #         KV = torch.bmm(K, V)
    #         V = torch.bmm(0.25 * V, 13 * I - torch.bmm(KV, 15 * I - torch.bmm(KV, 7 * I - KV)))
    #     return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}'