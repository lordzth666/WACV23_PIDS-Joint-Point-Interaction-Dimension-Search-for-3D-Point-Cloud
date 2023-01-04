from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import os
class SecondOrderInt(nn.Module):
    def __init__(self,
                 K_units: int,
                 intra_units: int):
        super().__init__()
        self.K_units = K_units
        self.intra_units = intra_units
        self.attn_network_fc1 = nn.Linear(K_units, intra_units)
        self.attn_network_relu1 = nn.ReLU(True)
        self.attn_network_fc2 = nn.Linear(intra_units, K_units)
        self.attn_network_sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x):
        # [n_pts, n_neighbors, n_kpts]
        x_2D = x.mean(1)
        x_2D_scale = self.attn_network_relu1(self.attn_network_fc1(x_2D))
        x_2D_scale = self.attn_network_sigmoid(self.attn_network_fc2(x_2D_scale))
        out = 0.5 * (torch.multiply(x, x_2D_scale.unsqueeze(1)) + x)
        # Dump High-order Interaction data if needed.
        """
        dump_linearcorr_file_dir = "official/NAS/auto_kpconv/attn_dump_semantickitti/nk_linear_corr"
        if not os.path.exists(dump_linearcorr_file_dir):
            os.makedirs(dump_linearcorr_file_dir)
        dump_attncorr_file_dir = "official/NAS/auto_kpconv/attn_dump_semantickitti/nk_attn_corr"
        if not os.path.exists(dump_attncorr_file_dir):
            os.makedirs(dump_attncorr_file_dir)
        name = "{}pts_{}kpts".format(x.size(0), x.size(-1))
        linear_corr_file_name = os.path.join(dump_linearcorr_file_dir, name)
        attn_corr_file_name = os.path.join(dump_attncorr_file_dir, name)
        npy_linear_corr = x.cpu().detach().numpy()
        npy_attn_corr = out.cpu().detach().numpy()
        np.save(linear_corr_file_name, npy_linear_corr)
        np.save(attn_corr_file_name, npy_attn_corr)
        """
        return out

    def reset_parameters(self):
        stddev_intra = np.sqrt(1.0 / self.intra_units)
        stddev_K = np.sqrt(1.0 / self.K_units)
        self.attn_network_fc1.weight.data.normal_(std=stddev_intra)
        nn.init.zeros_(self.attn_network_fc1.bias.data)
        self.attn_network_fc2.weight.data.normal_(std=stddev_K)
        nn.init.zeros_(self.attn_network_fc2.bias.data)
        
    def __repr__(self):
        repr_str = str(self.attn_network_fc1) + str(self.attn_network_fc2)
        return repr_str

# Useless. Left for experimental features.
"""
class EncoderDecoderScaledProductAttention(nn.Module):
    def __init__(self,
                 enc_units: int,
                 dec_units: int,
                 dk: int):
        super(EncoderDecoderScaledProductAttention, self).__init__()
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.dk = dk
        self.WQ = nn.Linear(self.enc_units, self.dk, bias=True)
        self.WK = nn.Linear(self.dec_units, self.dk, bias=True)
        self.reset_parameters()

    def forward(self, Q, K, V):
        # Q: [n_pts, n_dim]
        # K: [n_pts, n_dim]
        # V: [n_pts, n_dim]
        # [B, 1, N]; [B, 1, N] -> [B, N, N]
        Q_out, K_out = self.WQ(Q.mean(0, keepdim=True).unsqueeze_(1)), \
            self.WK(K.mean(0, keepdim=True).unsqueeze_(1))
        queries = torch.matmul(Q_out.transpose_(-2, -1), K_out) / sqrt(self.dk)
        # [B, N, N] @ [B, N, 1]
        attn = torch.softmax(queries, dim=-1)
        out = torch.matmul(attn, V.unsqueeze(-1)).squeeze_(-1) + V
        return out

    def reset_parameters(self):
        stddev = np.sqrt(1.0 / (self.dk))
        self.WQ.weight.data.normal_(std=stddev)
        nn.init.zeros_(self.WQ.bias.data)
        self.WK.weight.data.normal_(std=stddev)
        nn.init.zeros_(self.WK.bias.data)
"""

# Useless
class EncoderDecoderSqueezeExcitation(nn.Module):
    def __init__(self,
                 units: int,
                 ratio: float = 0.25):
        super(EncoderDecoderSqueezeExcitation, self).__init__()
        self.units = units
        self.ratio = ratio
        self.se_units = int(self.units * self.ratio)
        self.se_squeezed = nn.Linear(self.units, self.se_units, bias=True)
        self.se_squeezed_relu = nn.ReLU(True)
        self.se_excited = nn.Linear(self.se_units, self.units, bias=True)
        self.se_sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x):
        out = self.se_squeezed_relu(self.se_squeezed(x.mean(0, keepdim=True)))
        out = self.se_sigmoid(self.se_excited(out))
        return 0.5 * torch.multiply(x, out) + 0.5 * x

    def reset_parameters(self):
        stddev_units = np.sqrt(1.0 / self.units)
        stddev_se_units = np.sqrt(1.0 / self.se_units)
        self.se_squeezed.weight.data.normal_(std=stddev_se_units)
        nn.init.zeros_(self.se_squeezed.bias.data)
        self.se_excited.weight.data.normal_(std=stddev_units)
        nn.init.zeros_(self.se_excited.bias.data)
