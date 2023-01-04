import copy
from typing import (
    List,
    Optional
)

import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

from nasflow.io_utils.base_parser import parse_args_from_kwargs
from nasflow.algo.surrogate.nn_predictor_utils import NNModelBase

class DotProductWithLinear(nn.Module):
    def __init__(
            self,
            dense_tensor_in_dims: int,
            sparse_tensor_num_inputs: int,
            sparse_tensor_in_dims: int,
            out_dim: int):
        super(DotProductWithLinear, self).__init__()
        self.dense_tensor_in_dims = dense_tensor_in_dims
        self.sparse_tensor_num_inputs = sparse_tensor_num_inputs
        self.sparse_tensor_in_dims = sparse_tensor_in_dims
        self.dot_product_out_size = \
            (self.sparse_tensor_num_inputs + 1) * (self.sparse_tensor_num_inputs + 2) // 2 + dense_tensor_in_dims
        self.out_dim = out_dim
        # Add projection layer if needed.
        if self.dense_tensor_in_dims != self.sparse_tensor_in_dims:
            self.dotproduct_proj_layer = nn.Linear(self.dense_tensor_in_dims, self.sparse_tensor_in_dims)
        if self.dot_product_out_size != self.out_dim:
            self.output_mapping_layer = nn.Linear(self.dot_product_out_size, self.out_dim, bias=True)

    def forward(self, dense_tensor, sparse_tensor):
        if self.dense_tensor_in_dims != self.sparse_tensor_in_dims:
            dense_tensor_proj = self.dotproduct_proj_layer(dense_tensor)
        else:
            dense_tensor_proj = dense_tensor
        T = torch.cat([dense_tensor_proj.unsqueeze(1), sparse_tensor], dim=1)
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        _, ni, nj = Z.shape
        li, lj = torch.tril_indices(ni, nj, offset=0)
        Zflat = Z[:, li, lj]
        Zflat = torch.cat([dense_tensor, Zflat], dim=-1)
        if self.dot_product_out_size != self.out_dim:
            Zflat = self.output_mapping_layer(Zflat)
        return Zflat

class SelfAttachLinearLNReLU(nn.Module):
    def __init__(
            self, in_features: int,
            out_features: int,
            use_batchnorm: bool = False,
            dropout: float = 0.0):
        super(SelfAttachLinearLNReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        if self.out_features > self.in_features:
            self.module = nn.Sequential(
                nn.Linear(self.in_features, self.out_features - self.in_features, not self.use_batchnorm),
                nn.BatchNorm1d(self.out_features - self.in_features, momentum=0.01, eps=1e-5) if self.use_batchnorm else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(self.dropout),
            )
        else:
            self.module = nn.Sequential(
                nn.Linear(self.in_features, self.out_features, not self.use_batchnorm),
                nn.BatchNorm1d(self.out_features, momentum=0.01, eps=1e-5) if self.use_batchnorm else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(self.dropout),
            )

    def forward(self, x):
        out = self.module(x)
        if self.out_features > self.in_features:
            return torch.cat([x, out], dim=-1)
        else:
            return out

class LinearLNReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_batchnorm: bool = False):
        super(LinearLNReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_batchnorm = use_batchnorm
        self.module = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, not self.use_batchnorm),
            nn.BatchNorm1d(self.out_features, momentum=0.01, eps=1e-5) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(True),
        )

    def forward(self, x):
        out = self.module(x)
        return out

class ResidualLNReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_batchnorm: bool = False):
        super(ResidualLNReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_batchnorm = use_batchnorm
        self.module = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, not self.use_batchnorm),
            nn.BatchNorm1d(self.out_features, momentum=0.01, eps=1e-5) if self.use_batchnorm else nn.Identity(),
            nn.ReLU(True),
            nn.Linear(self.out_features, self.out_features, not self.use_batchnorm),
            nn.BatchNorm1d(self.out_features, momentum=0.01, eps=1e-5) if self.use_batchnorm else nn.Identity(),
        )
        if self.in_features != self.out_features:
            self.id_mapping = nn.Sequential(
                nn.Linear(self.in_features, self.out_features, not self.use_batchnorm),
                nn.BatchNorm1d(self.out_features, momentum=0.01, eps=1e-5)
            )

    def forward(self, x):
        out = self.module(x)
        x_id = self.id_mapping(x) if self.in_features != self.out_features else x
        return F.relu(out + x_id, True)

class DenseSparseNN(NNModelBase):
    def __init__(
            self,
            dense_nn_pos_units: List[int],
            dense_nn_arch_units: List[int],
            over_nn_units: List[int],
            activation: Optional[str] = "relu",
            num_dense_pos_features: int = 14,
            num_dense_arch_features: int = 14,
            num_sparse_features: int = 7,
            num_pos_sparse_interact_outputs: int = 50,
            num_arch_sparse_interact_outputs: int = 50,
            embedding_table_size: int = 3,
            embedding_dim: int = 8,
            dropout: float = 0.0):
        super(DenseSparseNN, self).__init__()
        self.dense_nn_pos_units = dense_nn_pos_units
        self.dense_nn_arch_units = dense_nn_arch_units
        self.over_nn_units = over_nn_units
        self.activation = activation
        self.num_dense_pos_features = num_dense_pos_features
        self.num_dense_arch_features = num_dense_arch_features
        self.num_sparse_features = num_sparse_features
        self.embedding_table_size = embedding_table_size
        self.embedding_dim = embedding_dim
        self.num_pos_sparse_interact_outputs = num_pos_sparse_interact_outputs
        self.num_arch_sparse_interact_outputs = num_arch_sparse_interact_outputs
        self.dropout = dropout
        # Create 2 dense archs.
        self.dense_pos_nn = self._create_dense_nn(
            self.num_dense_pos_features, self.dense_nn_pos_units)
        self.dense_arch_nn = self._create_dense_nn(
            self.num_dense_arch_features, self.dense_nn_arch_units)
        # Create embedding layers.
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(self.embedding_table_size, self.embedding_dim) \
                for _ in range(self.num_sparse_features)])
        # Fusion Arch: 3 Dot-products are leveraged.
        self.fuse_pos_sparse_nn = DotProductWithLinear(
            dense_nn_pos_units[-1], self.num_sparse_features, self.embedding_dim,
            self.num_pos_sparse_interact_outputs)
        self.fuse_arch_sparse_nn = DotProductWithLinear(
            dense_nn_arch_units[-1], self.num_sparse_features, self.embedding_dim,
            self.num_arch_sparse_interact_outputs)
        # Finally, create over_nn. (Overarch)
        self.overarch_in_dims = self.num_pos_sparse_interact_outputs + self.num_arch_sparse_interact_outputs
        self.overarch_nn = self._create_dense_nn(
            self.overarch_in_dims, self.over_nn_units)
        #self.dense_pos_dropout = nn.Dropout(self.dropout)
        #self.dense_arch_dropout = nn.Dropout(self.dropout)
        self.fused_dropout = nn.Dropout(self.dropout)
        self.model_head = nn.Linear(self.over_nn_units[-1], 1)

        # Define normalizer.
        self.dense_pos_normalizer = nn.BatchNorm1d(self.num_dense_pos_features, momentum=0.01, eps=1e-5,
            affine=False)
        self.dense_arch_normalizer = nn.BatchNorm1d(self.num_dense_arch_features, momentum=0.01, eps=1e-5,
            affine=False)

    def _create_dense_nn(self, in_dims, units):
        all_units = [in_dims] + units
        module_list = []
        for idx in range(len(all_units)-1):
            module_list.append(LinearLNReLU(
                all_units[idx], all_units[idx+1]))
        return nn.Sequential(*module_list)

    def embedding_sparse_inputs(self, sparse_inputs):
        output = torch.stack(
            [self.embedding_layers[i](sparse_inputs[:, i]) \
                for i in range(self.num_sparse_features)], 1)
        return output

    def forward(self, x):
        # First, split 3 inputs respectively.
        dense_pos_inputs = x[:, :self.num_dense_pos_features]
        offset = self.num_dense_pos_features
        dense_arch_inputs = x[:, offset:offset + self.num_dense_arch_features]
        offset += self.num_dense_arch_features
        sparse_inputs = x[:, offset:].long()
        # Normalize dense inputs.
        dense_pos_inputs = self.dense_pos_normalizer(dense_pos_inputs)
        dense_arch_inputs = self.dense_arch_normalizer(dense_arch_inputs)
        dense_pos_feats = self.dense_pos_nn(dense_pos_inputs)
        dense_arch_feats = self.dense_arch_nn(dense_arch_inputs)
        # Dropping all dense features.
        #dense_pos_feats = self.dense_pos_dropout(dense_pos_feats)
        #dense_arch_feats = self.dense_arch_dropout(dense_arch_feats)

        sparse_feats = self.embedding_sparse_inputs(sparse_inputs)
        dense_sparse_fused_features = torch.cat(
            [
                self.fuse_pos_sparse_nn(dense_pos_feats, sparse_feats),
                self.fuse_arch_sparse_nn(dense_arch_feats, sparse_feats),
            ],
            dim=-1
        )
        dense_sparse_fused_features = self.fused_dropout(dense_sparse_fused_features)
        return self.model_head(self.overarch_nn(dense_sparse_fused_features))
