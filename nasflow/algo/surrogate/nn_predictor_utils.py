import copy
from typing import (
    List,
    Optional,
)

import numpy as np
import torch
import torch.nn as nn

from nasflow.losses.loss_factory import get_loss_fn_from_lib
from nasflow.optim.ema import EMA
from math import sqrt

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight.data)
    else:
        pass
class NNModelBase(nn.Module):
    def __init__(self):
        super(NNModelBase, self).__init__()

    def load_weights(
            self,
            ckpt_path: Optional[str] = None, 
            exclude_ckpt_keys: Optional[List[str]] = None):
        if ckpt_path is None:
            print("Training from scratch!")
            return
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if exclude_ckpt_keys is not None:
            new_state_dict = {}
            for key in state_dict.keys():
                if key not in exclude_ckpt_keys:
                    new_state_dict.update({key: state_dict[key]})
        else:
             new_state_dict = state_dict
        self.load_state_dict(new_state_dict, strict=False)
        print("Loaded checkpoint from {}!".format(ckpt_path))

    def save_weights(
            self,
            ckpt_path: Optional[str] = None
    ):
        if ckpt_path is None:
            print("Warning: ckpt_path is 'None'. Not saving model...")
            return
        torch.save(self.state_dict(), ckpt_path)
        print("Saved checkpoint to {}!".format(ckpt_path))

    def forward(self, x):
        raise BaseException("This is a base class used for NN utilities.")


class EMANNModelBase(EMA):
    def __init__(self, nnmodel: NNModelBase, ema_decay: float):
        super(EMANNModelBase, self).__init__(nnmodel, ema_decay)

    def load_weights(
            self,
            ckpt_path: Optional[str] = None, 
            exclude_ckpt_keys: Optional[List[str]] = None):
        if ckpt_path is not None:
            self.shadow.load_weights(ckpt_path, exclude_ckpt_keys)
            self.model.load_weights(ckpt_path, exclude_ckpt_keys)

    def save_weights(
            self,
            ckpt_path: Optional[str] = None
    ):
        self.shadow.save_weights(ckpt_path)

    def loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        if self.training:
            return self.model.loss(outputs, labels)
        else:
            return self.shadow.loss(outputs, labels)

class NNBackBone(NNModelBase):
    def __init__(
            self,
            in_dims: int,
            units: List[int],
            activation: Optional[str] = "relu"):
        super(NNBackBone, self).__init__()
        self.in_dims = in_dims
        self.units = units
        self.activation = activation
        all_units = [self.in_dims] + self.units
        module_list = []
        for idx in range(len(all_units)-1):
            module_list.append(nn.Linear(all_units[idx], all_units[idx+1]))
            module_list.append(nn.LayerNorm(all_units[idx+1]))
            module_list.append(nn.ReLU(True))
        self.model_arch = nn.Sequential(*module_list)
        self.model_head = nn.Linear(all_units[-1], 1)
        self.loss_fn = get_loss_fn_from_lib('mse-loss')()

    def forward(self, x):
        return self.model_head(self.model_arch(x))

    def loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        if self.training:
            self.steps += 1
        return self.loss_fn(outputs.flatten(), labels)

class NNBackBoneEmbedding(NNModelBase):
    def __init__(
            self,
            in_dims: int,
            units: List[int],
            activation: Optional[str] = "relu",
            embedding_num_inputs: int = 5,
            embedding_table_size: int = 5,
            embedding_dim: int = 8,):
        """
        Initialize a NN predictor with embedding layers.
        Inputs are organized in the format of [B, n_inputs, num_features].
        """
        super(NNBackBoneEmbedding, self).__init__()
        self.in_dims = in_dims
        self.units = units
        self.activation = activation
        self.embedding_num_inputs = embedding_num_inputs
        self.embedding_table_size = embedding_table_size
        self.embedding_dim = embedding_dim
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(self.embedding_table_size, self.embedding_dim) \
                for _ in range(self.embedding_num_inputs)])
        all_units = [self.in_dims] + self.units
        module_list = []
        for idx in range(len(all_units)-1):
            module_list.append(nn.Linear(all_units[idx], all_units[idx+1], bias=True))
            # module_list.append(nn.LayerNorm(all_units[idx+1]))
            module_list.append(nn.LeakyReLU(0.1, True))
        self.model_arch = nn.Sequential(*module_list)
        self.model_head = nn.Linear(all_units[-1], 1)
        
    def embedding_sparse_inputs(self, sparse_inputs):
        output = torch.cat(
            [self.embedding_layers[i](sparse_inputs[:, i]) \
                for i in range(self.embedding_num_inputs)], 1)
        output = output.view(output.size(0), -1)
        return output

    def forward(self, x):
        embedded_inputs = self.embedding_sparse_inputs(x.long())
        out = self.model_head(self.model_arch(embedded_inputs))
        return out


class NNBackBoneDense(NNModelBase):
    def __init__(
            self,
            in_dims: int,
            units: List[int],
            activation: Optional[str] = "relu",
            dropout: float = 0.0,
            first_bn: bool = False,
            use_sigmoid: bool = False,
            normalize_adj_mat: bool = True):
        """
        Initialize a NN predictor with embedding layers.
        Inputs are organized in the format of [B, n_inputs, num_features].
        """
        super(NNBackBoneDense, self).__init__()
        self.in_dims = in_dims
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.first_bn = first_bn
        self.use_sigmoid = use_sigmoid
        self.normalize_adj_mat = normalize_adj_mat
        if self.normalize_adj_mat:
            self.normalize_fn = lambda x: x * 2 - 1
        elif self.first_bn:
            # Insert a BatchNorm layer to normalize inputs.
            self.batchnorm_input = nn.BatchNorm1d(
                self.in_dims, eps=1e-5, momentum=0.01, affine=False)
        all_units = [self.in_dims] + self.units
        module_list = []
        for idx in range(len(all_units)-1):
            module_list.append(nn.Linear(all_units[idx], all_units[idx+1], bias=False))
            module_list.append(nn.BatchNorm1d(all_units[idx+1], eps=1e-3, momentum=0.1))
            module_list.append(nn.ReLU(True))
        self.model_arch = nn.Sequential(*module_list)
        self.model_head = nn.Linear(all_units[-1], 1)
        if self.use_sigmoid:
            self.sigmoid_out = nn.Sigmoid()

    def forward(self, x):
        if self.normalize_adj_mat:
            out = self.normalize_fn(x)
        elif self.first_bn:
            out = self.batchnorm_input(x)
        else:
            out = x
        out = self.model_arch(out)
        if self.dropout != 0:
            out = torch.nn.functional.dropout(out, self.dropout)
        out = self.model_head(out)
        if self.use_sigmoid:
            out = self.sigmoid_out(out)
        return out
