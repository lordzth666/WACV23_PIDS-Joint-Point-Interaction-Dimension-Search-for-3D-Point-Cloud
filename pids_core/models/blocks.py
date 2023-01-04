import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from pids_core.kernels.kernel_points import load_kernels
from pids_core.models.utils import drop_connect_impl
from pids_core.models.high_order_interaction import SecondOrderInt

def gather(x, idx):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    for i, ni in enumerate(idx.size()[1:]):
        x = x.unsqueeze(i+1)
        new_s = list(x.size())
        new_s[i+1] = ni
        x = x.expand(new_s)
    n = len(idx.size())
    for i, di in enumerate(x.size()[n:]):
        idx = idx.unsqueeze(i+n)
        new_s = list(idx.size())
        new_s[i+n] = di
        idx = idx.expand(new_s)
    return x.gather(0, idx)

def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))

def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x_padded = torch.zeros_like(x[:1, :])
    x = torch.cat((x, x_padded), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[:, 0])

def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x_padded = torch.zeros_like(x[:1, :])
    x = torch.cat((x, x_padded), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features

def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """
    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for _, length in enumerate(batch_lengths):

        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)


def swish(x):
    return x * F.sigmoid(x)

# Vanilla version of the KPConv class
class KPConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 p_dim: int,
                 in_channels: int,
                 out_channels: int,
                 KP_extent: float,
                 radius: float,
                 fixed_kernel_points: str = 'center',
                 KP_influence: str = 'linear',
                 aggregation_mode: str = 'sum',
                 deformable: bool = False,
                 modulated: bool = False,
                 use_attn: bool = False,
                 attn_dk: int = 100,
                 use_relu: bool = False,
                 use_bn: bool = False,
                 bn_momentum: float = 0.01,
                 bn_epsilon: float = 1e-5,
                ):
        """
        Initialize parameters for KPConv.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        :param bias: whether include bias in this layer or not.
        """
        super(KPConv, self).__init__()
        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.use_attn = use_attn
        self.attn_dk = attn_dk
        if self.use_attn:
            self.attn_block = KernelPointAttention(
                self.K, intra_units=self.K)
        self.use_bn = use_bn
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels,
                                             momentum=bn_momentum,
                                             eps=bn_epsilon)
        self.bias = Parameter(torch.zeros(self.out_channels, dtype=torch.float32),
                              requires_grad=True) if not self.use_bn else None
        # Running variable containing deformed KP distance to input points. (used in `regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None
        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)
        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        self.use_bn = use_bn
        self.use_relu = use_relu
        # Reset parameters
        self.reset_parameters()
        # Initialize kernel points
        self.kernel_points = self.init_KP()

        self.flops = 0
        # Add optional bias.
        self.bias = Parameter(torch.zeros(self.out_channels, dtype=torch.float32),
                              requires_grad=True) if not use_bn else None

    def reset_parameters(self):
        stddev = np.sqrt(1.0 / (self.K * self.out_channels))
        self.weights.data.normal_(std=stddev)
        if self.deformable:
            nn.init.zeros_(self.offset_bias)

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """

        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)
        return Parameter(torch.from_numpy(K_points_numpy), requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        # Add a fake point in the last row for shadow neighbors
        s_pts_padded = torch.zeros_like(s_pts[:1, :]) + 1e6
        s_pts = torch.cat((s_pts, s_pts_padded), 0)
        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        new_neighb_inds = neighb_inds

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        # Do linear.
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points
        sq_distances = torch.sum(differences * differences, dim=-1)
        # [n_pts, n_neighbors, n_kpts]
        all_weights = F.relu(1 - torch.sqrt(sq_distances) / self.KP_extent, True)
        if self.use_attn:
            all_weights = self.attn_block(all_weights)
        all_weights = all_weights.transpose(1, 2)
        weighted_features = torch.matmul(all_weights, neighb_x)
        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)
        # Convolution sum [n_points, out_fdim]
        result = torch.sum(kernel_outputs, dim=0)
        # Process BN and/or bias add.
        result = self.batch_norm(result) if self.use_bn else result + self.bias
        # Process ReLU.
        result = F.relu(result, True) if self.use_relu else result
        return result

    def __repr__(self):
        return 'KPConv(kernel_size: {}, radius: {:.2f}, in_feat: {:d}, out_feat: {:d}, bias: {}, BN: {}, ReLU: {}, Attention: {})'.format(
            self.K, self.radius, self.in_channels, self.out_channels,
            self.bias is not None, self.use_bn, self.use_relu, self.use_attn)


class PointInt(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 p_dim: int,
                 in_channels: int,
                 out_channels: int,
                 KP_extent: float,
                 radius: float,
                 fixed_kernel_points: str = 'center',
                 KP_influence: str = 'linear',
                 aggregation_mode: str = 'sum',
                 deformable: bool = False,
                 modulated: bool = False,
                 use_attn: bool = False,
                 attn_dk: int = 100,
                 use_relu: bool = False,
                 use_bn: bool = False,
                 bn_momentum: float = 0.01,
                 bn_epsilon: float = 1e-5,):
        super().__init__()
        assert out_channels == in_channels, \
            ValueError("Output channel should be the same as \
                       input channel for PointInt.")
        assert not deformable, ValueError("Deformable option not supported!")
        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None
        self.dw_weights = Parameter(torch.zeros((self.K, in_channels), dtype=torch.float32),
                                 requires_grad=True)

        self.offset_conv_dim = None

        self.use_attn = use_attn
        self.attn_dk = attn_dk
        # Define the neighbor-kernel attention block.
        if self.use_attn:
            self.attn_block = SecondOrderInt(self.K, self.K)

        self.use_bn = use_bn
        self.use_relu = use_relu

        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels,
                                            momentum=bn_momentum,
                                            eps=bn_epsilon)

        self.bias = Parameter(torch.zeros(self.out_channels, dtype=torch.float32),
                              requires_grad=True) if not self.use_bn else None
        # Reset parameters
        self.reset_parameters()
        # Initialize kernel points
        self.kernel_points = self.init_KP()
        # print(self.kernel_points)
        self.flops = 0

    def reset_parameters(self):
        stddev = np.sqrt(1.0 / self.K)
        self.dw_weights.data.normal_(std=stddev)
        # kaiming_uniform_(self.dw_weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)

    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        K_points_numpy = load_kernels(self.radius,
                                      self.K,
                                      dimension=self.p_dim,
                                      fixed=self.fixed_kernel_points)
        return Parameter(torch.from_numpy(K_points_numpy),
                         requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        ######################
        # Deformed convolution
        ######################
        # Add a fake point in the last row for shadow neighbors
        s_pts_padding = torch.zeros_like(s_pts[:1, :]) + 1e6
        s_pts = torch.cat((s_pts, s_pts_padding), 0)
        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)
        deformed_K_points = self.kernel_points
        new_neighb_inds = neighb_inds
        # Add a zero feature for shadow neighbors
        x_padding = torch.zeros_like(x[:1, :])
        x = torch.cat((x, x_padding), 0)
        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)
        # Apply distance weights [n_points, n_kpoints, in_fdim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points
        sq_distances = torch.sum(differences ** 2, dim=-1)
        # Remove F.relu
        all_weights = F.relu(1 - torch.sqrt(sq_distances) / self.KP_extent, True)
        # Neighbor-Kernel Attention starts here.
        if self.use_attn:
            all_weights = self.attn_block(all_weights)
        all_weights = all_weights.transpose(1, 2)
        # [n_pts, n_kpts, n_neighbors]
        weighted_features = torch.matmul(all_weights, neighb_x)
        # Depthwise kernel is applied here.
        kernel_outputs = torch.multiply(weighted_features, self.dw_weights)
        # Convolution sum [n_points, out_fdim]
        result = torch.sum(kernel_outputs, dim=1)
        # Process BN and/or bias add.
        result = self.batch_norm(result) if self.use_bn else result + self.bias
        # Process ReLU.
        result = F.relu(result, True) if self.use_relu else result
        return result

    def __repr__(self):
        repr_str = 'PointInt(kernel_size: {}, radius: {:.2f}, in_feat: {:d}, out_feat: {:d}, attention: {}, use_bn: {}, use_relu: {})\n'\
        .format(
            self.K, self.radius, self.in_channels, self.out_channels, self.use_attn, self.use_bn, self.use_relu)
        return repr_str
    
    
class UnaryBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 use_bn,
                 bn_momentum,
                 bn_epslion,
                 no_relu=False,
                 use_dense_init=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super().__init__()
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epslion
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=not use_bn)
        self.use_dense_init = use_dense_init

        self.batch_norm = None if not use_bn \
            else torch.nn.BatchNorm1d(out_dim,
                                      eps=self.bn_epsilon,
                                      momentum=bn_momentum)
        #if not no_relu:
        #    self.leaky_relu = nn.LeakyReLU(0.1)
        self.flops = 0
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_dense_init:
            print("Dense intialization!")
            torch.nn.init.normal_(self.mlp.weight, 0.0, 0.01)
        else:
            stddev = np.sqrt(1. / self.out_dim)
            self.mlp.weight.data.normal_(0.0, std=stddev)
        if not self.use_bn:
            torch.nn.init.zeros_(self.mlp.bias)

    def forward(self, x, batch=None):
        x = self.mlp(x)
        if self.use_bn:
            x = self.batch_norm(x)
        if not self.no_relu:
            x = F.relu(x, True)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'\
            .format(self.in_dim,
                    self.out_dim,
                    str(self.use_bn),
                    str(not self.no_relu))

class SimpleBlock(nn.Module):
    def __init__(self,
                 block_name,
                 in_dim,
                 out_dim,
                 radius,
                 layer_ind,
                 config,
                 k=15):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.bn_epsilon = config.batch_norm_epsilon
        self.use_bn = config.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(k,
                             config.in_points_dim,
                             in_dim,
                             out_dim,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated,
                             use_bn=self.use_bn,
                             bn_momentum=self.bn_momentum,
                             bn_epsilon=self.bn_epsilon,
                             use_relu=True)
        self.flops = 0

    def forward(self, x, batch):
        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return x

class ResnetBottleneckBlock(nn.Module):
    def __init__(self,
                 block_name,
                 in_dim,
                 out_dim,
                 radius,
                 layer_ind,
                 config,
                 k=15):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.bn_epsilon = config.batch_norm_epsilon
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum, self.bn_epsilon)
        else:
            self.unary1 = nn.Identity()

        # KPConv block
        self.KPConv = KPConv(k,
                             config.in_points_dim,
                             out_dim // 4,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=config.modulated,
                             use_bn=self.use_bn,
                             bn_momentum=self.bn_momentum,
                             bn_epsilon=self.bn_epsilon,
                             use_relu=True,)
        self.batch_norm_conv = nn.BatchNorm1d(
            out_dim // 4,
            momentum=self.bn_momentum,
            eps=self.bn_epsilon)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4,
                                 out_dim,
                                 self.use_bn,
                                 self.bn_momentum,
                                 self.bn_epsilon,
                                 no_relu=True)
        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim,
                                             out_dim,
                                             self.use_bn,
                                             self.bn_momentum,
                                             self.bn_epsilon,
                                             no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        # self.leaky_relu = nn.LeakyReLU(0.1)
        self.flops = 0

    def forward(self, features, batch):
        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return F.relu(x + shortcut, True)

class PointOperator(nn.Module):
    def __init__(self,
                 block_name,
                 in_dim,
                 out_dim,
                 radius,
                 layer_ind,
                 config,
                 k: int = 15,
                 expand: int = 6,
                 use_attn: bool = False):
        """
        Initialize a Point Operator block that allows the search of point dimensions.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super().__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.bn_momentum = config.batch_norm_momentum
        self.bn_epsilon = config.batch_norm_epsilon
        self.use_bn = config.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand = expand
        self.use_attn = use_attn
        # First, apply expansion MLP
        expanded_dims = int(in_dim * expand)
        if self.expand != 1:
            self.unary1 = UnaryBlock(in_dim,
                                     expanded_dims,
                                     self.use_bn,
                                     self.bn_momentum,
                                     self.bn_epsilon,
                                     no_relu=False)
        else:
            self.unary1 = nn.Identity()

        # 1st-order point operator.
        self.point_Int = PointInt(k,
                               config.in_points_dim,
                               expanded_dims,
                               expanded_dims,
                               current_extent,
                               radius,
                               fixed_kernel_points=config.fixed_kernel_points,
                               KP_influence=config.KP_influence,
                               aggregation_mode=config.aggregation_mode,
                               deformable='deform' in block_name,
                               modulated=config.modulated,
                               use_bn=self.use_bn,
                               bn_momentum=self.bn_momentum,
                               bn_epsilon=self.bn_epsilon,
                               use_relu=True,
                               use_attn=self.use_attn
                        )
    
        # Second, projection.
        self.unary2 = UnaryBlock(expanded_dims,
                                 out_dim,
                                 self.use_bn,
                                 self.bn_momentum,
                                 self.bn_epsilon,
                                 no_relu=True)
        # Other operations
        # self.leaky_relu = nn.LeakyReLU(0.1)
        self.flops = 0
        self.max_drop_connect = config.drop_connect
        self.cur_epochs = 0
        self.max_epoch = config.max_epoch
        self.drop_connect = self.max_drop_connect * self.cur_epochs / self.max_epoch
        return

    def step_drop_connect(self, verbose=False):
        self.cur_epochs += 1
        self.drop_connect = min(
            self.max_drop_connect, 
            self.max_drop_connect * self.cur_epochs / self.max_epoch
        )
        if verbose:
            print("Drop connect updated to {}!".format(self.drop_connect))

    def forward(self, features, batch):
        if 'strided' in self.block_name:
            q_pts = batch.points[self.layer_ind + 1]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.pools[self.layer_ind]
        else:
            q_pts = batch.points[self.layer_ind]
            s_pts = batch.points[self.layer_ind]
            neighb_inds = batch.neighbors[self.layer_ind]
        # First downscaling mlp
        x = self.unary1(features)
        # Convolution
        x = self.point_Int(q_pts, s_pts, neighb_inds, x)
        # Second upscaling mlp
        x = self.unary2(x)
        if 'strided' in self.block_name or self.in_dim != self.out_dim:
            return x
        # Apply a drop_path in-between.
        return features + drop_connect_impl(x, self.drop_connect, self.training)

class GlobalAverageBlock(nn.Module):
    def __init__(self):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        """
        super(GlobalAverageBlock, self).__init__()

    def forward(self, x, batch):
        return global_average(x, batch.lengths[-1])

class NearestUpsampleBlock(nn.Module):
    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind,
                                                                  self.layer_ind - 1)

class MaxPoolBlock(nn.Module):
    def __init__(self, layer_ind):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch.pools[self.layer_ind + 1])
