import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch

from pids_core.models.blocks_choice import block_decider
from pids_core.models.blocks import (
    KPConv,
    PointInt,
    UnaryBlock,
    )
from pids_core.models.losses import LabelSmoothingCrossEntropy
from pids_core.models.high_order_interaction import EncoderDecoderSqueezeExcitation
from pids_core.models.utils import (
    get_simple_block_params,
    get_PIDS_params,
    get_resnet_block_params)


def p2p_fitting_regularizer(net):
    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, (KPConv, PointInt)) and m.deformable:
            ##############
            # Fitting loss
            ##############
            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)
            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))
            ################
            # Repulsive loss
            ################
            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent
            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)

class PIDS_Classification(nn.Module):
    """
    Class defining PIDS_Classification
    This is inherited from the KPCNN architecture.
    """
    def __init__(self, config):
        super().__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0

        out_dim = -1
        for _, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (out_dim % 3 != 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            if block.startswith("pids"):
                params = get_PIDS_params(block)
                k, e, i, o, a = params['k'], params['e'], params['i'], params['o'], params['a']
                out_dim = o
                # Apply the good block function defining tf ops
                self.block_ops.append(block_decider(block,
                                                    r,
                                                    i,
                                                    o,
                                                    layer,
                                                    config,
                                                    k=k,
                                                    e=e,
                                                    use_attn=(a == 1)))
            elif block.startswith("simple"):
                # Apply the good block function defining tf ops
                params = get_simple_block_params(block)
                k, i, o = params['k'], params['i'], params['o']
                out_dim = o
                self.block_ops.append(block_decider(block,
                                                    r,
                                                    i,
                                                    o,
                                                    layer,
                                                    config,
                                                    k=k))
            elif block.startswith("resnetb"):
                # Apply the good block function defining tf ops
                params = get_resnet_block_params(block)
                k, i, o = params['k'], params['i'], params['o']
                out_dim = o
                self.block_ops.append(block_decider(block,
                                                    r,
                                                    i,
                                                    o,
                                                    layer,
                                                    config,
                                                    k=k))
            elif block.startswith("global_average"):
                self.block_ops.append(block_decider(block,
                                                    r,
                                                    -1,
                                                    -1,
                                                    layer,
                                                    config, )
                )
            elif block.startswith('unary'):
                params = get_resnet_block_params(block)
                i, o = params['i'], params['o']
                out_dim = o
                self.block_ops.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=-1))
            else:
                raise NotImplementedError(block)

            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim,
                                   config.head_conv_dim,
                                   use_bn=False,
                                   bn_momentum=config.batch_norm_momentum,
                                   bn_epslion=config.batch_norm_epsilon,
                                   no_relu=False)
        self.head_softmax = UnaryBlock(config.head_conv_dim, config.num_classes, False, 0, 0, no_relu=True,
                                       use_dense_init=True)

        ################
        # Network Losses
        ################

        if config.label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss()
        else:
            print("Use label smoothing loss with smooth_factor={}!".format(config.label_smoothing))
            self.criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
            
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):
        # Save all block operations in a list of modules
        x = batch.features.clone().detach()
        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)
        # Head of network
        x = self.head_mlp(x, batch)
        if config.dropout != .0:
            x = F.dropout(x, p=config.dropout)
        x = self.head_softmax(x, batch)
        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total

class PIDS_Segmentation(nn.Module):
    """
    Class defining PIDS_Segmentation
    This is inherited from the "KPFCNN" architecture in KPConv paper.
    """

    def __init__(self,
                 config,
                 lbl_values,
                 ign_lbls,
                 enable_encoder_decoder_attention=False):
        super().__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius

        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        self.enable_encoder_decoder_attention = enable_encoder_decoder_attention

        if self.enable_encoder_decoder_attention:
            self.enc_dec_attention_blocks = nn.ModuleList([])

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []
        self.encoder_output_channels = []
        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):
            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            if block.startswith("pids"):
                params = get_PIDS_params(block)
                k, e, i, o, a = params['k'], params['e'], params['i'], params['o'], params['a']
                out_dim = o
                # Apply the good block function defining tf ops
                self.encoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=k,
                                                         e=e,
                                                         use_attn=(a == 1)))
            elif block.startswith("simple"):
                # Apply the good block function defining tf ops
                params = get_simple_block_params(block)
                k, i, o = params['k'], params['i'], params['o']
                out_dim = o
                self.encoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=k))
            elif block.startswith("resnetb"):
                # Apply the good block function defining tf ops
                params = get_resnet_block_params(block)
                k, i, o = params['k'], params['i'], params['o']
                out_dim = o
                self.encoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=k))
            elif block.startswith('unary'):
                params = get_resnet_block_params(block)
                i, o = params['i'], params['o']
                out_dim = o
                self.encoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=-1))
            elif block.startswith("global_average") or block.endswith("upsample"):
                self.encoder_blocks.append(block_decider(block,
                                                         r,
                                                         -1,
                                                         -1,
                                                         layer,
                                                         config, ))
            else:
                raise NotImplementedError("Block {} not found!".format(block))

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                # The input of the current layer serves as the output of last layer.
                self.encoder_output_channels.append(i)

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                
        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):
            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                self.decoder_concats.append(block_i)
                if self.enable_encoder_decoder_attention:
                    num_elements = int(o) + int(self.encoder_output_channels[layer])
                    self.enc_dec_attention_blocks.append(
                        EncoderDecoderSqueezeExcitation(num_elements)
                        )
            if block.startswith("pids"):
                params = get_PIDS_params(block)
                k, e, i, o, a = params['k'], params['e'], params['i'], params['o'], params['a']
                # Apply the good block function defining tf ops
                self.decoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=k,
                                                         e=e,
                                                         use_attn=(a == 1)))
            elif block.startswith("simple"):
                # Apply the good block function defining tf ops
                params = get_simple_block_params(block)
                k, i, o = params['k'], params['i'], params['o']
                self.decoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=k))
            elif block.startswith("resnetb"):
                # Apply the good block function defining tf ops
                params = get_resnet_block_params(block)
                k, i, o = params['k'], params['i'], params['o']
                self.decoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=k))
            elif block.startswith('unary'):
                params = get_resnet_block_params(block)
                i, o = params['i'], params['o']
                out_dim = o
                self.decoder_blocks.append(block_decider(block,
                                                         r,
                                                         i,
                                                         o,
                                                         layer,
                                                         config,
                                                         k=-1))

            elif block.startswith("global_average") or "upsample" in block:
                self.decoder_blocks.append(block_decider(block,
                                                         r,
                                                         -1,
                                                         -1,
                                                         layer,
                                                         config, ))
            else:
                raise NotImplementedError("Module %s not found!" % block)

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5

        self.head_mlp = UnaryBlock(o,
                                   config.head_conv_dim,
                                   use_bn=False,
                                   bn_momentum=config.batch_norm_momentum,
                                   bn_epslion=config.batch_norm_epsilon,
                                   no_relu=False)
        self.head_softmax = UnaryBlock(config.head_conv_dim, self.C, False, 0, 0, no_relu=True,
                                       use_dense_init=True)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss

        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

    def forward(self, batch, config):
        # Get input features
        x = batch.features.clone().detach()
        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
            # print("Success!")

        decoder_attn_layer_idx = 0
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                pop_op = skip_x.pop()
                if self.enable_encoder_decoder_attention:       # NOTE: This option is not used currently.
                    x = torch.cat([x, pop_op], dim=1)
                    x = self.enc_dec_attention_blocks[decoder_attn_layer_idx](x)
                    decoder_attn_layer_idx += 1
                else:
                    x = torch.cat([x, pop_op], dim=1)
            x = block_op(x, batch)

        x = self.head_mlp(x, batch)
        if config.dropout != .0:
            x = F.dropout(x, p=config.dropout)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
