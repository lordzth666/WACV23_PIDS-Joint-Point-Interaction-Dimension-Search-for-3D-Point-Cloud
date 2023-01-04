# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

import math
from pids_core.utils.config import Config

class Mv2Modelnet40Config(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'ModelNet40'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = 'classification'

    # Number of CPU threads for the input pipeline
    input_threads = 5

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple_k15_i1_o32',
                    'pids_e1_k15_i32_o16',

                    'pids_strided_e4_k15_i16_o24',
                    'pids_e4_k15_i24_o24',

                    'pids_strided_e4_k15_i24_o32',
                    'pids_e4_k15_i32_o32',
                    'pids_e4_k15_i32_o32',

                    'pids_strided_e4_k15_i32_o64',
                    'pids_e4_k15_i64_o64',
                    'pids_e4_k15_i64_o64',
                    'pids_e4_k15_i64_o64',

                    'pids_e4_k15_i64_o96',
                    'pids_e4_k15_i96_o96',
                    'pids_e4_k15_i96_o96',

                    'pids_strided_e4_k15_i96_o160',
                    'pids_e4_k15_i160_o160',
                    'pids_e4_k15_i160_o160',
                    'pids_e4_k15_i160_o320',

                    'global_average']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = -1

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    in_features_dim = 1

    # Can the network learn modulations
    modulated = True

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.01
    batch_norm_epsilon = 1e-5

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 0.1              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 100

    # Learning rate management
    learning_rate = 0.02
    momentum = 0.9

    # Dropout
    dropout = 0.25

    grad_clip_norm = 10.0

    # Number of batch
    batch_num = 16
    val_batch_num = 16

    # Weight decay
    weight_decay = 1e-5

    # Number of steps per epochs
    epoch_steps = 620

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 10
    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 1.0
    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None

class Mv2Modelnet40ConfigBlockAttn(Mv2Modelnet40Config):
    use_interblock_attn = True

class GoldenModelnet40Config(Mv2Modelnet40Config):
    """
    Override the parameters you want to modify for this dataset
    """
    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.001
    batch_norm_epsilon = 1e-5
    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 300

    # Learning rate management
    learning_rate = 0.016
    momentum = 0.9

    # Weight decay
    weight_decay = 3e-4
    dropout = .2
    drop_connect = 0.0
    # Validation size.
    validation_size = 100
    # EMA
    ema = 0.99

    # Number of batch
    batch_num = 16
    val_batch_num = 16

class SimpleBaselineModelnet40Config(Mv2Modelnet40Config):
    """
    Override the parameters you want to modify for this dataset
    """
    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.003
    batch_norm_epsilon = 1e-5
    #####################
    # Training parameters
    #####################
    first_subsampling_dl = 0.02
    # Maximal number of epochs
    max_epoch = 300
    # Learning rate management
    learning_rate = 0.016
    momentum = 0.9
    # Weight decay
    weight_decay = 1e-4
    dropout = .5
    drop_connect = 0.2
    # Validation size.
    validation_size = 180
    # EMA
    ema = 0.997
    # Number of batch
    batch_num = 16
    val_batch_num = 16
    # Add label smoothing
    label_smoothing = 0.0
    # Gradient Clipping
    grad_clip_norm = 10.0
    
    # Augmentation
    augment_rotation = 'none'
    augment_scale_min = 2. / 3.
    augment_scale_max = 3. / 2.
    augment_noise = 0.0
    augment_color = 1.0
    
    translate_scale = 0.2

class LATModelnet40Config(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'ModelNet40'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = 'classification'

    # Number of CPU threads for the input pipeline
    input_threads = 1

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple_k15_i1_o32',
                    'pids_e1_k15_i32_o16',

                    'pids_strided_e4_k15_i16_o24',
                    'pids_e4_k15_i24_o24',

                    'pids_strided_e4_k15_i24_o32',
                    'pids_e4_k15_i32_o32',
                    'pids_e4_k15_i32_o32',

                    'pids_strided_e4_k15_i32_o64',
                    'pids_e4_k15_i64_o64',
                    'pids_e4_k15_i64_o64',
                    'pids_e4_k15_i64_o64',

                    'pids_e4_k15_i64_o96',
                    'pids_e4_k15_i96_o96',
                    'pids_e4_k15_i96_o96',

                    'pids_strided_e4_k15_i96_o160',
                    'pids_e4_k15_i160_o160',
                    'pids_e4_k15_i160_o160',
                    'pids_e4_k15_i160_o320',

                    'global_average']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = -1

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    in_features_dim = 1

    # Can the network learn modulations
    modulated = True

    # Batch normalization parameters
    use_batch_norm = False
    batch_norm_momentum = 0.01
    batch_norm_epsilon = 1e-3

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 200

    # Learning rate management
    learning_rate = 0.04
    momentum = 0.9

    # Dropout
    dropout = 0.0

    lr_decays = {}
    for i in range(1, max_epoch):
        lr_decays.update({
            i: (math.cos(math.pi * i / max_epoch) + 1) / (math.cos(math.pi * (i-1) / max_epoch) + 1)})
    # lr_decays = {i: 0.1**(1/100) for i in range(1, max_epoch)}

    grad_clip_norm = 10.0

    # Number of batch
    batch_num = 8
    val_batch_num = 8

    # Weight decay
    weight_decay = 1e-4

    # Number of steps per epochs
    epoch_steps = 620

    # Number of validation examples per epoch
    validation_size = 30

    # Number of epoch between each checkpoint
    checkpoint_gap = 10

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 1.0

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None

class SearchModelnet40Config(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'ModelNet40'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = 'classification'

    # Number of CPU threads for the input pipeline
    input_threads = 5

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple_k15_i1_o16',
                    'pids_e1_k15_i16_o8',

                    'pids_strided_e4_k15_i8_o12',
                    'pids_e4_k15_i12_o12',

                    'pids_strided_e4_k15_i12_o16',
                    'pids_e4_k15_i16_o16',
                    'pids_e4_k15_i16_o16',

                    'pids_strided_e4_k15_i16_o32',
                    'pids_e4_k15_i32_o32',
                    'pids_e4_k15_i32_o32',
                    'pids_e4_k15_i32_o32',

                    'pids_e4_k15_i32_o48',
                    'pids_e4_k15_i48_o48',
                    'pids_e4_k15_i48_o48',

                    'pids_strided_e4_k15_i48_o80',
                    'pids_e4_k15_i80_o80',
                    'pids_e4_k15_i80_o80',

                    'pids_e4_k15_i80_o160',
                    'global_average']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = -1

    head_conv_dim = 512

    # Size of the first subsampling grid in meter.
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    in_features_dim = 1

    # Can the network learn modulations
    modulated = True

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.01
    batch_norm_epsilon = 1e-5

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 0.1              # Multiplier for the fitting/repulsive loss
    # Multiplier for learning rate applied to the deformations
    deform_lr_factor = 0.1
    # Distance of repulsion for deformed kernel points
    repulse_extent = 1.2

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 7

    # Learning rate management
    optimizer = "sgd"
    learning_rate = 0.002
    momentum = 0.9
    lr_schedule = 'cosine-no-warmup'

    # Dropout
    dropout = 0.0

    # lr_decays = {i: 0.1**(1/100) for i in range(1, max_epoch)}

    grad_clip_norm = 10.0

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Weight decay
    weight_decay = 1e-5

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 200

    # Number of epoch between each checkpoint
    checkpoint_gap = 10

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    #augment_scale_min = 0.8
    #augment_scale_max = 1.2
    #augment_noise = 0.001
    augment_color = 1.0

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = "experiments/modelnet40_temp"

    # Num votes
    num_votes = 3
    no_validation = True

    # Auto Mixed Precision
    amp = False
    # EMA
    ema = 0.9
    # test smooth
    test_smooth = 0.0
    # Attention options.
    use_enc_dec_attn = True