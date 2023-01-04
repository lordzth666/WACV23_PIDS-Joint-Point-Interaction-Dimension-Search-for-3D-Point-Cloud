from pids_core.utils.config import Config
class ResNet50SemanticKittiConfigAdv(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SemanticKitti'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 4

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple_k15_i2_o64',
                    'resnetb_k15_i64_o128',
                    'resnetb_strided_k15_i128_o256',
                    'resnetb_k15_i256_o256',
                    'resnetb_k15_i256_o256',
                    'resnetb_strided_k15_i256_o512',
                    'resnetb_k15_i512_o512',
                    'resnetb_k15_i512_o512',
                    'resnetb_strided_k15_i512_o1024',
                    'resnetb_k15_i1024_o1024',
                    'resnetb_k15_i1024_o1024',
                    'resnetb_strided_k15_i1024_o2048',
                    'resnetb_k15_i2048_o2048',
                    'resnetb_k15_i2048_o2048',
                    'nearest_upsample',
                    'unary_k15_i3072_o1024',
                    'nearest_upsample',         #   =1024+512
                    'unary_k15_i1536_o512',     #  =512+256
                    'nearest_upsample',
                    'unary_k15_i768_o256',      # =256+128
                    'nearest_upsample',
                    'unary_k15_i384_o128']       # =128+64

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 4.0
    val_radius = 4.0
    n_frames = 1
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Number of kernel points
    num_kernel_points = -1

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

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
    first_features_dim = 128
    in_features_dim = 2

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.003
    batch_norm_epsilon = 1e-3

    # Weight decay
    weight_decay = 1e-4

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
    max_epoch = 90

    # Learning rate management
    learning_rate = 0.016
    momentum = 0.9
    lr_schedule = 'cosine'
    grad_clip_norm = 10.0

    # Dropout
    dropout = 0

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 200

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    # class proportion for R=10.0 and dl=0.08 (first is unlabeled)
    # 19.1 48.9 0.5  1.1  5.6  3.6  0.7  0.6  0.9 193.2 17.7 127.4 6.7 132.3 68.4 283.8 7.0 78.5 3.3 0.8
    #
    #

    # sqrt(Inverse of proportion * 100)
    # class_w = [1.430, 14.142, 9.535, 4.226, 5.270, 11.952, 12.910, 10.541, 0.719,
    #            2.377, 0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.505, 11.180]

    # sqrt(Inverse of proportion * 100)  capped (0.5 < X < 5)
    # class_w = [1.430, 5.000, 5.000, 4.226, 5.000, 5.000, 5.000, 5.000, 0.719, 2.377,
    #            0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.000, 5.000]

    # Do we nee to save convergence
    saving = True
    saving_path = None


    
class SearchSemanticKittiConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SemanticKitti'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 5

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple_k15_i2_o32',
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

                    'nearest_upsample',
                    'pids_e4_k15_i416_o160',      #   =320+96
                    'nearest_upsample',
                    'pids_e4_k15_i192_o96',     #  =160+32
                    'nearest_upsample',
                    'pids_e4_k15_i120_o64',      # =96+24
                    'nearest_upsample',
                    'pids_e4_k15_i80_o32']       # =64+16

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 4.0
    val_radius = 4.0
    n_frames = 1
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    batch_num = 10
    val_batch_num = 10
    # Number of kernel points
    num_kernel_points = -1

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

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
    first_features_dim = -1
    in_features_dim = 2

    head_conv_dim = 512

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.01
    batch_norm_epsilon = 1e-5

    # Weight decay
    weight_decay = 3e-4

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/r epulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 6

    # Learning rate management
    optimizer = 'adam'
    learning_rate = 0.002

    momentum = 0.9
    lr_schedule = 'cosine-no-warmup'

    grad_clip_norm = 10.0

    # Dropout
    dropout = 0.25
    drop_connect = 0.0

    # Number of steps per epochs
    epoch_steps = 750

    # Number of validation examples per epoch
    validation_size = 100
    
    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    # class proportion for R=10.0 and dl=0.08 (first is unlabeled)
    # 19.1 48.9 0.5  1.1  5.6  3.6  0.7  0.6  0.9 193.2 17.7 127.4 6.7 132.3 68.4 283.8 7.0 78.5 3.3 0.8
    #
    #

    # sqrt(Inverse of proportion * 100)
    # class_w = [1.430, 14.142, 9.535, 4.226, 5.270, 11.952, 12.910, 10.541, 0.719,
    #           2.377, 0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.505, 11.180]

    # sqrt(Inverse of proportion * 100)  capped (0.5 < X < 5)
    #class_w = [1.430, 5.000, 5.000, 4.226, 5.000, 5.000, 5.000, 5.000, 0.719, 2.377,
    #          0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.000, 5.000]

    # Do we need to save convergence
    saving = True
    saving_path = None

    # No validation FLAG.
    no_validation = True
    num_votes = 3

    # Auto Mixed Precision
    amp = False
    # EMA
    ema = 0.9
    # test smooth
    test_smooth = 0.0
    # Attention options.
    use_enc_dec_attn = True

class LATSemanticKittiConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """
    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SemanticKitti'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 1

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple_k15_i2_o32',
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

                    'nearest_upsample',
                    'pids_e4_k15_i416_o160',     #   =320+96
                    'nearest_upsample',
                    'pids_e4_k15_i192_o96',     #  =160+32
                    'nearest_upsample',
                    'pids_e4_k15_i120_o64',      # =96+24
                    'nearest_upsample',
                    'pids_e4_k15_i80_o32']       # =64+16

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 4.0
    val_radius = 4.0
    n_frames = 1
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    batch_num = 8
    val_batch_num = 8

    # Number of kernel points
    num_kernel_points = -1

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

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
    first_features_dim = 128
    in_features_dim = 2

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = False
    batch_norm_momentum = 0.01
    batch_norm_epsilon = 1e-3

    # Weight decay
    weight_decay = 1e-4

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
    max_epoch = 90

    # Learning rate management
    learning_rate = 0.016
    momentum = 0.9
    lr_schedule = 'cosine'

    grad_clip_norm = 10.0

    # Dropout
    dropout = 0.0

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 200

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    # class proportion for R=10.0 and dl=0.08 (first is unlabeled)
    # 19.1 48.9 0.5  1.1  5.6  3.6  0.7  0.6  0.9 193.2 17.7 127.4 6.7 132.3 68.4 283.8 7.0 78.5 3.3 0.8
    #
    #

    # sqrt(Inverse of proportion * 100)
    # class_w = [1.430, 14.142, 9.535, 4.226, 5.270, 11.952, 12.910, 10.541, 0.719,
    #            2.377, 0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.505, 11.180]

    # sqrt(Inverse of proportion * 100)  capped (0.5 < X < 5)
    # class_w = [1.430, 5.000, 5.000, 4.226, 5.000, 5.000, 5.000, 5.000, 0.719, 2.377,
    #            0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.000, 5.000]

    # Do we nee to save convergence
    saving = True
    saving_path = None

class Mv2SemanticKittiConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SemanticKitti'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 8

    #########################
    # Architecture definition
    #########################

    # Define layers 
    architecture = ['simple_k15_i2_o32',
                    'pids_e1_k15_i32_o16',

                    'pids_strided_e4_k15_i16_o24',
                    'pids_e4_k15_i24_o24',

                    'pids_deformable_strided_e4_k15_i24_o32',
                    'pids_deformable_e4_k15_i32_o32',
                    'pids_deformable_e4_k15_i32_o32',

                    'pids_deformable_strided_e4_k15_i32_o64',
                    'pids_deformable_e4_k15_i64_o64',
                    'pids_deformable_e4_k15_i64_o64',
                    'pids_deformable_e4_k15_i64_o64',

                    'pids_deformable_e4_k15_i64_o96',
                    'pids_deformable_e4_k15_i96_o96',
                    'pids_deformable_e4_k15_i96_o96',

                    'pids_deformable_strided_e4_k15_i96_o160',
                    'pids_deformable_e4_k15_i160_o160',
                    'pids_deformable_e4_k15_i160_o160',

                    'pids_deformable_e4_k15_i160_o320',

                    'nearest_upsample',
                    'unary_i416_o160',  # =320+96
                    'nearest_upsample',
                    'unary_i192_o96',  # =160+32
                    'nearest_upsample',
                    'unary_i120_o64',      # =96+24
                    'nearest_upsample',
                    'unary_i80_o32']       # =64+16

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 4.0
    val_radius = 4.0
    n_frames = 1
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Number of kernel points
    num_kernel_points = -1
    stem_conv_dim = 16

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

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
    first_features_dim = 128
    in_features_dim = 2

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.01
    batch_norm_epsilon = 1e-5

    # Weight decay
    weight_decay = 1e-5

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
    max_epoch = 100

    # Learning rate management
    learning_rate = 0.02
    momentum = 0.9
    lr_schedule = 'cosine'
    grad_clip_norm = 10.0

    # Dropout
    dropout = 0.25
    drop_connect = 0.0
    label_smoothing = 0.0

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 100

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Dimension of head conv
    head_conv_dim = 1024

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    # class proportion for R=10.0 and dl=0.08 (first is unlabeled)
    # 19.1 48.9 0.5  1.1  5.6  3.6  0.7  0.6  0.9 193.2 17.7 127.4 6.7 132.3 68.4 283.8 7.0 78.5 3.3 0.8
    #
    #

    # sqrt(Inverse of proportion * 100)
    # class_w = [1.430, 14.142, 9.535, 4.226, 5.270, 11.952, 12.910, 10.541, 0.719,
    #            2.377, 0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.505, 11.180]

    # sqrt(Inverse of proportion * 100)  capped (0.5 < X < 5)
    # class_w = [1.430, 5.000, 5.000, 4.226, 5.000, 5.000, 5.000, 5.000, 0.719, 2.377,
    #            0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.000, 5.000]

    # Do we nee to save convergence
    saving = True
    saving_path = None

    # Num votes
    num_votes = 5
    # Disable amp.
    amp = False
    # Enable EMA.
    ema = 0.9

class Mv2SemanticKittiConfigBlockAttn(Mv2SemanticKittiConfig):
    use_interblock_attn = True

class Mv2SemanticKittiConfigEncDecAttn(Mv2SemanticKittiConfig):
    use_enc_dec_attn = True

class Mv2SemanticKittiConfigAttnAll(Mv2SemanticKittiConfig):
    use_interblock_attn = True
    use_enc_dec_attn = True

class GoldenSemanticKittiConfig(Mv2SemanticKittiConfig):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.003
    batch_norm_epsilon = 1e-3

    # Weight decay
    weight_decay = 3e-4

    # Maximal number of epochs
    max_epoch = 300

    # Learning rate management
    learning_rate = 0.04

    # Dropout
    dropout = 0.5
    drop_connect = 0.0
    label_smoothing = 0.0
    # Gradient Clipping
    grad_clip_norm = 10.0

    # Extra augmentation
    augment_scale_min = 0.8
    augment_scale_max = 1.2

    # Do we nee to save convergence
    saving = True
    saving_path = None

    # Disable amp.
    amp = False
    # Enable EMA.
    ema = 0.997
    test_smooth = .95


class SimpleViewSemanticKittiConfig(Mv2SemanticKittiConfig):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.001
    batch_norm_epsilon = 1e-3

    # Weight decay
    weight_decay = 1e-4

    # Maximal number of epochs
    max_epoch = 300

    # Learning rate management
    learning_rate = 0.048

    # Dropout
    dropout = 0.5
    drop_connect = 0.0
    label_smoothing = 0.0

    # Do we nee to save convergence
    saving = True
    saving_path = None

    # Disable amp.
    amp = False
    # Enable EMA.
    ema = 0.997
    test_smooth = .9
    
    # Augmentation
    augment_rotation = 'none'
    augment_scale_min = 2. / 3.
    augment_scale_max = 3. / 2.
    augment_noise = 0.0
    augment_color = 1.0
    translate_scale = 0.2
