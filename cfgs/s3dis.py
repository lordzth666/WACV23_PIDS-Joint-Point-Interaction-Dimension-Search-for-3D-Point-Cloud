from pids_core.utils.config import Config

class Mv2S3DISConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'S3DIS'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 2

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple_k15_i5_o32',
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
                    'unary_i416_o160',      #   =320+96
                    'nearest_upsample',
                    'unary_i192_o96',     #  =160+32
                    'nearest_upsample',
                    'unary_i120_o64',      # =96+24
                    'nearest_upsample',
                    'unary_i80_o32']       # =64+16


    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 1.5

    # Number of kernel points
    num_kernel_points = -1
    stem_conv_dim = 16

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.04

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
    in_features_dim = 5

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
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
    learning_rate = 0.02
    momentum = 0.9
    lr_schedule = 'cosine'

    grad_clip_norm = 10.0

    # Number of batch
    batch_num = 8
    val_batch_num = 8

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 100

    # Number of epoch between each checkpoint
    checkpoint_gap = 45

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None
    # Gradient Clipping
    grad_clip_norm = 10.0

    weight_decay = 1e-5
    dropout = 0

    # Validation split
    s3dis_validation_split = 4

class Mv2S3DISConfigBlockAttn(Mv2S3DISConfig):
    use_interblock_attn = True

class Mv2S3DISConfigEncDecAttn(Mv2S3DISConfig):
    use_enc_dec_attn = True

class Mv2S3DISConfigAttnAll(Mv2S3DISConfig):
    use_interblock_attn = True
    use_enc_dec_attn = True

class GoldenS3DISConfig(Mv2S3DISConfig):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.003
    batch_norm_epsilon = 1e-3

    # Maximal number of epochs
    max_epoch = 300

    # Learning rate management
    learning_rate = 0.04

    # Do we nee to save convergence
    saving = True
    saving_path = None

    weight_decay = 1e-4
    dropout = 0.5
    drop_connect = 0.0
    label_smoothing = 0.0
    # Gradient Clipping
    grad_clip_norm = 10.0

    # Validation split
    s3dis_validation_split = 4
    # Enable EMA
    ema = 0.997
    test_smooth = .95
    
    # Increase validation size.
    validation_size = 250

class SimpleViewS3DISConfig(Mv2S3DISConfig):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.001
    batch_norm_epsilon = 1e-3

    # Maximal number of epochs
    max_epoch = 300

    # Learning rate management
    learning_rate = 0.04

    # Do we nee to save convergence
    saving = True
    saving_path = None

    weight_decay = 1e-4
    dropout = 0.5
    drop_connect = 0.0
    label_smoothing = 0.0

    # Validation split
    s3dis_validation_split = 4
    # Enable EMA
    ema = 0.997
    test_smooth = .95
    
    # Increase validation size.
    validation_size = 500

    # Augmentation
    augment_rotation = 'none'
    augment_scale_min = 2. / 3.
    augment_scale_max = 3. / 2.
    augment_noise = 0.0
    augment_color = 1.0
    translate_scale = 0.2



class ResNetS3DISConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'S3DIS'

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
    architecture = ['simple_k15_i5_o32',
                    'resnetb_k15_i32_o64',
                    'resnetb_strided_k15_i64_o128',
                    'resnetb_k15_i128_o128',
                    'resnetb_k15_i128_o128',
                    'resnetb_strided_k15_i128_o256',
                    'resnetb_deformable_k15_i256_o256',
                    'resnetb_deformable_k15_i256_o256',
                    'resnetb_deformable_strided_k15_i256_o512',
                    'resnetb_deformable_k15_i512_o512',
                    'resnetb_deformable_k15_i512_o512',
                    'resnetb_deformable_strided_k15_i512_o1024',
                    'resnetb_deformable_k15_i1024_o1024',
                    'nearest_upsample',
                    'unary_k15_i1536_o512',
                    'nearest_upsample',         #   =1024+512
                    'unary_k15_i768_o256',     #  =512+256
                    'nearest_upsample',
                    'unary_k15_i384_o128',      # =256+128
                    'nearest_upsample',
                    'unary_k15_i192_o64']       # =128+64

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 1.5

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.04

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
    in_features_dim = 5

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
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
    max_epoch = 90

    # Learning rate management
    learning_rate = 0.016
    momentum = 0.9
    lr_schedule = 'cosine'

    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 6

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 50

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

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None

    weight_decay = 5e-4
    dropout = 0

    # Validation split
    s3dis_validation_split = 4

class SearchS3DISConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'S3DIS'

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
    architecture = ['simple_k15_i5_o32',
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
                    'unary_i416_o160',      #   =320+96
                    'nearest_upsample',
                    'unary_i192_o96',     #  =160+32
                    'nearest_upsample',
                    'unary_i120_o64',      # =96+24
                    'nearest_upsample',
                    'unary_i80_o32']       # =64+16


    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 1.5

    # Number of kernel points
    num_kernel_points = -1

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.04

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
    in_features_dim = 5

    head_conv_dim = 512

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.01
    batch_norm_epsilon = 1e-3

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
    max_epoch = 12

    # Learning rate management
    optimizer = 'sgd'
    learning_rate = 0.02
    momentum = 0.9
    lr_schedule = 'none'

    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 6
    val_batch_num = 6

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 438

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

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None

    weight_decay = 1e-5
    dropout = 0

    # Logging settings
    num_votes = 2
    no_validation = True

    # Validation split
    s3dis_validation_split = 4

    # Auto Mixed Precision
    amp = True

class LATS3DISConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'S3DIS'

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
    architecture = ['simple_k15_i5_o32',
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

                    'pids_deformable_e4_k15_i160_o320',

                    'nearest_upsample',
                    'unary_i416_o160',      #   =320+96
                    'nearest_upsample',
                    'unary_i192_o96',     #  =160+32
                    'nearest_upsample',
                    'unary_i120_o64',      # =96+24
                    'nearest_upsample',
                    'unary_i80_o32']       # =64+16


    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 1.5

    # Number of kernel points
    num_kernel_points = -1

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.04

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
    in_features_dim = 5

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
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
    lr_schedule = 'cosine'
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 8
    val_batch_num = 8

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 50

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

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None

    weight_decay = 1e-4
    dropout = 0

    # Validation split
    s3dis_validation_split = 4
