# Common libs
import signal
import os
import sys
sys.path.append(os.getcwd())
import argparse
import time

from pids_core.utils.trainer import ModelTrainer
from pids_core.models.architectures import PIDS_Segmentation

# Dataset
from dataset_utils.data_pipe import get_S3DIS_dataset
from cfgs.s3dis import (
    Mv2S3DISConfig,
    Mv2S3DISConfigBlockAttn,
    Mv2S3DISConfigEncDecAttn,
    Mv2S3DISConfigAttnAll,
    GoldenS3DISConfig,
    SimpleViewS3DISConfig
)
from pids_search_space import arch_genotype
from pids_search_space.genotype import PIDS_seg_search_space_cfgs
from pids_search_space.pids_architect import (
    PIDS_Space_Seg,
    build_pids_comb_search_space_from_genotype)
from pids_search_space.utils import compound_scaling
from utils.profile_utils import (
    get_flops_and_params_and_batch_size,
    get_latency,
)

from training.train_SemanticKitti import get_block_args_from_config

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

TRAINING_CFG = {
    'mbnetv2': Mv2S3DISConfig,
    'golden': GoldenS3DISConfig,
    'simpleview': SimpleViewS3DISConfig,
}
def main(args):
    chosen_chkp = None
    # Initialize configuration class
    config = TRAINING_CFG[args.train_cfg]()

    # Get path from argument if given
    if args.save_path is not None:
        config.saving_path = args.save_path

    config.first_subsampling_dl = args.downsample_rate
    block_args = get_block_args_from_config(args.block_args_cfg)
    block_args = compound_scaling(block_args, args.k_multiplier, args.d_multiplier, args.c_multiplier)
    config.stem_conv_dim = int(config.stem_conv_dim * args.c_multiplier)
    print("New block args: {}".format(block_args))
    search_space_cnn_cfg, search_space_fcn_cfg = PIDS_seg_search_space_cfgs()
    search_space_cfg = search_space_cnn_cfg + search_space_fcn_cfg
    pids_search_spaces = build_pids_comb_search_space_from_genotype(search_space_cfg)
    search_space = PIDS_Space_Seg(pids_search_spaces,
                                        num_cnn_search_spaces=len(search_space_cnn_cfg),
                                        num_fcn_search_spaces=len(search_space_fcn_cfg))
    search_space.decode_str_encoding(block_args)
    argscope = search_space.generate_argscope()
    architecture = search_space.generate_arch_specs_from_argscope(argscope=argscope,
                                                                  num_stem_inp=config.in_features_dim,
                                                                  num_stem_oup=config.stem_conv_dim)
    config.architecture = architecture
    # Configure Hyperparameters.
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.optimizer is not None:
        config.optimizer = args.optimizer

    print("Validate on Area {}!".format(args.validate_area))
    config.s3dis_validation_split = args.validate_area - 1
    config.__init__()

    print(vars(config))

    training_loader, test_loader, training_dataset, _, _, _ \
        = get_S3DIS_dataset("eval", config)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = PIDS_Segmentation(config, training_dataset.label_values, training_dataset.ignored_labels,
                 enable_encoder_decoder_attention=config.use_enc_dec_attn)
    
    net = net.cuda()
    net.eval()
    print(net)

    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))

    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_multiplier", type=float, default=1.0,
                        help="Compound scaling factor for k.")
    parser.add_argument("--c_multiplier", type=float, default=1.0,
                        help="Compound scaling factor for c.")
    parser.add_argument("--d_multiplier", type=float, default=1.0,
                        help="Compound scaling factor for r.")
    parser.add_argument("--downsample_rate", type=float, default=0.04,
                        help="Downsample rate for Modelnet40")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Save path.")
    parser.add_argument("--block_args_cfg", type=str, default="mbnetv2",
                        help="Block args to determine the source of architecture.")
    parser.add_argument("--train_cfg", type=str, default="mbnetv2",
                        help="Training configuration of the architecture.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate that overrides the config.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay that overrides the config.")
    parser.add_argument("--validate_area", type=int, default=5,
                        help="Area to validate.")
    parser.add_argument("--optimizer", type=str, default=None,
        help="Optimizer that used to optimize 3D training.") 
    global_args = parser.parse_args()
    main(global_args)
