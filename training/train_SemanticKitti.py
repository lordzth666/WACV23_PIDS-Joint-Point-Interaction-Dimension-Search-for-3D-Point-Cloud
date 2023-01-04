# Common libs
import signal
import os
import sys
sys.path.append(os.getcwd())
import argparse
import time

import torch
# Dataset
from pids_core.utils.trainer import ModelTrainer
from pids_core.models.architectures import PIDS_Segmentation
from cfgs.semantic_kitti import (
    ResNet50SemanticKittiConfigAdv,
    Mv2SemanticKittiConfig,
    Mv2SemanticKittiConfigBlockAttn,
    Mv2SemanticKittiConfigEncDecAttn,
    Mv2SemanticKittiConfigAttnAll,
    GoldenSemanticKittiConfig,
    SimpleViewSemanticKittiConfig,
)

from dataset_utils.data_pipe import get_SemanticKITTI_dataset
from pids_search_space.genotype import PIDS_seg_search_space_cfgs
from pids_search_space.utils import compound_scaling
from pids_search_space import arch_genotype
from pids_search_space.pids_architect import (
    PIDS_Space_Seg, 
    build_pids_comb_search_space_from_genotype)
from utils.profile_utils import (
    get_flops_and_params_and_batch_size,
    get_latency,
)

def get_block_args_from_config(block_args_cfg):
    if block_args_cfg == 'golden':
        return arch_genotype.block_args_golden_semantickitti
    elif block_args_cfg == "golden-xlarge":
        return arch_genotype.block_args_golden_semantickitti_xlarge
    elif block_args_cfg == "golden-2xlarge":
        return arch_genotype.block_args_golden_semantickitti_2xlarge
    elif block_args_cfg == 'mbnetv2':
        return arch_genotype.block_args_mobilenetv2_kpfcn_no_attn
    elif block_args_cfg == "mbnetv2-interblock-attn":
        return arch_genotype.block_args_mobilenetv2_kpfcn_attn
    elif block_args_cfg == "golden-mobile":
        return arch_genotype.block_args_golden_mobile_semantickitti
    elif block_args_cfg.startswith("dense-sparse-ea-flops-opt-enlarge-"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpfcn_dense_sparse_predictor_regularized_ea_flops_opt_enlarge[key_name]
    elif block_args_cfg.startswith("dense-sparse-ea-flops-opt-"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpfcn_dense_sparse_predictor_regularized_ea_flops_opt[key_name]
    elif block_args_cfg.startswith("dense-sparse-ea-"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpfcn_dense_sparse_predictor_regularized_ea[key_name]
    elif block_args_cfg.startswith("embedding-ea-"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpfcn_embedding_nn_predictor_regularized_ea[key_name]
    elif block_args_cfg.startswith("random"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpfcn_random[key_name]


TRAINING_CFG = {
    'mbnetv2': Mv2SemanticKittiConfig,
    'golden': GoldenSemanticKittiConfig,
    'kpconv-default': ResNet50SemanticKittiConfigAdv,
    'simpleview': SimpleViewSemanticKittiConfig,
}

def main(args):
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chosen_chkp = None
    print('Data Preparation')
    print('****************')
    # Initialize configuration class
    config = TRAINING_CFG[args.train_cfg]()
    config.first_subsampling_dl = args.downsample_rate
    # Get path from argument if given
    if args.save_path is not None:
        config.saving_path = args.save_path
    if args.train_cfg != "kpconv-default" and args.block_args_cfg is not None:
        block_args = get_block_args_from_config(args.block_args_cfg)
        print("Decoding Block args...")
        block_args = compound_scaling(block_args, args.k_multiplier, args.d_multiplier, args.c_multiplier)
        # Scale the first input dimension.
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
        config.__init__()

    print(vars(config))
    #exit(-1)

    training_loader, test_loader, training_dataset, _, _, _ \
        = get_SemanticKITTI_dataset(args.split, config)
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_class_w(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')
    # Define network model
    t1 = time.time()
    net = PIDS_Segmentation(
        config, training_dataset.label_values, training_dataset.ignored_labels, 
        enable_encoder_decoder_attention=config.use_enc_dec_attn)
    net = net.cuda()
    net.eval()
    print(net)
    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)    
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    
    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))

    # Configure Hyperparameters.
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.optimizer is not None:
        config.optimizer = args.optimizer

    #tester = ModelTester(net, chkp_path=chosen_chkp)
    #with torch.no_grad():
    #    test_mean_flops, test_std_flops, mean_batch_size, std_batch_size = tester.test_flops(net, test_loader, config)
    #print("FLOPS: {:.2f} G".format(test_mean_flops / 1e9))
    #print("Mean batch size: {}".format(mean_batch_size))

    #print('Done in {:.1f}s\n'.format(time.time() - t1))

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('**************')
    print("Training config: {}")
    print(vars(config))
    print('**************')

    print('\nStart training')
    print('**************')
    
    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=None,
        help="Path to save the checkpoints.")
    parser.add_argument("--k_multiplier", type=float, default=1.0,
        help="Compound scaling factor for k.")
    parser.add_argument("--c_multiplier", type=float, default=1.0,
        help="Compound scaling factor for c.")
    parser.add_argument("--d_multiplier", type=float, default=1.0,
        help="Compound scaling factor for r.")
    parser.add_argument("--downsample_rate", type=float, default=0.06,
        help="Downsample rate for Modelnet40")
    parser.add_argument("--block_args_cfg", type=str, default=None,
        help="Block args to determine the source of architecture.")
    parser.add_argument("--train_cfg", type=str, default="mbnetv2",
        help="Training configuration of the architecture.")
    parser.add_argument("--split", type=str, default="eval",
        help="Data split.")
    # Cofnigs that overrides default settings.
    parser.add_argument("--optimizer", type=str, default=None,
        help="Optimizer that used to optimize 3D training.") 
    parser.add_argument("--learning_rate", type=float, default=None,
        help="Learning rate that overrides the config.")
    parser.add_argument("--weight_decay", type=float, default=None,
        help="Weight decay that overrides the config.")
    global_args = parser.parse_args()
    main(global_args)
