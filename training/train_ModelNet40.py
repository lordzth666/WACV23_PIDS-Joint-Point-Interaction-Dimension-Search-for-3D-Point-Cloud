
import os
import sys
sys.path.append(os.getcwd())
# Common libs
import signal
import argparse
import time

# from ptflops import get_model_complexity_info

# Dataset
from pids_core.utils.trainer import ModelTrainer
from pids_core.models.architectures import PIDS_Classification
from cfgs.modelnet40 import (
    Mv2Modelnet40Config,
    Mv2Modelnet40ConfigBlockAttn,
    GoldenModelnet40Config,
    SimpleBaselineModelnet40Config,
)
from pids_search_space.pids_architect import (
    PIDS_Space_Cls,
    build_pids_comb_search_space_from_genotype)
from pids_search_space.genotype import PIDS_cls_search_space_cfgs
from pids_search_space.utils import compound_scaling
from dataset_utils.data_pipe import get_ModelNet40_dataset
from pids_search_space import arch_genotype
from utils.profile_utils import (
    get_flops_and_params_and_batch_size,
    get_latency,
)

def get_block_args_from_config(block_args_cfg):
    if block_args_cfg == 'golden':
        return arch_genotype.block_args_golden_modelnet40
    elif block_args_cfg == 'mbnetv2':
        return arch_genotype.block_args_mobilenetv2_kpcnn_no_attn
    elif block_args_cfg == "mbnetv2-interblock-attn":
        return arch_genotype.block_args_mobilenetv2_kpcnn_attn
    elif block_args_cfg.startswith("embedding-ea"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpcnn_embedding_nn_predictor_regularized_ea[key_name]
    elif block_args_cfg.startswith("dense-sparse-ea-flops-opt-enlarge-"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpcnn_dense_sparse_predictor_regularized_ea_flops_opt_enlarge[key_name]
    elif block_args_cfg.startswith("dense-sparse-ea-flops-opt-"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpcnn_dense_sparse_predictor_regularized_ea_flops_opt_enlarge[key_name]
    elif block_args_cfg.startswith("dense-sparse-ea"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpcnn_dense_sparse_predictor_regularized_ea[key_name]
    elif block_args_cfg.startswith("random"):
        key_name = block_args_cfg.split("-")[-1]
        return arch_genotype.block_args_kpcnn_random[key_name]

TRAINING_CFG = {
    'mbnetv2': Mv2Modelnet40Config,
    'mbnetv2-interblock-attn': Mv2Modelnet40ConfigBlockAttn,
    'golden': GoldenModelnet40Config,
    'simple-baseline': SimpleBaselineModelnet40Config,
}

def main(args):
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chosen_chkp = None
    ##############
    # Prepare Data
    ##############
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = TRAINING_CFG[args.train_cfg]()
    if args.downsample_rate is not None:
        config.first_subsampling_dl = args.downsample_rate
    # Get path from argument if given
    if args.save_path is not None:
        config.saving_path = args.save_path

    if args.block_args_cfg is not None:
        print("\nDecoding block args ...")
        search_space_cnn_cfgs = PIDS_cls_search_space_cfgs()
        pids_search_spaces = build_pids_comb_search_space_from_genotype(search_space_cnn_cfgs)
        search_space = PIDS_Space_Cls(pids_search_spaces)
        block_args = get_block_args_from_config(args.block_args_cfg)
        block_args = compound_scaling(block_args, args.k_multiplier, args.d_multiplier, args.c_multiplier)
        config.stem_conv_dim = int(config.stem_conv_dim * args.c_multiplier)
        print("New block args: {}".format(block_args))
        search_space.decode_str_encoding(str_encoding=block_args)
        argscope = search_space.generate_argscope()
        architecture = search_space.generate_arch_specs_from_argscope(
            argscope, 
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

    config.__init__()

    training_loader, test_loader, _, _, _, _ = get_ModelNet40_dataset("eval", config)
    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = PIDS_Classification(config)
    net = net.cuda()
    print(config.architecture)
    print(net)
    # Measure FLOPS.
    net.eval()
    
    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    """
    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))
    """
    # flops, params = get_model_complexity_info(net, input_res=(60000, 1), input_constructor=input_constructor)
    # print("FLOPS: {:.4f} (G)" %(flops / 1e9))
    # print("Params: {:.4f} (M)" % (flops / 1e6))
    # net = torch.nn.DataParallel(net)

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    try:
        trainer.train(net, training_loader, test_loader, config)
    except Exception:
        print('Caught an error')
        os.kill(os.getpid(), signal.SIGINT)

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
    parser.add_argument("--downsample_rate", type=float, default=None,
                        help="Downsample rate for Modelnet40")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Save path.")
    parser.add_argument("--block_args_cfg", type=str, default=None,
                        help="Block args to determine the source of architecture.")
    parser.add_argument("--train_cfg", type=str, default="mbnetv2",
                        help="Training configuration of the architecture.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate that overrides the config.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay that overrides the config.")
    parser.add_argument("--optimizer", type=str, default=None,
        help="Optimizer that used to optimize 3D training.") 
    global_args = parser.parse_args()
    main(global_args)
