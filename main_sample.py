# pylint: disable=consider-using-f-string
import pickle
import os
import sys
sys.path.append(os.getcwd())
import argparse
import shutil
from typing import (
    Optional
)
import numpy as np
import torch

# Archs
from pids_core.models.architectures import (
    PIDS_Classification,
    PIDS_Segmentation,
)
from pids_core.utils.trainer import ModelTrainer
from pids_core.utils.tester import ModelTester
# Dataset and resource
from pids_search_space.pids_architect import (
    PIDS_Space_Cls,
    PIDS_Space_Seg, 
    build_pids_comb_search_space_from_genotype
)
from pids_search_space.genotype import (
    PIDS_seg_search_space_cfgs,
    PIDS_cls_search_space_cfgs,
)
from dataset_utils.data_pipe import (
    get_SemanticKITTI_dataset,
    get_S3DIS_dataset,
    get_ModelNet40_dataset)
# Import Config
from cfgs.modelnet40 import SearchModelnet40Config
from cfgs.semantic_kitti import (
    SearchSemanticKittiConfig,
    )
from cfgs.s3dis import SearchS3DISConfig
from utils.profile_utils import (
    get_flops_and_params_and_batch_size,
    get_latency,
)

from nasflow.io_utils.base_io import maybe_load_pickle_file

CFG_MODEL_ZOO = {
    'search_modelnet40_cfg': SearchModelnet40Config,
    'search_semantickitti_cfg': SearchSemanticKittiConfig,
    'search_s3dis_cfg': SearchS3DISConfig,
}

def load_config_and_train_modelnet40(config,
                                     model_root=None,
                                     train_random_seed: Optional[int] = None):
    if train_random_seed is not None:
        torch.manual_seed(train_random_seed)
        np.random.seed(train_random_seed)
    # Very important. We should init configs.
    config.__init__()

    # debug_timing(test_dataset, test_sampler, test_loader)
    # debug_show_clouds(training_dataset, training_sampler, training_loader)
    training_loader, test_loader, _, _, _, _ \
        = get_ModelNet40_dataset("search", config)

    print('\nModel Preparation')
    print('*****************')
    # Define network model
    net = PIDS_Classification(config)
    net = net.cuda()
    net.eval()
    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))

    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))
    net.train()
    trainer = ModelTrainer(net, config, chkp_path=None)
    trainer.train(net, training_loader, test_loader, config)
    # Define a trainer class
    chosen_chkp = os.path.join(config.saving_path, 'checkpoints', 'current_chkp.tar')
    tester = ModelTester(net, chkp_path=chosen_chkp)
    acc = tester.classification_test(net, test_loader, config, model_root=model_root)
    print("Test Acc: {:.4f}".format(acc))
    shutil.rmtree(config.saving_path, ignore_errors=True)
    shutil.rmtree(os.path.join(model_root, "test"), ignore_errors=True)

    return acc, flops, params, latency, batch_size

def load_config_and_train_semantickitti(config,
                                        model_root=None,
                                        train_random_seed: Optional[int] = None):
    if train_random_seed is not None:
        torch.manual_seed(train_random_seed)
        np.random.seed(train_random_seed)
    config.__init__()
    training_loader, test_loader, training_dataset, _, _, _ \
        = get_SemanticKITTI_dataset("search", config)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    net = PIDS_Segmentation(config, training_dataset.label_values, training_dataset.ignored_labels)
    net = net.cuda()
    print(config.architecture)
    print(net)
    # Measure FLOPS.
    net.eval()
    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))
    net.train()
    trainer = ModelTrainer(net, config, chkp_path=None)
    trainer.train(net, training_loader, test_loader, config)
    # Define a trainer class
    chosen_chkp = os.path.join(config.saving_path, 'checkpoints', 'current_chkp.tar')
    tester = ModelTester(net, chkp_path=chosen_chkp)
    acc = tester.slam_segmentation_test(net, test_loader, config, model_root=model_root)
    print("Test Acc: {:.4f}".format(acc))
    shutil.rmtree(config.saving_path, ignore_errors=True)
    shutil.rmtree(os.path.join(model_root, "test"), ignore_errors=True)
    return acc, flops, params, latency, batch_size

def get_flops_semantickitti(config,
                            model_root=None,
                            train_random_seed: Optional[int] = None):
    if train_random_seed is not None:
        torch.manual_seed(train_random_seed)
        np.random.seed(train_random_seed)
    config.__init__()
    _, test_loader, training_dataset, _, _, _ \
        = get_SemanticKITTI_dataset("search", config)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    net = PIDS_Segmentation(config, training_dataset.label_values, training_dataset.ignored_labels)
    net = net.cuda()
    print(config.architecture)
    # Measure FLOPS.
    net.eval()
    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))
    return 0, flops, params, latency, batch_size


def get_flops_modelnet40(config,
                         model_root=None,
                         train_random_seed: Optional[int] = None):
    if train_random_seed is not None:
        torch.manual_seed(train_random_seed)
        np.random.seed(train_random_seed)
    # Very important. We should init configs.
    config.__init__()

    # debug_timing(test_dataset, test_sampler, test_loader)
    # debug_show_clouds(training_dataset, training_sampler, training_loader)
    training_loader, test_loader, _, _, _, _ \
        = get_ModelNet40_dataset("search", config)
    print('\nModel Preparation')
    print('*****************')
    # Define network model
    net = KPCNN(config)
    net = net.cuda()
    print(config.architecture)
    net.eval()
    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))
    return 0, flops, params, latency, batch_size

def load_config_and_train_s3dis(config,
                                model_root=None,
                                train_random_seed: Optional[int] = None):
    if train_random_seed is not None:
        torch.manual_seed(train_random_seed)
        np.random.seed(train_random_seed)

    config.__init__()
    training_loader, test_loader, training_dataset, _, _, _ \
        = get_S3DIS_dataset("search", config)
    print('\nModel Preparation')
    print('*****************')
    # Define network model
    net = PIDS_Segmentation(config, training_dataset.label_values, training_dataset.ignored_labels)
    net = net.cuda()
    print(config.architecture)
    print(net)
    net.eval()
    flops, params, batch_size = get_flops_and_params_and_batch_size(net, config, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    latency = get_latency(net, config, test_loader)
    print("Latency: {} ms".format(latency * 1000))
    net.train()
    trainer = ModelTrainer(net, config, chkp_path=None)
    trainer.train(net, training_loader, test_loader, config)
    # Define a trainer class
    chosen_chkp = os.path.join(config.saving_path, 'checkpoints', 'current_chkp.tar')

    tester = ModelTester(net, chkp_path=chosen_chkp)
    acc = tester.cloud_segmentation_test(net, test_loader, config, model_root=model_root)
    print("Test Acc: {:.4f}".format(acc))
    shutil.rmtree(config.saving_path, ignore_errors=True)
    shutil.rmtree(os.path.join(model_root, "test"), ignore_errors=True)
    return acc, flops, params, latency, batch_size

TASK_EVAL_FN = {
    'modelnet40': load_config_and_train_modelnet40,
    'semantickitti': load_config_and_train_semantickitti,
    "s3dis": load_config_and_train_s3dis,
    'semantickitti-flops': get_flops_semantickitti,
    'modelnet40-flops': get_flops_modelnet40,
}

def eval_arch(task, config, model_root=None, train_random_seed=None):
    """
    Evaluate function that carries architecture performance evaluation per task.
    Args:
        task (str): Task to evaluate. Can be 'modelnet40', 'semantickitti' or 's3dis'.
        config (Any): Evaluation configuration.
        model_root (Optional[str]): A string indicating the save path for a model.
    Return:
        Evaluation results.
    """
    if not task in TASK_EVAL_FN:
        raise NotImplementedError("Task {} is not supported!".format(task))
    return TASK_EVAL_FN[task](config,
                              model_root=model_root,
                              train_random_seed=train_random_seed)

def main(args):
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)

    if args.task in ["modelnet40", "modelnet40-flops"]:			# Search for KPCNN
        search_space_cfgs = PIDS_cls_search_space_cfgs()
        pids_search_spaces = build_pids_comb_search_space_from_genotype(search_space_cfgs)
        search_space = PIDS_Space_Cls(pids_search_spaces)
        config = CFG_MODEL_ZOO[args.search_config]()
        config.saving_path = os.path.join(args.model_root, "checkpoint")
        saved_record_path = os.path.join(args.model_root, "{}-pids.records".format(args.task))
        saved_records = []
        print(saved_record_path)
        if os.path.exists(saved_record_path):
            saved_records = maybe_load_pickle_file(saved_record_path)
            print("Loaded saved records. Records has {} items.".format(len(saved_records)))
        else:
            saved_records = []
            print("Initialize from scratch...")

    elif args.task in ["s3dis", "semantickitti", 'semantickitti-flops']:	# Search for KPFCN.
        search_space_cnn_cfg, search_space_fcn_cfg = PIDS_seg_search_space_cfgs()
        all_search_cfg = search_space_cnn_cfg + search_space_fcn_cfg
        pids_search_spaces = build_pids_comb_search_space_from_genotype(all_search_cfg)
        search_space = PIDS_Space_Seg(pids_search_spaces,
                                      num_cnn_search_spaces=len(search_space_cnn_cfg),
                                      num_fcn_search_spaces=len(search_space_fcn_cfg),
                                    )
        config = CFG_MODEL_ZOO[args.search_config]()
        saved_records = []
        config.saving_path = os.path.join(args.model_root, "checkpoint")

        saved_record_path = os.path.join(args.model_root, "{}-pids.records".format(args.task))
        if os.path.exists(saved_record_path):
            saved_records = maybe_load_pickle_file(saved_record_path)
            print("Loaded saved records. Records has {} items.".format(len(saved_records)))
        else:
            saved_records = []
            print("Initialize from scratch...")

    for idx in range(args.budget):
        print("Evaluating {:d} of {:d} architectures!".format(idx, args.budget))
        search_space.__seed__()
        argscope = search_space.generate_argscope()
        config.architecture = search_space.generate_arch_specs_from_argscope(
            argscope,
            num_stem_inp=config.in_features_dim)
        acc, flops, params, latency, batch_size = eval_arch(
            args.task,
            config,
            model_root=args.model_root,
            train_random_seed=args.train_random_seed)
        record = {
            'acc': acc,
            'params (M)': params / 1e6,
            'flops (M)': flops / 1e6,
            'latency (ms)': latency * 1000,
            'batch_size': batch_size,
            'arch': config.architecture,
            'block_args': search_space.str_encode()}
        saved_records.append(record)
        with open(saved_record_path, 'wb') as fp:
            pickle.dump(saved_records, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=300,
                        help="Budget to sample.")
    parser.add_argument("--model_root", type=str, default=None,
                        help="Model root.")
    parser.add_argument("--task", type=str, default="modelnet40",
                        help="Task to evaluate.",
                        choices=['modelnet40', 's3dis', 'semantickitti', 
                        'semantickitti-flops', 'modelnet40-flops'])
    parser.add_argument("--search_config", type=str, default=None,
                        help="Search Config")
    parser.add_argument("--sample_random_seed", type=int, default=None,
                        help="Random seed for sampling archs.")
    parser.add_argument("--train_random_seed", type=int, default=None,
                        help="Random seed used to initialize proxy training.")
    global_args = parser.parse_args()
    main(global_args)
