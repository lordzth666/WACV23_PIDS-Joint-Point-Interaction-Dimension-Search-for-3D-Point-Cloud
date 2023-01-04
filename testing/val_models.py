
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import sys
import copy
sys.path.append(os.getcwd())
import numpy as np
import torch
import argparse

# Dataset
from pids_core.datasets.ModelNet40 import *
from pids_core.datasets.S3DIS import *
from pids_core.datasets.SemanticKitti import *
from torch.utils.data import DataLoader

from pids_core.utils.config import Config
from pids_core.utils.tester import ModelTester
from pids_core.models.architectures import (
    PIDS_Classification,
    PIDS_Segmentation,
)
from utils.profile_utils import (
    get_flops_and_params_and_batch_size,
    get_latency,
)
from dataset_utils.data_pipe import (
    get_SemanticKITTI_dataset,
    get_S3DIS_dataset,
    get_ModelNet40_dataset
)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#
def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None,
                        help="Path to load the result.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name of the model")
    args = parser.parse_args()
    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # chosen_log = 'results/Log_2020-12-05_05-00-16'  # => ModelNet40
    chosen_log = args.result_path  # => ModelNet40

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    # GPU_ID = '0'

    # Set GPU visible device
    # os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'chkp_best.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)
    config.__init__()

    #print(vars(config))
    #exit(-1)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    config.validation_size = 200
    config.input_threads = 8
    
    # Disable test-time augmentation.
    if config.dataset != "ModelNet40":
        config.augment_scale_min = 0.999999
        config.augment_scale_max = 1.000001
        config.augment_noise = 0.0
        config.augment_color = 1.0
        translate_scale = 0.0
    
    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'ModelNet40':
        training_loader, test_loader, _, _, _, _ = get_ModelNet40_dataset("eval", config)
    elif config.dataset == 'S3DIS':
        training_loader, test_loader, training_dataset, test_dataset, _, _ \
            = get_S3DIS_dataset("eval", config)
    elif config.dataset == 'SemanticKitti':
        training_loader, test_loader, training_dataset, test_dataset, _, _ \
            = get_SemanticKITTI_dataset("eval", config, balance_class_test=False)
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    print('\nModel Preparation')
    config_latency = copy.deepcopy(config)
    config_latency.use_batch_norm = False
    config_latency.dropout = 0.0

    # Define network model for latency measurment and FLOPS count.
    t1 = time.time()
    if config.dataset_task == 'classification':
        net_no_batchnorm = PIDS_Classification(config_latency)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net_no_batchnorm = PIDS_Segmentation(config_latency, 
            test_dataset.label_values, 
            test_dataset.ignored_labels, 
            enable_encoder_decoder_attention=config_latency.use_enc_dec_attn,)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config_latency.dataset_task)
    net_no_batchnorm = net_no_batchnorm.cuda()
    net_no_batchnorm.eval()
    flops, params, batch_size = get_flops_and_params_and_batch_size(net_no_batchnorm, config_latency, test_loader)
    print("FLOPS: {} M, Params: {} M, Mean Batch Size: {}".format(
        flops / 1e6, params / 1e6, batch_size))
    # print(net_no_batchnorm)
    latency = get_latency(net_no_batchnorm, config_latency, test_loader)
    print("Latency: {} ms".format(latency * 1000))

    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = PIDS_Classification(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = PIDS_Segmentation(config, 
            test_dataset.label_values, 
            test_dataset.ignored_labels, 
            enable_encoder_decoder_attention=config.use_enc_dec_attn,)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # No need to wrap EMA. weights will be loaded anyway.
    print(net)
    net = net.cuda()
    net.eval()

    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('\nStart test')
    print('**********\n')

    # Assign a model root for testing.
    if args.model_name is None:
        model_root = "test_{}_{}".format(config.dataset_task, int(time.time()))
    else:
        model_root = "test_{}_{}".format(config.dataset_task, args.model_name)
    print("Testing model in {}".format(model_root))

    # Testing
    with torch.no_grad():
        if config.dataset_task == 'classification':
            tester.classification_test(net, test_loader, config, model_root=model_root)
        elif config.dataset_task == 'cloud_segmentation':
            tester.cloud_segmentation_test(net, test_loader, config, model_root=model_root)
        elif config.dataset_task == 'slam_segmentation':
            tester.slam_segmentation_test(net, test_loader, config, model_root=model_root)
        else:
            raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
