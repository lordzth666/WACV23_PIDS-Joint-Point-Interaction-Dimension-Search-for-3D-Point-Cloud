import os
import sys
sys.path.append(os.getcwd())
from typing import (
    Optional,
    List,
    Any
)
from tqdm import tqdm
from math import exp
import argparse
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.stats.mstats import kendalltau
import nevergrad as ng2

from nasflow.io_utils.base_io import maybe_write_json_file
from nasflow.dataset.nas_dataset import NASDataSet
from nasflow.algo.optimization.nevergrad_opt import (
    NeverGradNGOpt,
    NeverGradDEOpt
)
from nasflow.io_utils.base_parser import parse_args_from_kwargs

from predictor.predictor_model_zoo import get_predictor
from predictor.hparams import Hparams

from train_predictor import (
    PointClsMapFn,
    PointSegMapFn
)

def hash_fn(a):
    return "".join([str(int(x)) for x in a])

@torch.no_grad()
def eval_corrleation_with_predictor(dataset_iterator, predictor):
    all_outputs = []
    all_labels = []
    predictor.eval()
    for _, batch in enumerate(dataset_iterator):
        inputs, labels = [x[0] for x in batch], [x[1] for x in batch]
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)
        if predictor.use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = predictor.predict(inputs)
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, 0)
    all_labels = torch.cat(all_labels, 0)
    all_outputs = all_outputs.cpu().numpy().flatten()
    all_labels = all_labels.cpu().numpy()
    pearson_r, _ = pearsonr(all_outputs.flatten(), all_labels)
    kendall_tau, _ = kendalltau(all_outputs.flatten(), all_labels)
    mse_loss = np.mean(np.square(all_outputs.flatten() - all_labels))
    return pearson_r, kendall_tau, mse_loss

def train_and_evaluate_per_predictor(
        nas_dataset,
        num_inputs: int,
        in_dims: int,
        args: Any,
        verbose: bool = False,
        num_epochs: int = 150,
        batch_size: int = 64,
        **kwargs):
    """
    Train & Evaluate a predictor using different hyperparameters.
    """
    predictor = get_predictor(
        args.nn_arch,
        nas_dataset,
        in_dims,
        num_epochs,
        num_inputs,
        args.predictor_loss_fn_name,
        args.ranking_loss_fn_name,
        batch_size,
        **kwargs)
    # print(predictor.core_ml_arch)
    predictor.load_weights(args.pretrain_ckpt_path, args.pretrain_exclude_ckpt_keys)
    predictor.fit(verbose)
    # predictor.save_weights(args.save_ckpt_path)

    predictor.eval()
    train_dataset_iterator = nas_dataset.iter_map_and_batch(
        split='train', shuffle=False, drop_last_batch=False, batch_size=128)
    test_dataset_iterator = nas_dataset.iter_map_and_batch(
        split='test', shuffle=False, drop_last_batch=False, batch_size=128)
    train_pearson, train_kendall, train_loss = eval_corrleation_with_predictor(train_dataset_iterator, predictor)
    test_pearson, test_kendall, test_loss = eval_corrleation_with_predictor(test_dataset_iterator, predictor)
    
    return {
        'train_kendall_tau': train_kendall,
        'train_pearson_rau': train_pearson,
        'test_kendall_tau': test_kendall,
        'test_pearson_rau': test_pearson,
        'test_mse_loss': 99999 if np.isnan(test_loss) else test_loss
    }

def main(args):
    # in_dims = 151
    if args.task in ['modelnet40']:
        num_inputs = 7
        in_dims = 7 * 64 * num_inputs
        map_fn_records_onehot = PointClsMapFn().map_fn_onehot
        map_fn_records_ordinal = lambda x: PointClsMapFn().map_fn_ordinal(
            x, normalize=True, min_val=78.47, max_val=83.9)
        map_fn_records_dense = lambda x: PointClsMapFn().map_fn_dense(
            x, normalize=True, min_val=78.47, max_val=83.9)
        map_fn_records_dense_sparse = lambda x: PointClsMapFn().map_fn_dense_and_sparse(
            x, normalize=True, min_val=78.47, max_val=83.9)
    elif args.task in ['modelnet40-flops']:
        num_inputs = 7
        in_dims = 7 * 64 * num_inputs if args.map_fn_name != 'dense' else 7 * num_inputs
        map_fn_records_onehot = PointClsMapFn().map_fn_onehot
        map_fn_records_ordinal = PointClsMapFn().map_fn_ordinal_flops
        map_fn_records_dense = PointClsMapFn().map_fn_dense_flops
        map_fn_records_dense_sparse = PointClsMapFn().map_fn_dense_and_sparse_flops
    elif args.task in ['semantickitti-flops']:
        num_inputs = 11
        in_dims = 7 * 64 * num_inputs if args.map_fn_name != 'dense' else 7 * num_inputs
        map_fn_records_onehot = PointSegMapFn().map_fn_onehot
        map_fn_records_ordinal = PointSegMapFn().map_fn_ordinal_flops
        map_fn_records_dense = PointSegMapFn().map_fn_dense_flops
        map_fn_records_dense_sparse = PointSegMapFn().map_fn_dense_and_sparse_flops
    elif args.task in ['semantickitti']:
        num_inputs = 11
        in_dims = 7 * 64 * num_inputs if args.map_fn_name != 'dense' else 7 * num_inputs
        map_fn_records_onehot = PointSegMapFn().map_fn_onehot
        map_fn_records_ordinal = lambda x: PointSegMapFn().map_fn_ordinal(
            x, normalize=True, min_val=0.22802, max_val=0.27645)
        map_fn_records_dense_sparse = lambda x: PointSegMapFn().map_fn_dense_and_sparse(
            x, normalize=True, min_val=0.22802, max_val=0.27645)
        map_fn_records_dense = lambda x: PointSegMapFn().map_fn_dense(
            x, normalize=True, min_val=0.22802, max_val=0.27645)

    map_fn_lib = {
        'onehot': map_fn_records_onehot,
        'ordinal': map_fn_records_ordinal,
        'dense': map_fn_records_dense,
        'dense-sparse': map_fn_records_dense_sparse
    }
    map_fn = map_fn_lib[args.map_fn_name]


    # ModelNet-40 FLOPS hparams: 'learning_rate': 0.08149492357811797, 'weight_decay': 0.0008060759095245629, 'margin': 0.0017515664867573466
    # ModelNet-40 hparams: 'learning_rate': 0.05746790111573563, 'weight_decay': 0.00017620684974022291, 'margin': 0.007120979304889882, 'ranking_loss_coef': 0.15538152707628894
    hparams = Hparams(args.hparams_json_path)

    random_seeds = [233, 751, 7, 1001, 4, 11, 5, 3001, 9, 11]
    all_test_mses = []
    all_test_kendalls = []
    for seed in tqdm(random_seeds):
        nas_dataset = NASDataSet(
            args.root_dir,
            args.pattern,
            args.record_name,
            map_fn=map_fn,
            random_state=seed,
            cache=True)
        return_dict = train_and_evaluate_per_predictor(
            nas_dataset,
            num_inputs,
            in_dims,
            args,
            learning_rate=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
            margin=hparams.margin,
            ranking_loss_coef=hparams.ranking_loss_coef,
            verbose=False,
            num_epochs=150,
        )
        all_test_mses.append(return_dict['test_mse_loss'])
        all_test_kendalls.append(return_dict['test_kendall_tau'])

    mean_test_mses, std_test_mses = np.mean(all_test_mses), np.std(all_test_mses)
    mean_test_kendalls, std_test_kendalls = np.mean(all_test_kendalls), np.std(all_test_kendalls)
    print("Test MSE: {} +/- {}".format(mean_test_mses, std_test_mses))
    print("Test Kendall Tau: {} +/- {}".format(mean_test_kendalls, std_test_kendalls))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparams_json_path", type=str, default=None,
        help="Hparams configuration for single-run trials."
    )
    parser.add_argument(
        "--root_dir", type=str, default=None,
        help="Root directory for the dataset.")
    parser.add_argument(
        "--pattern", type=str, default=None,
        help="Pattern of the record files.")
    parser.add_argument(
        "--record_name", type=str, default=None,
        help="Record file name.")
    parser.add_argument(
        "--task", type=str, default="modelnet40",
        help="Task to train the predictor."
    )
    parser.add_argument(
        "--pretrain_ckpt_path", type=str, default=None,
        help="Path to the pretrained checkpoint."
    )
    parser.add_argument("--pretrain_exclude_ckpt_keys", type=str, nargs='*',
        default=None)
    parser.add_argument(
        "--save_ckpt_path", type=str, default=None,
        help="Path to the saved checkpoint."
    )
    parser.add_argument(
        '--nn_arch', type=str, default=None,
        help="NN architecture to train the predictor",
        choices=[
            'embedding-nn',
            'dense-nn',
            'dense-sparse-nn',]
    )
    parser.add_argument(
        '--map_fn_name', type=str, default=None,
        help="Map function when processing architecture dataset",
        choices=['one-hot', 'ordinal', 'dense-sparse', 'dense']
    )
    parser.add_argument(
        '--predictor_loss_fn_name', type=str, default='mse-loss',
        help="Predictor loss function name to choose from."
    )
    parser.add_argument(
        '--ranking_loss_fn_name', type=str, default='margin-ranking-loss',
        help="Ranking loss function name for predictor training."
    )
    parser.add_argument(
        "--opt_log_path", type=str, default=None,
        help="Optimization logging path."
    )
    global_args = parser.parse_args()
    main(global_args)
