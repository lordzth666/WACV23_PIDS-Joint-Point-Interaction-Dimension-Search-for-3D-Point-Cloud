import os
import sys
sys.path.append(os.getcwd())
from typing import (
    Optional,
    List,
    Any
)
from math import exp
import argparse
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.stats.mstats import kendalltau
import nevergrad as ng

from nasflow.io_utils.base_io import maybe_write_json_file
from nasflow.dataset.nas_dataset import NASDataSet
from nasflow.algo.optimization.nevergrad_opt import (
    NeverGradNGOpt,
    NeverGradDEOpt
)
from nasflow.io_utils.base_parser import parse_args_from_kwargs

from pids_search_space.pids_architect import (
    PIDS_Space_Cls,
    PIDS_Space_Seg,
    build_pids_comb_search_space_from_genotype,
) 

from pids_search_space.genotype import (
    PIDS_cls_search_space_cfgs,
    PIDS_seg_search_space_cfgs,
)
from predictor.predictor_model_zoo import get_predictor
from predictor.hparams import Hparams

class PointClsMapFn:
    def __init__(self):
        search_space_cnn_cfg = PIDS_cls_search_space_cfgs()
        pids_search_spaces = build_pids_comb_search_space_from_genotype(search_space_cnn_cfg)
        self.search_space = PIDS_Space_Cls(
                pids_search_spaces,
                )

    def map_fn_onehot(self, x):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode()
        return encode_list, x['acc'].item()

    def map_fn_ordinal_flops(self, x):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='ordinal')
        flops = x['flops (M)'].item() / 1000.
        return encode_list, flops

    def map_fn_ordinal(self, x, normalize=True, min_val=None, max_val=None):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='ordinal')
        acc = (x['acc'].item() - min_val) / (max_val - min_val) if normalize else x['acc'].item()
        return encode_list, acc

    def map_fn_dense(self, x, normalize=True, min_val=None, max_val=None):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-encoding')
        dense_enc = []
        for item in encode_list:
            dense_enc += item
        acc = (x['acc'].item() - min_val) / (max_val - min_val) if normalize else x['acc'].item()
        return dense_enc, acc

    def map_fn_dense_flops(self, x):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-encoding')
        dense_enc = []
        for item in encode_list:
            dense_enc += item
        flops = x['flops (M)'].item() / 1000.
        return dense_enc, flops


    @staticmethod
    def sigmoid(x):
        return 1. / (1. + exp(0 - x))

    def map_fn_dense_and_sparse(self, x, normalize=True, min_val=None, max_val=None):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-sparse-encoding')
        assert len(encode_list) % 3 == 0, \
            ValueError("Encode list after 'dense-sparse-encoding' must be divisible by 3!")
        dense_pos_enc, dense_arch_enc, sparse_attn_enc = [], [], []
        for idx in range(len(encode_list) // 3):
            dense_pos_enc.extend(encode_list[idx * 3])
            dense_arch_enc.extend(encode_list[idx * 3 + 1])
            sparse_attn_enc.extend(encode_list[idx * 3 + 2])
        full_dense_sparse_enc = dense_pos_enc + dense_arch_enc + sparse_attn_enc
        acc = (x['acc'].item() - min_val) / (max_val - min_val) if normalize else x['acc'].item()
        return full_dense_sparse_enc, acc

    def map_fn_dense_and_sparse_flops(self, x):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-sparse-encoding')
        assert len(encode_list) % 3 == 0, \
            ValueError("Encode list after 'dense-sparse-encoding' must be divisible by 3!")
        dense_pos_enc, dense_arch_enc, sparse_attn_enc = [], [], []
        for idx in range(len(encode_list) // 3):
            dense_pos_enc.extend(encode_list[idx * 3])
            dense_arch_enc.extend(encode_list[idx * 3 + 1])
            sparse_attn_enc.extend(encode_list[idx * 3 + 2])
        full_dense_sparse_enc = dense_pos_enc + dense_arch_enc + sparse_attn_enc
        flops = x['flops (M)'].item() / 1000.
        return full_dense_sparse_enc, flops
    
class PointSegMapFn:
    def __init__(self):
        search_space_cnn_cfg, search_space_fcn_cfg = PIDS_seg_search_space_cfgs()
        all_search_cfg = search_space_cnn_cfg + search_space_fcn_cfg
        pids_search_spaces = build_pids_comb_search_space_from_genotype(all_search_cfg)
        self.search_space = PIDS_Space_Seg(
            pids_search_spaces,
            num_cnn_search_spaces=len(search_space_cnn_cfg),
            num_fcn_search_spaces=len(search_space_fcn_cfg),
        )

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + exp(0 - x))

    def map_fn_onehot(self, x):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(aggregate=True)
        return encode_list, x['acc'].item()

    def map_fn_ordinal(self, x, normalize=True, min_val=None, max_val=None):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='ordinal', aggregate=True)
        acc = (x['acc'].item() - min_val) / (max_val - min_val) if normalize else x['acc'].item()
        return encode_list, acc

    def map_fn_ordinal_flops(self, x):  
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='ordinal', aggregate=True)
        return encode_list, x['flops (M)'].item() / 1000.

    def map_fn_dense_and_sparse(self, x, normalize=True, min_val=None, max_val=None):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-sparse-encoding', aggregate=True)
        assert len(encode_list) % 3 == 0, \
            ValueError("Encode list after 'dense-sparse-encoding' must be divisible by 3!")
        dense_pos_enc, dense_arch_enc, sparse_attn_enc = [], [], []
        for idx in range(len(encode_list) // 3):
            dense_pos_enc.extend(encode_list[idx * 3])
            dense_arch_enc.extend(encode_list[idx * 3 + 1])
            sparse_attn_enc.extend(encode_list[idx * 3 + 2])
        full_dense_sparse_enc = dense_pos_enc + dense_arch_enc + sparse_attn_enc
        acc = (x['acc'].item() - min_val) / (max_val - min_val) if normalize else x['acc'].item()
        return full_dense_sparse_enc, acc

    def map_fn_dense_and_sparse_flops(self, x):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-sparse-encoding', aggregate=True)
        assert len(encode_list) % 3 == 0, \
            ValueError("Encode list after 'dense-sparse-encoding' must be divisible by 3!")
        dense_pos_enc, dense_arch_enc, sparse_attn_enc = [], [], []
        for idx in range(len(encode_list) // 3):
            dense_pos_enc.extend(encode_list[idx * 3])
            dense_arch_enc.extend(encode_list[idx * 3 + 1])
            sparse_attn_enc.extend(encode_list[idx * 3 + 2])
        full_dense_sparse_enc = dense_pos_enc + dense_arch_enc + sparse_attn_enc
        flops = x['flops (M)'].item() / 1000.
        return full_dense_sparse_enc, flops

    def map_fn_dense(self, x, normalize=True, mean_val=None, std_val=None):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-encoding')
        dense_enc = []
        for item in encode_list:
            dense_enc += item
        acc = self.sigmoid((x['acc'].item() - mean_val) /
                           std_val) if normalize else x['acc'].item()
        return dense_enc, acc

    def map_fn_dense_flops(self, x):
        self.search_space.decode(str_encoding=x['block_args'])
        encode_list = self.search_space.encode(style='dense-encoding')
        dense_enc = []
        for item in encode_list:
            dense_enc += item
        flops = x['flops (M)'].item() / 1000.
        return dense_enc, flops


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
    predictor.save_weights(args.save_ckpt_path)

    predictor.eval()    
    train_dataset_iterator = nas_dataset.iter_map_and_batch(
        split='train', shuffle=False, drop_last_batch=False, batch_size=128)
    test_dataset_iterator = nas_dataset.iter_map_and_batch(
        split='test', shuffle=False, drop_last_batch=False, batch_size=128)
    train_pearson, train_kendall, train_loss = eval_corrleation_with_predictor(train_dataset_iterator, predictor)
    test_pearson, test_kendall, test_loss = eval_corrleation_with_predictor(test_dataset_iterator, predictor)
    print("-----------------------------------------------------------------------")
    print("Evaluating config: lr={:.8f}, wd={:.8f}, margin={:.8f}, ranking_coef={:.8f}".format(
        kwargs['learning_rate'], kwargs['weight_decay'], kwargs['margin'], kwargs['ranking_loss_coef']))
    print("Training Pearson: {:.5f}, Training Kendall Tau: {:.5f}".format(train_pearson, train_kendall))
    print("Testing Pearson: {:.5f}, Testing Kendall Tau: {:.5f}".format(test_pearson, test_kendall))
    print("Training Loss: {:.5f}, Testing Loss: {:.5f}".format(train_loss, test_loss))
    print("-----------------------------------------------------------------------")
    """
    return {
        'train_kendall_tau': train_kendall,
        'train_pearson_rau': train_pearson,
        'test_kendall_tau': test_kendall,
        'test_pearson_rau': test_pearson,
    }
    """
    del predictor
    torch.cuda.empty_cache()
    # Objective should be negative kendall_tau.
    if np.isnan(test_kendall):
        return 99999
    else:
        return test_loss

def main(args):
    # Get the map functions to train the predictor.
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
    
    # Library for map functions.
    map_fn_lib = {
        'onehot': map_fn_records_onehot,
        'ordinal': map_fn_records_ordinal,
        'dense': map_fn_records_dense,
        'dense-sparse': map_fn_records_dense_sparse
    }
    map_fn = map_fn_lib[args.map_fn_name]


    nas_dataset = NASDataSet(
        args.root_dir,
        args.pattern,
        args.record_name,
        map_fn=map_fn,
        cache=True)

    single_run = True
    method = 'de-opt'

    # ModelNet-40 FLOPS hparams: 'learning_rate': 0.08149492357811797, 'weight_decay': 0.0008060759095245629, 'margin': 0.0017515664867573466
    # ModelNet-40 hparams: 'learning_rate': 0.05746790111573563, 'weight_decay': 0.00017620684974022291, 'margin': 0.007120979304889882, 'ranking_loss_coef': 0.15538152707628894
    if single_run:
        hparams = Hparams(args.hparams_json_path)
        _ = train_and_evaluate_per_predictor(
            nas_dataset,
            num_inputs,
            in_dims,
            args,
            learning_rate=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
            margin=hparams.margin,
            ranking_loss_coef=hparams.ranking_loss_coef,
            verbose=True,
            num_epochs=150,
            )
    elif method == 'ng-opt':
        parameterization = ng.p.Instrumentation(
            # a log-distributed scalar between 0.001 and 1.0
            learning_rate=ng.p.Log(lower=0.001, upper=0.1),
            # an integer from 1 to 12
            weight_decay=ng.p.Log(lower=1e-2, upper=1e-5),
            margin=ng.p.Scalar(lower=0.00, upper=0.01),
            ranking_loss_coef=ng.p.Log(lower=0.1, upper=1.0)
        )
        opt_wrapper = NeverGradNGOpt(
            train_and_evaluate_per_predictor,
            parameterization,
            nas_dataset,
            num_inputs,
            in_dims,
            args,
            num_workers=4,
            budget=800,
            log_path=args.opt_log_path
            )
        soln = opt_wrapper.minimize()
        print(soln)
    elif method == 'de-opt':
        parameterization = ng.p.Instrumentation(
            # a log-distributed scalar between 0.001 and 1.0
            learning_rate=ng.p.Log(lower=0.001, upper=0.1),
            # an integer from 1 to 12
            weight_decay=ng.p.Log(lower=1e-2, upper=1e-5),
            margin=ng.p.Scalar(lower=0.00, upper=0.01),
            ranking_loss_coef=ng.p.Log(lower=0.1, upper=1.0)
        )
        opt_wrapper = NeverGradDEOpt(
            train_and_evaluate_per_predictor,
            parameterization,
            nas_dataset,
            num_inputs,
            in_dims,
            args,
            num_workers=4,
            budget=800,
            log_path=args.opt_log_path
        )
        soln = opt_wrapper.minimize()
        print(soln)

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
