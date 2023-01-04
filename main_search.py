import os
import sys
sys.path.append(os.getcwd())
from math import ceil
import argparse
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from nasflow.io_utils.base_io import maybe_write_json_file
from nasflow.algo.optimization.random_search import RandomSearch
from nasflow.algo.optimization.regularized_ea import RegularizedEvolution

from nasflow.algo.optimization.policy_rl_beta import PolicyBasedRLSearch
from nasflow.algo.optimization.rl_env import RLEnv
from nasflow.algo.optimization.rl_policy import default_policy_fn

from pids_search_space.sampler import (
    PIDS_Cls_Sampler,
    PIDS_Seg_Sampler
)
from pids_search_space.decoder import (
    PIDSVanillaDecoder,
    PIDSDenseSparseDecoder
)
from predictor.predictor_model_zoo import get_predictor
from pids_search_space.pids_architect import (
    PIDS_Space_Cls,
    PIDS_Space_Seg,
    build_pids_comb_search_space_from_genotype
)
from pids_search_space.genotype import (
    PIDS_cls_search_space_cfgs,
    PIDS_seg_search_space_cfgs,
)

def main(args):
    np.random.seed(args.seed)
    if args.task in ["modelnet40"]:
        search_space_cfgs = PIDS_cls_search_space_cfgs()
        pids_search_spaces = build_pids_comb_search_space_from_genotype(search_space_cfgs)
        search_space = PIDS_Space_Cls(pids_search_spaces)
        sampler = PIDS_Cls_Sampler(search_space, encode_style=args.encode_style)
        if args.encode_style == 'dense-sparse-encoding':
            decoder = PIDSDenseSparseDecoder()
        else:
            decoder = PIDSVanillaDecoder()
        # Initialize predictors and get evaluation function
        num_inputs = 7
        in_dims = 7 * 64 * num_inputs
    elif args.task in ['semantickitti', 's3dis']:
        search_space_cnn_cfg, search_space_fcn_cfg = PIDS_seg_search_space_cfgs()
        all_search_cfg = search_space_cnn_cfg + search_space_fcn_cfg
        pids_search_spaces = build_pids_comb_search_space_from_genotype(all_search_cfg)
        search_space = PIDS_Space_Seg(pids_search_spaces,
                                            num_cnn_search_spaces=len(search_space_cnn_cfg),
                                            num_fcn_search_spaces=len(search_space_fcn_cfg),
                                            )
        sampler = PIDS_Seg_Sampler(search_space, encode_style=args.encode_style)
        if args.encode_style == 'dense-sparse-encoding':
            decoder = PIDSDenseSparseDecoder()
        else:
            decoder = PIDSVanillaDecoder()
        # Initialize predictors and get evaluation function
        num_inputs = 11
        in_dims = 7 * 64 * num_inputs

    # Load accuracy predictor.
    acc_predictor = get_predictor(
        args.predictor_arch,
        None,
        in_dims,
        0,
        num_inputs)
    acc_predictor.load_weights(args.acc_predictor_ckpt_path, None)
    acc_predictor.eval()
    # Load flops predictor
    flops_predictor = get_predictor(
        args.predictor_arch,
        None,
        in_dims,
        0,
        num_inputs)
    flops_predictor.load_weights(args.flops_predictor_ckpt_path, None)
    flops_predictor.eval()
    eval_fn = lambda x: 0 - acc_predictor.predict(torch.Tensor(x)).cpu().flatten().numpy() \
        + np.log(flops_predictor.predict(torch.Tensor(x)).cpu().flatten().numpy()) * args.flops_penalty_coef
    
    if args.method == "random":
        searcher = RandomSearch(eval_fn, sampler, decoder)
        assert args.random_search_budget % args.random_search_partition_size == 0, \
            "Partition size must be a divisor of search budget!"
        num_partitions = args.random_search_budget // args.random_search_partition_size
        all_best_archs, all_best_arch_objs = [], []
        minimize_fn = lambda idx: searcher.minimize(
            return_history=False,
            eval_batch_size=args.random_search_eval_batch_size,
            budget=args.random_search_partition_size)
        all_vars = thread_map(
            minimize_fn, range(num_partitions), max_workers=5)
        all_best_archs = [item[0] for item in all_vars]
        all_best_arch_objs = [item[1] for item in all_vars]
    elif args.method == "regularized-ea":
        searcher = RegularizedEvolution(
            eval_fn, sampler, decoder, search_space,
            gene_encode_type=args.encode_style,
            n_generations=args.ea_n_generation,
            init_population=args.ea_init_population,
            sample_ratio=args.ea_sample_ratio,
            mutation_size=args.ea_mutation_size,
            crossover_size=args.ea_crossover_size,
            exploitation_size=args.ea_exploitation_size,
            )
        all_best_archs, all_best_arch_objs = searcher.minimize(
            return_history=False)
    elif args.method == "policy-rl":
        encoding_space = []        
        for cfg in search_space_cfgs:
            for key in cfg.keys():
                encoding_space.append(len(cfg[key]))
        #print(encoding_space)
        env = RLEnv(eval_fn, encoding_space)
        #multi_env = BaseVectorEnv([env for _ in range(concurrency)], MultiThreadEnvWorker)
        policy_fn = default_policy_fn(env)
        searcher = PolicyBasedRLSearch(eval_fn, sampler, decoder, search_space, env, policy_fn)
        _, _, sample_history = searcher.minimize(return_history=True)
        all_best_archs_encodings = sample_history['best_encodings']
        all_best_arch_objs = sample_history['best_rewards']
        all_best_archs = []
        for n in range(len(all_best_archs_encodings)):
            block_arg = []
            ind = 0
            for cfg in search_space_cfgs:
                cfg_arg = ''
                for key in cfg.keys():
                    cfg_arg = cfg_arg+str(key)+str(cfg[key][all_best_archs_encodings[n][ind]])+'_'
                cfg_arg.strip('_')
                block_arg.append(cfg_arg)
                ind += 1
            all_best_archs.append({'block_args':block_arg,'rewards':all_best_arch_objs[n]})

    # Finally, final top_k archs.
    all_best_archs = np.asarray(all_best_archs)
    all_best_arch_objs = np.asarray(all_best_arch_objs)
    # Remove duplicate ones.
    _, unique_indices = np.unique(all_best_arch_objs, return_index=True)
    all_best_archs = all_best_archs[unique_indices]
    all_best_arch_objs = all_best_arch_objs[unique_indices]
    sorted_indices = np.argsort(all_best_arch_objs)
    all_best_archs = all_best_archs[sorted_indices]
    all_best_arch_objs = all_best_arch_objs[sorted_indices]
    json_dump_list = []
    for idx in range(args.top_k):
        json_dump_list.append(
            {
                'block_args': \
                    all_best_archs[idx]['block_args'],
                'score' if args.flops_penalty_coef != 0 else 'acc': \
                    all_best_arch_objs[idx].item(),
            }
        )
    # Mkdir to json path first.
    json_path_dir = args.dump_json_path.split("/")[:-1]
    json_path_dir = "/".join(json_path_dir)
    print(json_path_dir)
    os.makedirs(json_path_dir, exist_ok=True)
    maybe_write_json_file(json_dump_list, args.dump_json_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=233,
        help="Random Seed."
    )
    parser.add_argument(
        "--predictor_arch", type=str, default="dense-sparse-nn",
        help="Architecture of predictor.")
    parser.add_argument(
        "--task", type=str, default="modelnet40",
        help="Task that predictor operates on", choices=['modelnet40', 'semantickitti']
    )
    parser.add_argument(
        "--encode_style", type=str, default='dense-sparse-encoding',
        help="Encoding style.",
        choices=['one-hot', 'ordinal', 'dense-sparse-encoding'])
    parser.add_argument(
        "--acc_predictor_ckpt_path", type=str, default=None,
        help="Ckpt path for an accuracy predictor.")
    parser.add_argument(
        "--flops_predictor_ckpt_path", type=str, default=None,
        help="Ckpt path for a flops predictor.")
    parser.add_argument(
        "--flops_penalty_coef", type=float, default=0.0,
        help="flops coefficient."
    )
    parser.add_argument(
        '--method', type=str, default="random",
        help="Search method", choices=['random', "regularized-ea", "policy-rl"]
    )
    parser.add_argument(
        "--random_search_budget", type=int, default=100000,
        help="Budget for random search."
    )
    parser.add_argument(
        "--random_search_partition_size", type=int, default=10000,
        help="Partition size to run random search."
    )
    parser.add_argument("--top_k", type=int, default=10,
        help="Top-k arch to keep.")
    parser.add_argument(
        "--random_search_eval_batch_size", type=int, default=500,
        help="Eval batch size for random search."
    )
    # EA configurations.
    parser.add_argument(
        "--ea_n_generation", type=int, default=360,
        help="Number of generations for EA."
    )
    parser.add_argument(
        "--ea_init_population", type=int, default=200,
        help="Init population for EA."
    )
    parser.add_argument(
        "--ea_sample_ratio", type=int, default=0.75,
        help="Number of parent archs selected for mutation/cross-over."
    )
    parser.add_argument(
        "--ea_mutation_size", type=int, default=200,
        help="Mutation size for EA."
    )
    parser.add_argument(
        "--ea_crossover_size", type=int, default=0,
        help="Crossover size for EA."
    )
    parser.add_argument(
        "--ea_exploitation_size", type=int, default=0,
        help="EA exploitation size."
    )
    # Json Dump configuration.
    parser.add_argument(
        "--dump_json_path", type=str, default=None,
        help="File path to dump the json file."
     )
    global_args = parser.parse_args()
    main(global_args)