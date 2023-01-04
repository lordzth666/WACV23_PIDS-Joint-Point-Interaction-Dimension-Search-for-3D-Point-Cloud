import numpy as np
import random
import torch.multiprocessing as mp
from tqdm import tqdm

from nasflow.io_utils.base_parser import parse_args_from_kwargs
from nasflow.algo.optimization.optimizer import BaseOptimizer
from nasflow.algo.optimization.sampler import BaseSampler
from nasflow.algo.optimization.decoder import BaseDecoder
class RegularizedEvolution(BaseOptimizer):
    def __init__(
            self,
            eval_fn, sampler: BaseSampler,
            decoder: BaseDecoder,
            search_space,
            *args,
            **kwargs):
        super(RegularizedEvolution, self).__init__(
            eval_fn, sampler, decoder, *args, **kwargs)

        # Parse the args
        self.args = args
        # This is also equivalent to the 'aging population size'.
        self.init_population_size = parse_args_from_kwargs(
            kwargs, "init_population", 200)
        self.sample_ratio = parse_args_from_kwargs(
            kwargs, "sample_ratio", 0.75)
        self.mutation_size = parse_args_from_kwargs(
            kwargs, "mutation_size", 150)
        self.crossover_size = parse_args_from_kwargs(
            kwargs, "crossover_size", 50)
        self.exploitation_size = parse_args_from_kwargs(
            kwargs, 'exploitation_size', 50
        )
        # Percetange of top sampled arch for crossover.
        self.crossover_top_ratio = parse_args_from_kwargs(
            kwargs, 'crossover_top_ratio', 0.1
        )
        self.gene_encode_type = parse_args_from_kwargs(
            kwargs, 'gene_encode_type', 'ordinal'
        )
        self.init_generations_no_aging = parse_args_from_kwargs(
            kwargs, 'init_generations_no_aging', 0
        )
        self.iter = parse_args_from_kwargs(
            kwargs, "n_generations", 10
        )
        self.grow_size = self.mutation_size + self.crossover_size + self.exploitation_size

        self.gene_choice = search_space
        self.gene_len = len(self.gene_choice)

    @staticmethod
    def hash_block_args(block_args):
        return hash("".join(block_args))

    def minimize(self, **kwargs):
        return_history = parse_args_from_kwargs(
            kwargs, 'return_history', False)
        
        popu = [self.sample() for _ in range(self.init_population_size)]
        popu_score = self.score(popu)

        # Now, put only the top 'self.grow_size' samples.
        sorted_args = np.argsort(popu_score)[:self.grow_size]
        popu = np.asarray(popu)[sorted_args].tolist()

        best_popu = []
        best_popu_scores = []

        #all_scores_list = []

        for i in range(self.iter):
            num_mutations = (self.iter-i) // 80 + 1
            ## scores and sort
            sample_size = int(len(popu) * self.sample_ratio)
            sampled_popu = np.random.choice(popu, sample_size, replace=False)
            sampled_popu_scores = self.score(sampled_popu)
            scores_ind = np.argsort(np.array(sampled_popu_scores))
            print("Start Iteration {}:".format(i))
            print("Iteration {}, Lowest loss: {}".format(i, min(sampled_popu_scores)))
            # Parent set to lowest candidate in population.
            print(sampled_popu[scores_ind[0]]['block_args'])
            for topk in range(2):
                best_popu_scores.append(sampled_popu_scores[scores_ind[topk]])
                best_popu.append(sampled_popu[scores_ind[topk]])

            print("| Best gene history {}: {}".format(i, sampled_popu_scores[scores_ind[0]]))

            n_crossover_top_archs = int(self.crossover_top_ratio * sample_size)
            ## mutate_generation
            mutation_popu = [self.mutate(
                sampled_popu[scores_ind[0]], num_mutations) for _ in range(self.mutation_size)]
            ## Crossover generation.
            crossover_popu = [
                self.crossover([
                    np.random.choice(sampled_popu[scores_ind[:n_crossover_top_archs]]), \
                    np.random.choice(sampled_popu[scores_ind[:n_crossover_top_archs]])
                    ]) \
                for _ in range(self.crossover_size)
            ]
            # Exploitation
            new_expo_popu = [self.sample() for _ in range(self.exploitation_size)]

            popu = popu + mutation_popu + crossover_popu + new_expo_popu
            if i >= self.init_generations_no_aging:
                # Remove dead.
                popu = popu[self.grow_size:]

        best_gene_history = {'best_popu': best_popu,
                             'best_popu_scores': best_popu_scores}
        if not return_history:
            return best_gene_history['best_popu'], best_gene_history['best_popu_scores']
        else:
            return best_gene_history['best_popu'], best_gene_history['best_popu_scores'], best_gene_history

    def score(self, population):
        decoded_popu = [self.decode(x) for x in population]
        #print(decoded_popu)
        popu_scores = self.eval_fn(decoded_popu, *self.args)
        return np.array(popu_scores)

    def mutate(self, gene, num_mutations=1):
        self.gene_choice.decode(str_encoding=gene['block_args'])
        for _ in range(num_mutations):
            idx = np.random.choice(self.gene_len)
            self.gene_choice.__seed_one__(idx)
        encode_list = self.gene_choice.encode(style=self.gene_encode_type, aggregate=True)
        block_args = self.gene_choice.str_encode()
        return {'block_args': block_args, 'encoding': encode_list}

    def crossover(self, parents, crossover_prob=0.2):
        #print(parents)
        new_gene_args = []
        for i in range(self.gene_len):
            if np.random.uniform() < crossover_prob:
                new_gene_args.append(parents[0]['block_args'][i])
            else:
                new_gene_args.append(parents[1]['block_args'][i])
        self.gene_choice.decode(str_encoding=new_gene_args)
        encode_list = self.gene_choice.encode(style=self.gene_encode_type, aggregate=True)
        block_args = self.gene_choice.str_encode()
        return {'block_args': block_args, 'encoding': encode_list}

    def satisfy_constraints(self, gene):
        #Not implemented
        return True
