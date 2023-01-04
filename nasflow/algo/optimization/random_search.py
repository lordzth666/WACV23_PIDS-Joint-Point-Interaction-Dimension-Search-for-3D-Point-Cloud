import numpy as np
from nasflow.io_utils.base_parser import parse_args_from_kwargs
from nasflow.algo.optimization.optimizer import BaseOptimizer
from nasflow.algo.optimization.sampler import BaseSampler
from nasflow.algo.optimization.decoder import BaseDecoder

class RandomSearch(BaseOptimizer):
    def __init__(
            self,
            eval_fn,
            sampler: BaseSampler,
            decoder: BaseDecoder,
            *args,
            **kwargs):
        super(RandomSearch, self).__init__(eval_fn, sampler, decoder, *args, **kwargs)

    def minimize(self, **kwargs):
        idx = 0
        budget = parse_args_from_kwargs(kwargs, 'budget', 100)
        # Evaluate batch size. If >1, will concatenate all samples in a large sample
        # to perform evaluation. This speeds up efficiency in predictor-based NAS.
        # Note: the eval_fn should also be able to predict batch data if eval_batch_size > 1.
        eval_batch_size = parse_args_from_kwargs(kwargs, 'eval_batch_size', 1)
        # Whether return history or not.
        return_history = parse_args_from_kwargs(
            kwargs, 'return_history', False)
        all_samples = []
        all_objectives = []
        while idx < budget:
            sampled_candidates = [self.sample()
                                  for _ in range(eval_batch_size)]
            decoded_candidates = [self.decode(x) for x in sampled_candidates]
            objectives = self.eval_fn(decoded_candidates, *self.args)
            all_samples.extend(sampled_candidates)
            all_objectives.extend(objectives)
            idx += eval_batch_size

        best_sample_idx = np.argmin(all_objectives)
        print("Best objective: {}".format(all_objectives[best_sample_idx]))
        if not return_history:
            return all_samples[best_sample_idx], all_objectives[best_sample_idx]
        else:
            return all_samples[best_sample_idx], all_objectives[best_sample_idx], all_samples
