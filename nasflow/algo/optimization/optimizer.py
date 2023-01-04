from nasflow.io_utils.base_parser import parse_args_from_kwargs
from nasflow.algo.optimization.sampler import BaseSampler
from nasflow.algo.optimization.decoder import BaseDecoder

class BaseOptimizer:
    """
    A base optimizer to carry optimization over multiple problems.
    Problems must have the format of eval_fn(candidate, **kwargs) format.
    """
    def __init__(
            self,
            eval_fn,
            sampler: BaseSampler,
            decoder: BaseDecoder,
            *args,
            **kwargs):
        self.eval_fn = eval_fn
        self.args = args
        self.kwargs = kwargs
        self.sampler = sampler
        self.decoder = decoder

    def minimize(self, **kwargs):
        raise NotImplementedError("Base class method minimize() is not implemented!")

    def sample(self, **kwargs):
        return self.sampler.sample()

    def decode(self, x, **kwargs):
        return self.decoder.decode(x)
