import nevergrad as ng
from concurrent import futures

from nasflow.io_utils.base_parser import parse_args_from_kwargs
class NeverGradBaseOpt:
    def __init__(self, eval_fn, parametrization, *args, **kwargs):
        self.eval_fn = eval_fn
        self.parametrization = parametrization
        self.args = args
        # This kwargs will be overriden by params in nevergrad optimizer.
        # A check in kwargs is needed.
        self.kwargs = kwargs
        self.eval_fn_wrapper = lambda **kwargs: self.eval_fn(*(self.args), **kwargs)
        self.optimizer = None

    def minimize(self):
        raise BaseException("This is a base class that has not yet been implemented!")

class NeverGradNGOpt(NeverGradBaseOpt):
    def __init__(self, eval_fn, parameterization, *args, **kwargs):
        super(NeverGradNGOpt, self).__init__(eval_fn, parameterization, *args, **kwargs)
        self.num_workers = parse_args_from_kwargs(kwargs, 'num_workers', 4)
        self.budget = parse_args_from_kwargs(kwargs, 'budget', 100)
        self.log_path = parse_args_from_kwargs(kwargs, 'log_path', None)
        self.optimizer = ng.optimizers.NGOpt(
            parametrization=self.parametrization, budget=self.budget, num_workers=self.num_workers)
        if self.log_path is not None:
            logger = ng.callbacks.ParametersLogger(self.log_path)
            self.optimizer.register_callback("tell", logger)

    def minimize(self):
        with futures.ThreadPoolExecutor(max_workers=self.optimizer.num_workers) as executor:
            _ = self.optimizer.minimize(self.eval_fn_wrapper, executor=executor, batch_mode=False)
        recommendation = self.optimizer.provide_recommendation()
        return recommendation

class NeverGradDEOpt(NeverGradBaseOpt):
    def __init__(self, eval_fn, parameterization, *args, **kwargs):
        super(NeverGradDEOpt, self).__init__(eval_fn, parameterization, *args, **kwargs)
        self.num_workers = parse_args_from_kwargs(kwargs, 'num_workers', 4)
        self.budget = parse_args_from_kwargs(kwargs, 'budget', 100)
        self.log_path = parse_args_from_kwargs(kwargs, 'log_path', None)
        self.optimizer = ng.optimizers.TwoPointsDE(
            parametrization=self.parametrization, budget=self.budget, num_workers=self.num_workers)
        if self.log_path is not None:
            logger = ng.callbacks.ParametersLogger(self.log_path)
            self.optimizer.register_callback("tell", logger)

    def minimize(self):
        with futures.ThreadPoolExecutor(max_workers=self.optimizer.num_workers) as executor:
            _ = self.optimizer.minimize(self.eval_fn_wrapper, executor=executor, batch_mode=False)
        recommendation = self.optimizer.provide_recommendation()
        return recommendation
