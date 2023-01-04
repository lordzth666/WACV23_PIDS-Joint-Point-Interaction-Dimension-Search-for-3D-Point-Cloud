import numpy as np
import nevergrad as ng

def create_predictor_list_with_args(*args):
    all_lists = []
    for arg in args:
        if arg == -1:
            return all_lists
        else:
            return arg

def create_searchable_params_for_orignal_archs():
    parameterization = ng.p.Instrumentation(
        # a log-distributed scalar between 0.001 and 1.0
        learning_rate=ng.p.Log(lower=0.005, upper=0.5),
        # an integer from 1 to 12
        weight_decay=ng.p.Log(lower=1e-6, upper=1e-3),
        margin=ng.p.Scalar(lower=0.00, upper=0.01),
        ranking_loss_coef=ng.p.Log(lower=0.1, upper=1.0)
    )
    