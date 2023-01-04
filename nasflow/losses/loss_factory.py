import torch

from nasflow.losses.ranking_loss import MarginRankingLoss

_all_losses_lib = {
    'mse-loss': torch.nn.MSELoss,
    'mae-loss': torch.nn.L1Loss,
    'smoothed-l1-loss': torch.nn.SmoothL1Loss,
    'smoothed-l1-loss-nnpred': lambda: torch.nn.SmoothL1Loss(beta=0.2),
    'margin-ranking-loss': MarginRankingLoss,
}

def get_loss_fn_from_lib(loss_fn_name):
    if loss_fn_name is None:
        print("Warning: No loss fn specified.")
        return lambda *args, **kwargs: None
    if loss_fn_name not in _all_losses_lib:
        raise KeyError("Loss function name {} not supported! Supported losses are: {}".format(
            loss_fn_name, _all_losses_lib.keys()
        ))
    return _all_losses_lib[loss_fn_name]
