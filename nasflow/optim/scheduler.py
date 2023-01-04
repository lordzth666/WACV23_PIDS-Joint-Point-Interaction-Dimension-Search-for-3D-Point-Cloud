import torch
from nasflow.optim.cosine_with_warmup import CosineAnnealingLRWarmup

def get_lr_scheduler(optimizer, lr_scheduler='cosine', num_epochs=90):
    if lr_scheduler == "none":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [num_epochs*10, num_epochs*20])
    elif lr_scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [num_epochs//3, num_epochs*2//3], gamma=0.1)
    elif lr_scheduler == "cosine-pred":
        lr_scheduler = CosineAnnealingLRWarmup(
            optimizer, T_max=num_epochs, last_epoch=-1, T_up=int(num_epochs * 0.25))
    elif lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLRWarmup(
            optimizer, T_max=num_epochs, last_epoch=-1, T_up=5)
    elif lr_scheduler == "cosine-no-warmup":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, last_epoch=-1)
    else:
        raise NotImplementedError
    return lr_scheduler
