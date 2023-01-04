from math import sqrt

import torch
import torch.nn as nn

from nasflow.optim.ema import EMA

_DEVICE_CFG = torch.cuda.is_available()

def get_l2_loss(model, weight_decay, regularize_depthwise=True):
    assert isinstance(model, torch.nn.Module)
    if isinstance(model, EMA):
        reg_model = model.model
    else:
        reg_model = model
    l2_loss = None
    for n, m in reg_model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(m, nn.Conv2d):
                if m.groups == m.out_channels and (not regularize_depthwise):
                    # Put much fewer regularization on depthwise.
                    # This seems to be even better than a rough 'regularize_depth' option.
                    if l2_loss is None:
                        l2_loss = sqrt(1. / m.groups) * \
                            torch.square(torch.norm(m.weight, 2))
                    else:
                        l2_loss += sqrt(1. / m.groups) * \
                            torch.square(torch.norm(m.weight, 2))
                else:
                    if l2_loss is None:
                        l2_loss = torch.square(torch.norm(m.weight, 2))
                    else:
                        l2_loss += torch.square(torch.norm(m.weight, 2))
            else:
                if l2_loss is None:
                    l2_loss = torch.square(torch.norm(m.weight, 2))
                else:
                    l2_loss += torch.square(torch.norm(m.weight, 2))
    if _DEVICE_CFG:
        l2_loss = torch.mul(l2_loss, torch.tensor(weight_decay).cuda())
    else:
        l2_loss = torch.mul(l2_loss, torch.tensor(weight_decay))

    return l2_loss
