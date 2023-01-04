import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nasflow.optim.ema import EMA

from pids_core.models.loss_utils import lovasz_softmax_flat

_DEVICE_CFG = torch.cuda.is_available()
def get_l2_loss(model, weight_decay, regularize_depthwise=True):
    # assert isinstance(model, torch.nn.Module)
    l2_loss = None
    model_reg = model.model if isinstance(model, EMA) else model
    for n, m in model_reg.named_parameters():
        if len(m.size()) != 1:
            if "dw_weights" in n and not regularize_depthwise:
                if l2_loss is None:
                    l2_loss = torch.square(torch.norm(m, 2)) / m.size()[-1]
                else:
                    l2_loss += torch.square(torch.norm(m, 2)) / m.size()[-1]
            else:
                if l2_loss is None:
                    l2_loss = torch.square(torch.norm(m, 2))
                else:
                    l2_loss += torch.square(torch.norm(m, 2))

    if _DEVICE_CFG:
        l2_loss = torch.mul(l2_loss, torch.tensor(weight_decay).cuda())
    else:
        l2_loss = torch.mul(l2_loss, torch.tensor(weight_decay))

    return l2_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        indices_non_negative = (target >= 0)
        x_ = x[indices_non_negative]
        target_ = target[indices_non_negative]
        logprobs = F.log_softmax(x_, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target_.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    def __repr__(self):
        return "LabelSmoothingCrossEntropy(smoothing={:.2f})".format(self.smoothing)


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, pred, target):
        # Remove the loss with ignored labels.
        pred_ = pred[indices_non_negative]
        target_ = target[indices_non_negative]


