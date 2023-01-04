import torch
import torch.nn as nn

class MarginRankingLoss(nn.Module):
    def __init__(self,
                 margin: float = 0.0,
                 aggregation: str = 'reduce_mean'):
        super(MarginRankingLoss, self).__init__()
        assert margin >= 0, "Margin should be non-negative, but {} found!".format(margin)
        assert aggregation in ['reduce_mean', 'reduce_sum'], \
                NotImplementedError("Aggregation {} not implemented!")
        self.margin = margin
        self.aggregation = aggregation

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                labels: torch.Tensor):
        # GT: x>y -> 1
        losses = torch.nn.functional.relu(torch.multiply(-labels, (x - y)) - self.margin, True)
        if self.aggregation == 'reduce_mean':
            return losses.mean()
        elif self.aggregation == 'reduce_sum':
            return losses.sum()
        else:
            raise NotImplementedError("Aggregation method {} should be implemented!".format(self.aggregation))
