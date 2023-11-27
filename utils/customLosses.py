import torch
import torch.nn as nn


class L2CELoss(nn.Module):
    def __init__(self, epsilon, l2_alpha):
        super(L2CELoss, self).__init__()
        self.epsilon = epsilon
        self.l2_alpha = l2_alpha

    def forward(self, y_pred, y_true, pest):
        ce_loss = nn.functional.cross_entropy(y_pred, y_true)
        l2_norm = torch.norm(torch.abs(pest) - self.epsilon, p=2)
        loss = ce_loss + self.l2_alpha * l2_norm
        return loss
