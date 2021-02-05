import torch
import torch.nn as nn
import torch.nn.functional as F


class DCENetLoss(nn.Module):
    def __init__(self, config):
        super(DCENetLoss, self).__init__()
        self.beta = config['beta']
        self.pred_seq = config['pred_seq']
    
    def forward(self, mu, log_var, y_pred, y_true):
        kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
        mse_loss = self.pred_seq * F.mse_loss(y_pred, y_true)

        loss = mse_loss * self.beta  + kl_loss * (1 - self.beta)
        return loss
