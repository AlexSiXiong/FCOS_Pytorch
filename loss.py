import nntplib
import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, pred, target):
        pred = pred.view(-1, 1)
        target = pred.view(-1, 1)

        pred = torch.cat((1-pred, pred), dim=1)

        class_mask = torch.zeros(pred.shape[0]. pred.shape[1]).cuda()

        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(1).view(-1, 1)
        probs = torch.clamp(probs, 1e-8, 1.-1e-8)

        log_p = probs.log()

        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1-self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha

        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
        batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        
        return loss