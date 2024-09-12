
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F

from .cross_entropy import ce_loss
from torch.nn.functional import binary_cross_entropy_with_logits



def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    # assert name in ['ce','bce', 'mse', 'kl','sim_un']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    elif name == 'kl':
        loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
        loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
    elif name == 'ce':
        loss = ce_loss(logits, targets, reduction='none')
    elif name == 'bce':
        loss = binary_cross_entropy_with_logits(logits, targets, reduction='none')
    elif name == 'sim_un':
        loss = (torch.sum(-F.logsigmoid(logits) * targets.detach(), dim=1) * (torch.max(F.sigmoid(logits), dim=1).values > 0.5)).mean()

    if mask is not None and name != 'kl':
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)