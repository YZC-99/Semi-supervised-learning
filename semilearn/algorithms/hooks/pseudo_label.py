# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import smooth_targets

class PseudoLabelingHook(Hook):
    """
    Pseudo Labeling Hook
    """
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def gen_ulb_targets(self, 
                        algorithm, 
                        logits, 
                        use_hard_label=True, 
                        T=1.0,
                        softmax=True, # whether to compute softmax for logits, input must be logits
                        label_smoothing=0.0,
                        multi_label=True
                        ):
        
        """
        generate pseudo-labels from logits/probs

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """

        logits = logits.detach()
        if multi_label:
            # multi-label classification
            if use_hard_label:
                # return hard label directly
                # pseudo_label = torch.sigmoid(logits)
                pseudo_label = torch.where(logits > 0.5, torch.ones_like(logits), torch.zeros_like(logits))
                return pseudo_label
            else:
                # return soft label
                # pseudo_label = torch.sigmoid(logits)
                return logits
        else:
            if use_hard_label:
                # return hard label directly
                pseudo_label = torch.argmax(logits, dim=-1)
                if label_smoothing:
                    pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
                return pseudo_label

            # return soft label
            if softmax:
                # pseudo_label = torch.softmax(logits / T, dim=-1)
                pseudo_label = algorithm.compute_prob(logits / T)
            else:
                # inputs logits converted to probabilities already
                pseudo_label = logits
            return pseudo_label
        