import torch

probs_ulb_fc = torch.tensor([[0.1, 0.6, 0.7], [0.3, 0.4, 0.3], [0.5, 0.6, 0.2]])
probs_ulb_fc_add_positive_means = torch.tensor([[0.1, 0.6, 0.7], [0.3, 0.8, 0.6], [0.5, 0.3, 0.2]])

print(probs_ulb_fc > 0.5)
print(probs_ulb_fc_add_positive_means > 0.5)
print((probs_ulb_fc > 0.5) != (probs_ulb_fc_add_positive_means > 0.5))