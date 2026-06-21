import torch
import torch.nn.functional as F

"""
Loss functions for sparse electron counting data.

Available:
    mse                  — Standard MSE as in UDVD. Unbiased but poor gradient for sparse data.
    log_mse              — MSE in log(1+x) space. Equalizes dynamic range.
    poisson_nll          — Poisson NLL. Correct for counting noise.
                           Note: systematic upward bias (see Poisson NLL bias section).
    weighted_poisson_nll — Poisson NLL weighted by inverse PACBED.
                           Boosts high-angle signal contribution.
"""

LOSS_REGISTRY = {}

def register_loss(name):
    def register_fn(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return register_fn

@register_loss("mse")
def mse_loss(outputs, targets):
    return F.mse_loss(outputs, targets)

@register_loss("log_mse")
def log_mse_loss(outputs, targets):
    return F.mse_loss(torch.log1p(outputs), torch.log1p(targets))

@register_loss("poisson_nll")
def poisson_nll_loss(outputs, targets):
    return torch.mean(outputs - targets * torch.log(outputs + 1e-8))

class WeightedPoissonNLLLoss:
    """Poisson NLL with PACBED-derived spatial weighting."""
    def __init__(self, data, device):
        pacbed = data.mean(axis=(0, 1))
        weight = 1.0 / (pacbed + pacbed.mean() * 0.1)
        weight = weight / weight.mean()
        self.weight_map = torch.tensor(weight, dtype=torch.float32).unsqueeze(0).to(device)

    def __call__(self, outputs, targets):
        nll = outputs - targets * torch.log(outputs + 1e-8)
        return torch.mean(self.weight_map * nll)

def build_loss_fn(name, data=None, device=None):
    if name == 'weighted_poisson_nll':
        return WeightedPoissonNLLLoss(data, device)
    if name in LOSS_REGISTRY:
        return LOSS_REGISTRY[name]
    raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys()) + ['weighted_poisson_nll']}")
