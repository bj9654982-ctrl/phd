"""Reconstructible reference-inspired insertion; not official NLCNet."""
from torch import nn
from src.reference_nlfe import ReferenceNLFE


def attach(model, variant):
    if variant == 'A':
        return model
    if variant not in ('L', 'N'):
        raise ValueError(variant)
    if hasattr(model.encoder_level3, 'reference_branch'):
        raise ValueError('already attached')
    window, pool = (4, 1) if variant == 'L' else (32, 4)
    branch = ReferenceNLFE(192, heads=4, window=window, pool=pool)
    nn.init.zeros_(branch.proj.weight); nn.init.zeros_(branch.proj.bias)
    nn.init.zeros_(branch.ffn[-1].weight); nn.init.zeros_(branch.ffn[-1].bias)
    # Appending preserves all original numerical state_dict keys.
    model.encoder_level3.add_module('reference_branch', branch.to(next(model.parameters()).device))
    return model
