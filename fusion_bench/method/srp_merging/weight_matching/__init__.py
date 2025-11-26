"""
Weight Matching Module for Fastfood Model Merging

This module implements weight-based permutation matching (Git Re-Basin approach)
as an optional preprocessing step for model merging. It aligns neurons across
models before merging to improve fusion quality.

Key Components:
- weight_matching: Main algorithm for finding optimal permutations
- get_permutation_spec: Analyzes model architecture to determine permutable axes
- apply_perm: Applies permutations to model parameters

Usage:
    from fusion_bench.method.fastfood_merging.weight_matching import (
        weight_matching,
        get_permutation_spec,
        apply_perm,
    )
"""

from .weight_matching import weight_matching, cross_features_inner_product
from .core import (
    get_permutation_spec,
    apply_perm,
    make_identity_perm,
    PermutationSpec,
    Permutation,
    Axis,
)

__all__ = [
    'weight_matching',
    'cross_features_inner_product',
    'get_permutation_spec',
    'apply_perm',
    'make_identity_perm',
    'PermutationSpec',
    'Permutation',
    'Axis',
]
