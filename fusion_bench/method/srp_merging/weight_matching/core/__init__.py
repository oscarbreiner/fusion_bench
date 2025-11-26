"""
Core utilities and data structures for weight matching in model merging.

This module provides the fundamental data structures and utilities
for permutation-based model merging, including permutation specifications,
application functions, and model analysis tools.
"""

from .utils import (
    Axis,
    PermutationGroup,
    PermutationSpec,
    Permutation,
    apply_perm,
    make_identity_perm,
    make_random_perm,
    invert_perm,
    count_linear_flops,
)

from .solvers import scipy_solve_lsa

from .compiler import get_permutation_spec

__all__ = [
    'Axis',
    'PermutationGroup',
    'PermutationSpec',
    'Permutation',
    'apply_perm',
    'make_identity_perm',
    'make_random_perm',
    'invert_perm',
    'count_linear_flops',
    'scipy_solve_lsa',
    'get_permutation_spec',
]