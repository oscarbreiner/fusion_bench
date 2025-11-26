import torch

from collections.abc import Sequence
from copy import copy, deepcopy
from typing import Union

from .core import (
    Permutation,
    PermutationSpec,
    apply_perm,
    make_identity_perm,
    scipy_solve_lsa,
)

# Type alias for state dictionary
StateDict = dict[str, torch.Tensor]


def cross_features_inner_product(x, y, a: int):
    """
    Compute the inner product between features across a specific axis.
    
    This matches the original PLeaS implementation for weight-based permutation matching.
    
    Args:
        x (torch.Tensor): First tensor
        y (torch.Tensor): Second tensor
        a (int): Axis along which to compute inner products
        
    Returns:
        torch.Tensor: Matrix of inner products with shape (x.shape[a], y.shape[a])
    """
    x = torch.movedim(x, a, 0).reshape(x.shape[a], -1)
    y = torch.movedim(y, a, 0).reshape(y.shape[a], -1)
    return x @ y.T


def weight_matching(
    spec: PermutationSpec,
    state_as: Union[StateDict, Sequence[StateDict]],
    state_bs: Union[StateDict, Sequence[StateDict]],
    max_iter=100,
    init_perm=None,
    inplace=False,
    skip_suffixes=("running_mean", "running_var"),
    skip_missing=True,
    lsa_solver=scipy_solve_lsa,
    cross_weights=cross_features_inner_product,
    verbose=True,
    seed=0,
    return_costs=False,
) -> Permutation:
    if isinstance(state_as, dict):
        state_as = [state_as]
    if isinstance(state_bs, dict):
        state_bs = [state_bs]

    assert len(state_as) == len(state_bs)

    if not inplace:
        state_bs = [copy(state_b) for state_b in state_bs]

    perm = make_identity_perm(spec) if init_perm is None else deepcopy(init_perm)
    if init_perm is not None:
        for state_b in state_bs:
            apply_perm(init_perm, spec, state_b, inplace=True)

    perm_names = list(perm.keys())
    device = next(iter(state_as[0].values())).device
    all_costs = {}
    rng = torch.Generator()
    rng.manual_seed(seed)

    with torch.no_grad():
        for iteration in range(max_iter):
            progress = False
            for p_ix in torch.randperm(len(perm_names), generator=rng):
                p = perm_names[p_ix]
                pg = spec[p]
                n, axes = pg.size, pg.state
                A = torch.zeros(n, n, device=device)
                for ax in axes:
                    if ax.key.endswith(skip_suffixes):
                        continue
                    for state_a, state_b in zip(state_as, state_bs):
                        if skip_missing and not (
                            ax.key in state_a and ax.key in state_b
                        ):
                            continue
                        # Detach for safety with keep_vars=True state dicts
                        w_a, w_b = state_a[ax.key].detach(), state_b[ax.key].detach()
                        A.add_(cross_weights(w_a, w_b, ax.axis))

                assert A.norm() > 0
                newP = lsa_solver(A)

                oldL, newL = A.diag().sum(), A[torch.arange(n), newP].sum()
                progress = progress or newL > oldL + 1e-12
                
                # Only print if verbose and show progress for first few iterations or significant changes
                if verbose and (iteration < 3 or newL - oldL > 1e-6):
                    print(f"  iter {iteration} | {p.key}:{p.axis} | improvement: {newL - oldL:.6e}")

                perm[p] = perm[p][newP]
                all_costs[p] = A
                for state_b in state_bs:
                    apply_perm({p: newP}, spec, state_b, inplace=True)

            if not progress:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        if return_costs:
            return perm, all_costs
        return perm








