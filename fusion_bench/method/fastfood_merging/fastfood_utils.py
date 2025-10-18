# fastfood_utils.py
"""
Utility functions for Fastfood/SRHT-based model merging.

This module contains:
- Fastfood/SRHT projection operators
- TIES merging functions
- EMA (Exponential Moving Average) merging
- Zero-aware aggregation functions
- Helper utilities
"""

from __future__ import annotations
import math
import hashlib
from typing import List, Tuple, Callable

import torch
from torch import Tensor

# Constants
EPS = 1e-12


# ============================================================================
# Fastfood / SRHT Core Operations
# ============================================================================

def next_pow2(n: int) -> int:
    """
    Compute the next power of 2 greater than or equal to n.
    
    Args:
        n: Input integer
        
    Returns:
        Next power of 2 >= n
    """
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def fwht_inplace_ortho(x: Tensor) -> Tensor:
    """
    In-place orthonormal Fast Walsh-Hadamard Transform (FWHT) along the last dimension.
    
    The transform is scaled by 1/sqrt(n) to maintain orthonormality.
    
    Args:
        x: Input tensor with last dimension being the transform dimension
        
    Returns:
        Transformed tensor (modified in-place)
    """
    n = x.shape[-1]
    if n <= 1:
        return x
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :]
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.view(-1, n)
        h *= 2
    x.mul_(1.0 / math.sqrt(n))
    return x


def seed_from_string(s: str) -> int:
    """
    Generate a deterministic seed from a string using MD5 hash.
    
    Args:
        s: Input string
        
    Returns:
        Integer seed derived from the string
    """
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")


def create_fastfood_ops(
    global_dim: int,
    proj_dim: int,
    *,
    seed_key: str,
    device: torch.device,
    use_G: bool,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """
    Build a Fastfood operator with:
      V = H Π G H B ∈ R^{L×L}, L = 2^⌈log2 D⌉
      P = random row subset of size m = proj_dim
    We return:
      fwd(x)  = sqrt(L/m) * P V [x; 0]
      lift(y) = V^T P^T (y / sqrt(L/m))
    The same (B, G, Π, P) are reused for all tensors sharing `seed_key`.
    """
    torch.manual_seed(seed_from_string(seed_key))
    D = int(global_dim)
    L = next_pow2(D)
    m = max(1, int(proj_dim))

    # Fastfood parameters
    B = (torch.randint(0, 2, (L,), dtype=torch.int8, device=device) * 2 - 1).to(
        dtype=torch.float32
    )
    G = (
        torch.randn(L, device=device, dtype=torch.float32)
        if use_G
        else torch.ones(L, device=device, dtype=torch.float32)
    )
    Pi = torch.randperm(L, device=device)
    inv_Pi = torch.argsort(Pi)

    # JL row subset and scaling (subsampled SRHT)
    row_idx = torch.randperm(L, device=device)[:m]
    scale = math.sqrt(L / m)

    def fwd(xD: Tensor) -> Tensor:
        assert xD.shape[-1] == D
        x = xD
        if D < L:
            x = torch.nn.functional.pad(x, (0, L - D))
        x = x.to(torch.float32, copy=False)
        x.mul_(B)
        fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G)
        fwht_inplace_ortho(x)
        x = x.index_select(dim=-1, index=row_idx)  # P V x
        return (scale * x).contiguous()

    def lift(y: Tensor) -> Tensor:
        y = (y.to(torch.float32, copy=False) / scale)
        y_full = torch.zeros(y.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
        y_full.index_copy_(dim=-1, index=row_idx, source=y)  # P^T y
        fwht_inplace_ortho(y_full)
        y_full.mul_(G)
        y_full = y_full[..., inv_Pi]
        fwht_inplace_ortho(y_full)
        y_full.mul_(B)  # V^T P^T y
        return y_full[..., :D].contiguous()

    return fwd, lift


# ============================================================================
# TIES Merging Functions
# ============================================================================

@torch.no_grad()
def resolve_zero_signs(sign_to_mult: Tensor, method: str = "majority") -> Tensor:
    """
    Resolve zero signs in a tensor by majority or minority rule.
    
    Args:
        sign_to_mult: Tensor with signs to resolve
        method: Resolution method - "majority" or "minority"
    
    Returns:
        Tensor with resolved signs
    """
    majority_sign = torch.sign(sign_to_mult.sum())
    
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


@torch.no_grad()
def resolve_sign(v: Tensor) -> Tensor:
    """
    Resolve the sign of a tensor by majority rule across tasks.
    
    Args:
        v: Input tensor [K, ...] where K is number of tasks
    
    Returns:
        Tensor with resolved signs for each parameter position
    """
    sign_to_mult = torch.sign(v.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


@torch.no_grad()
def ties_disjoint_merge(v: Tensor, merge_func: str, sign_to_mult: Tensor) -> Tensor:
    """
    Perform TIES disjoint merging using a specified merge function.
    
    Only parameters that agree with the elected sign are merged.
    
    Args:
        v: Input tensor [K, ...] where K is number of tasks
        merge_func: Merge function - "mean", "sum", or "max"
        sign_to_mult: Tensor with elected signs for merging
    
    Returns:
        Merged tensor
    """
    merge_func = merge_func.split("-")[-1]
    
    # Select entries that agree with the elected sign
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, v > 0, v < 0)
        selected_entries = v * rows_to_keep
    else:
        # Fallback: select all non-zero entries
        rows_to_keep = v != 0
        selected_entries = v * rows_to_keep
    
    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"TIES merge method {merge_func} is not defined.")
    
    return disjoint_aggs


@torch.no_grad()
def ties_merge_subspace(
    U: Tensor,
    ties_merge_func: str = "sum",
    weights: List[float] | None = None
) -> Tensor:
    """
    TIES merging in subspace (no trimming, only elect + disjoint merge).
    
    Args:
        U: [K, ..., M] stacked task vectors in subspace
        ties_merge_func: "sum", "mean", or "max" for disjoint aggregation
        weights: Task importance weights (applied as scaling before TIES)
    
    Returns:
        Merged vector in subspace
    """
    K = U.shape[0]
    if K == 0:
        raise ValueError("No task vectors to merge")
    if K == 1:
        w = weights[0] if weights else 1.0
        return w * U[0]
    
    # Flatten for easier processing
    orig_shape = U.shape[1:]
    U_flat = U.view(K, -1)  # [K, numel]
    
    # Apply task weights if provided (before TIES processing)
    if weights is not None:
        w_tensor = torch.tensor(weights, dtype=U_flat.dtype, device=U_flat.device).view(K, 1)
        U_flat = w_tensor * U_flat
    
    # TIES Phase 1: Elect (Skip Trim - no parameter pruning)
    print("RESOLVING SIGN (TIES)")
    final_signs = resolve_sign(U_flat)
    
    # TIES Phase 2: Disjoint Merge
    print(f"DISJOINT AGGREGATION (TIES): {ties_merge_func}")
    merged_tv = ties_disjoint_merge(U_flat, ties_merge_func, final_signs)
    
    return merged_tv.view(orig_shape)


# ============================================================================
# EMA (Exponential Moving Average) Merging Functions
# ============================================================================

@torch.no_grad()
def ema_adaptive_beta(
    z_acc: Tensor,
    z_new: Tensor,
    gamma: float = 1.2,
    w_c: float = 0.6,
    w_s: float = 0.4,
    eps: float = 1e-8
) -> float:
    """
    Compute adaptive β_t for EMA based on alignment and scale.
    
    The adaptive beta balances the accumulator and new task based on:
    - Cosine alignment (how well they agree in direction)
    - Scale ratio (relative magnitudes)
    
    Args:
        z_acc: Current accumulator vector
        z_new: New task vector to incorporate
        gamma: Sigmoid scaling factor (default 1.2)
        w_c: Weight for cosine alignment term (default 0.6)
        w_s: Weight for scale ratio term (default 0.4)
        eps: Small constant for numerical stability
    
    Returns:
        β_t ∈ [0,1]: mixing coefficient for EMA update
    """
    if z_acc.norm() < eps:
        return 0.0
    
    # Cosine alignment: how well z_new aligns with current accumulator
    c = (z_acc @ z_new) / (z_acc.norm() * z_new.norm() + eps)
    
    # Scale ratio: relative magnitude of accumulator vs new task
    s = z_acc.norm() / (z_new.norm() + eps)
    
    # Map to β via sigmoid (higher alignment & balanced scale → higher β)
    beta = torch.sigmoid(gamma * (w_c * c + w_s * s))
    return float(beta.item())


@torch.no_grad()
def ema_merge_subspace(
    U: Tensor,
    task_order: str = "given",
    ema_gamma: float = 1.2,
    ema_w_c: float = 0.6,
    ema_w_s: float = 0.4,
    weights: List[float] | None = None,
    custom_order: List[int] | None = None
) -> Tensor:
    """
    EMA merging in subspace with adaptive β_t.
    
    Processes tasks sequentially, adaptively weighting based on alignment
    and scale between the current accumulator and each new task.
    
    Args:
        U: [K, ..., M] stacked task vectors in subspace
        task_order: "given", "random", "cosine_similarity", or "custom"
        ema_gamma: Sigmoid scaling factor for β computation
        ema_w_c: Weight for cosine alignment term
        ema_w_s: Weight for scale ratio term
        weights: Task importance weights (applied as scaling)
        custom_order: List of task indices when task_order="custom"
    
    Returns:
        Merged vector in subspace
    """
    K = U.shape[0]
    if K == 0:
        raise ValueError("No task vectors to merge")
    if K == 1:
        w = weights[0] if weights else 1.0
        return w * U[0]
    
    # Flatten for easier processing
    orig_shape = U.shape[1:]
    U_flat = U.view(K, -1)  # [K, numel]
    
    # Determine processing order
    if task_order == "random":
        indices = torch.randperm(K).tolist()
    elif task_order == "custom":
        if custom_order is None:
            raise ValueError("custom_order must be provided when task_order='custom'")
        if len(custom_order) != K or set(custom_order) != set(range(K)):
            raise ValueError(f"custom_order must be a permutation of [0,1,...,{K-1}], got {custom_order}")
        indices = custom_order
    elif task_order == "cosine_similarity":
        # Start with first, then order remaining by cosine similarity to current accumulator
        indices = [0]
        z_acc = U_flat[0].clone()
        remaining = set(range(1, K))
        
        while remaining:
            # Find task most similar to current accumulator
            best_idx = None
            best_sim = -2.0  # cosine ∈ [-1,1]
            
            for idx in remaining:
                z_cand = U_flat[idx]
                if z_acc.norm() > 1e-8 and z_cand.norm() > 1e-8:
                    sim = (z_acc @ z_cand) / (z_acc.norm() * z_cand.norm())
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = idx
            
            if best_idx is not None:
                indices.append(best_idx)
                remaining.remove(best_idx)
                # Update accumulator for next iteration (simple average so far)
                z_acc = torch.stack([U_flat[i] for i in indices]).mean(dim=0)
            else:
                # Fallback: just take first remaining
                best_idx = next(iter(remaining))
                indices.append(best_idx)
                remaining.remove(best_idx)
    else:  # "given"
        indices = list(range(K))
    
    # EMA streaming update
    z_acc = torch.zeros_like(U_flat[0])
    
    for i, task_idx in enumerate(indices):
        z_new = U_flat[task_idx]
        
        # Apply task weight if provided
        if weights is not None:
            z_new = weights[task_idx] * z_new
        
        # Adaptive β based on current accumulator and new task
        beta = ema_adaptive_beta(z_acc, z_new, ema_gamma, ema_w_c, ema_w_s)
        
        # EMA update: z_acc = β * z_acc + (1-β) * z_new
        z_acc = beta * z_acc + (1 - beta) * z_new
    
    return z_acc.view(orig_shape)


# ============================================================================
# Zero-Aware Aggregation Functions
# ============================================================================

@torch.no_grad()
def zero_aware_aggregate(
    U: Tensor,
    merge_func: str,
    weights: List[float] | None = None,
    **kwargs
) -> Tensor:
    """
    Zero-aware aggregation of task vectors with multiple merge strategies.
    
    This function handles merging of task vectors while respecting zero entries,
    which is crucial for sparse or disentangled parameter merging.
    
    Supported merge functions:
      - 'sum': Elementwise sum (weights optional)
      - 'mean': Elementwise mean over *nonzero* contributors only
      - 'max': Elementwise argmax by |u_k|; zeros preserved
      - 'signmax': Per position, pick dominant sign, then largest |u_k| with that sign
      - 'ema': Exponential Moving Average with adaptive β_t
      - 'ties_sum': TIES merging with sum aggregation
      - 'ties_mean': TIES merging with mean aggregation
      - 'ties_max': TIES merging with max aggregation
    
    Args:
        U: [K, ..., M] stacked task vectors (K = number of tasks)
        merge_func: Aggregation function name
        weights: Optional task importance weights
        **kwargs: Additional parameters (e.g., EMA parameters)
    
    Returns:
        Merged tensor
    """
    K = U.shape[0]
    mf = merge_func.lower()
    
    valid_funcs = {"sum", "mean", "max", "signmax", "ema", "ties_sum", "ties_mean", "ties_max"}
    if mf not in valid_funcs:
        raise ValueError(f"merge_func={merge_func} not in {valid_funcs}")

    # EMA merging
    if mf == "ema":
        return ema_merge_subspace(
            U,
            task_order=kwargs.get("ema_task_order", "given"),
            ema_gamma=kwargs.get("ema_gamma", 1.2),
            ema_w_c=kwargs.get("ema_w_c", 0.6),
            ema_w_s=kwargs.get("ema_w_s", 0.4),
            weights=weights,
            custom_order=kwargs.get("ema_custom_order", None)
        )

    # TIES merging variants
    if mf.startswith("ties_"):
        ties_func = mf.split("_", 1)[1]  # Extract "sum", "mean", or "max"
        return ties_merge_subspace(
            U,
            ties_merge_func=ties_func,
            weights=weights
        )

    # Sum aggregation
    if mf == "sum":
        if weights is None:
            return U.sum(dim=0)
        w = torch.tensor(weights, dtype=U.dtype, device=U.device).view(
            K, *([1] * (U.ndim - 1))
        )
        return (w * U).sum(dim=0)

    # Mean aggregation (zero-aware)
    mask = (U != 0)
    if mf == "mean":
        if weights is None:
            denom = mask.sum(dim=0).clamp_min(1)
            return (U * mask).sum(dim=0) / denom
        w = torch.tensor(weights, dtype=U.dtype, device=U.device).view(
            K, *([1] * (U.ndim - 1))
        )
        num = (w * U * mask).sum(dim=0)
        den = (w * mask).sum(dim=0).clamp_min(1e-12)
        return num / den

    absU = U.abs()

    # Max aggregation
    if mf == "max":
        idx = absU.argmax(dim=0)
        out = torch.gather(U, dim=0, index=idx.unsqueeze(0)).squeeze(0)
        all_zero = (absU.sum(dim=0) == 0)
        if all_zero.any():
            out = out.masked_fill(all_zero, 0.0)
        return out

    # Signmax aggregation
    if mf == "signmax":
        sgn = torch.sign(U)                 # {-1, 0, +1}
        pos_count = (sgn > 0).sum(dim=0)
        neg_count = (sgn < 0).sum(dim=0)

        dom_pos = pos_count > neg_count     # positive majority
        dom_neg = neg_count > pos_count     # negative majority
        tie_or_none = ~(dom_pos | dom_neg)  # tie or all zeros

        match_pos = (U > 0) & dom_pos.unsqueeze(0)
        match_neg = (U < 0) & dom_neg.unsqueeze(0)
        match = match_pos | match_neg

        masked_mag = absU.clone()
        masked_mag[~match] = -float("inf")

        idx_dom = masked_mag.argmax(dim=0)
        idx_fallback = absU.argmax(dim=0)
        idx = torch.where(tie_or_none, idx_fallback, idx_dom)

        out = torch.gather(U, dim=0, index=idx.unsqueeze(0)).squeeze(0)

        all_zero = (absU.sum(dim=0) == 0)
        if all_zero.any():
            out = out.masked_fill(all_zero, 0.0)
        return out


# ============================================================================
# Helper Utilities
# ============================================================================

def layer_key(name: str) -> str:
    """
    Heuristic layer-grouping key for parameter names.
    
    This works for most Hugging Face transformer models by extracting
    the first 2-3 components of the parameter name.
    
    Args:
        name: Full parameter name (e.g., "model.layer.0.attention.query.weight")
        
    Returns:
        Layer group key (e.g., "model.layer.0")
    """
    parts = name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name


def compute_global_dim(state_dict: dict, keys: List[str]) -> int:
    """
    Compute the total number of parameters across specified keys.
    
    Args:
        state_dict: Model state dictionary
        keys: List of parameter names to include
        
    Returns:
        Total number of parameters
    """
    total_dim = 0
    for k in keys:
        if k in state_dict:
            total_dim += state_dict[k].numel()
    return total_dim


def normalize_weights(weights: List[float] | None, num_tasks: int) -> List[float]:
    """
    Normalize task weights to sum to 1.
    
    Args:
        weights: Optional list of task weights
        num_tasks: Number of tasks
        
    Returns:
        Normalized weights (uniform if None provided)
    """
    if weights is None:
        return [1.0 / num_tasks] * num_tasks
    
    if len(weights) != num_tasks:
        raise ValueError(f"Number of weights ({len(weights)}) != number of tasks ({num_tasks})")
    
    total = sum(weights) + EPS
    return [w / total for w in weights]
