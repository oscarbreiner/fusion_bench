# projection_size_estimator.py
"""
Projection Size Estimator for Fastfood Subspace Merging

This module provides adaptive projection size estimation for subspace merging operations.
Supports two modes:
  - tensor: Per-tensor projection size estimation
  - layer: Per-layer projection size estimation (shared across layer's parameters)

Five strategies:
  - fixed: m = ratio * d_last
  - random: m ~ U[m_min, f_max * d_last]
  - rank: m = beta * estimated_rank
    - Tensor mode: uses stable rank (||W||_F^2 / ||W||_2^2) via power iteration
    - Layer mode: uses effective rank (entropy-based) from SVD variance spectrum
  - layer_progressive: Progressive scaling from start_proj_size to end_proj_size
    - Requires layer_idx and num_layers
    - Linear growth: m(t) = start + t * (end - start)
    - Exponential growth: m(t) = start * (end/start)^t
    - where t = layer_idx / (num_layers - 1) ∈ [0,1]
  - layer_group: Split network into feature extraction vs head with different ratios
    - Requires layer_idx and boundary_layer
  - layer_power_law: Power-law redistribution within each layer
    - Operates PER-LAYER: Groups parameters by layer, applies power-law budget within each layer
    - Formula: m_i = B * (d_i^alpha) / sum(d_j^alpha) where B = global_ratio * sum(d_j in layer)

Usage Example:
    >>> from projection_size_estimator import ProjSizeCfg, proj_size_for
    >>> 
    >>> cfg = ProjSizeCfg(ratio=0.25, beta=2.5, pow2_round=True)
    >>> 
    >>> # Tensor mode with fixed strategy
    >>> weight = torch.randn(512, 256)
    >>> m = proj_size_for(weight, mode="tensor", strategy="fixed", cfg=cfg)
    >>> 
    >>> # Layer mode with rank strategy
    >>> layer_params = {"attn.weight": torch.randn(512, 256), "attn.bias": torch.randn(512)}
    >>> m = proj_size_for(layer_params, mode="layer", strategy="rank", cfg=cfg)
    >>> 
    >>> # Layer progressive strategy
    >>> cfg_progressive = ProjSizeCfg(start_proj_size=64, end_proj_size=512, growth_mode="linear")
    >>> m = proj_size_for(weight, mode="tensor", strategy="layer_progressive", 
    ...                   cfg=cfg_progressive, layer_idx=5, num_layers=12)
    >>> 
    >>> # Layer power-law strategy (per-layer budget)
    >>> cfg_power = ProjSizeCfg(global_ratio=0.25, power_law_alpha=0.85)
    >>> layer_dims = [256, 512, 1024]  # Dimensions of tensors in THIS LAYER
    >>> m = proj_size_for(weight, mode="tensor", strategy="layer_power_law",
    ...                   cfg=cfg_power, all_dims=layer_dims)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
import math
import random
import torch

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
import random
import math
import torch

Mode = Literal["tensor", "layer"]
Strategy = Literal["fixed", "random", "rank", "layer_progressive", "layer_group", "layer_power_law"]
Pow2Mode = Literal["ceil", "floor", "nearest"]
GrowthMode = Literal["ceil", "floor", "nearest"]
GrowthMode = Literal["linear", "exponential"]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProjSizeCfg:
    """
    Configuration for projection size estimation.
    
    Attributes:
        m_min: Hard lower bound for projection size
        f_max: Maximum fraction of d_last (m <= f_max * d_last)
        pow2_round: Whether to round m to nearest power-of-two
        pow2_mode: Rounding mode - "ceil", "floor", or "nearest"
        
        ratio: For "fixed" strategy - m = ratio * d_last
        beta: For "rank" strategy - m = beta * estimated_rank
        
        start_proj_ratio: For "layer_progressive" strategy - projection ratio at first layer (0.0-1.0)
        end_proj_ratio: For "layer_progressive" strategy - projection ratio at last layer (0.0-1.0)
        growth_mode: For "layer_progressive" strategy - "linear" or "exponential" growth
        
        rng: Random number generator for "random" strategy (uses global if None)
    """
    m_min: int = 16
    f_max: float = 1.0
    pow2_round: bool = False
    pow2_mode: Pow2Mode = "ceil"
    
    ratio: float = 0.25
    beta: float = 2.5
    
    start_proj_ratio: float = 0.1
    end_proj_ratio: float = 1.0
    growth_mode: GrowthMode = "linear"
    
    # Layer group strategy parameters
    group_boundary_layer: int = 5  # Layer index separating feature extraction from head
    feature_proj_ratio: float = 0.3  # Projection ratio for feature extraction layers
    head_proj_ratio: float = 0.8  # Projection ratio for head layers
    
    # Global power-law strategy parameters
    global_ratio: float = 0.25     # Overall compression budget (average ratio across all layers)
    power_law_alpha: float = 0.85  # Power-law exponent for dimension-aware redistribution
                                   # alpha < 1: larger layers compressed more, smaller layers less
                                   # alpha = 1: uniform ratio (equivalent to fixed strategy)
                                   # alpha > 1: larger layers compressed less, smaller layers more
    
    rng: Optional[random.Random] = None


# ============================================================================
# Power-of-Two Utilities
# ============================================================================

def next_pow2(n: int) -> int:
    """Return next power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def prev_pow2(n: int) -> int:
    """Return previous power of 2 <= n."""
    if n <= 1:
        return 1
    return 1 << (n.bit_length() - 1)


def nearest_pow2(n: int) -> int:
    """Return nearest power of 2 to n."""
    up, down = next_pow2(n), prev_pow2(n)
    return up if (up - n) <= (n - down) else down


def round_pow2(n: int, mode: Pow2Mode = "ceil") -> int:
    """
    Round n to a power of 2.
    
    Args:
        n: Input integer
        mode: Rounding mode - "ceil", "floor", or "nearest"
        
    Returns:
        Rounded power of 2
    """
    n = max(1, int(n))
    if mode == "ceil":
        return next_pow2(n)
    elif mode == "floor":
        return prev_pow2(n)
    else:  # nearest
        return nearest_pow2(n)


def clip_and_pow2(
    m: int,
    m_min: int,
    m_max: int,
    use_pow2: bool,
    mode: Pow2Mode
) -> int:
    """
    Clip m to [m_min, m_max] and optionally round to power-of-two.
    
    Args:
        m: Target projection size
        m_min: Minimum allowed size
        m_max: Maximum allowed size
        use_pow2: Whether to round to power-of-two
        mode: Rounding mode if use_pow2=True
        
    Returns:
        Clipped (and optionally rounded) projection size
    """
    m = max(m_min, min(int(m), int(m_max)))
    return round_pow2(m, mode) if use_pow2 else m


# ============================================================================
# Layer Progressive Projection Size Computation
# ============================================================================

def compute_layer_progressive_ratio(
    layer_idx: int,
    num_layers: int,
    start_ratio: float,
    end_ratio: float,
    growth_mode: GrowthMode = "linear"
) -> float:
    """
    Compute projection ratio for a layer based on its index using progressive scaling.
    
    Args:
        layer_idx: Current layer index (0-based)
        num_layers: Total number of layers
        start_ratio: Projection ratio at first layer (0.0-1.0, layer_idx=0)
        end_ratio: Projection ratio at last layer (0.0-1.0, layer_idx=num_layers-1)
        growth_mode: "linear" or "exponential" growth
        
    Returns:
        Projection ratio for the given layer (0.0-1.0)
        
    Examples:
        >>> # Linear growth from 0.1 to 1.0 over 10 layers
        >>> compute_layer_progressive_ratio(0, 10, 0.1, 1.0, "linear")
        0.1
        >>> compute_layer_progressive_ratio(9, 10, 0.1, 1.0, "linear")
        1.0
        >>> compute_layer_progressive_ratio(5, 10, 0.1, 1.0, "linear")
        0.55
        
        >>> # Exponential growth
        >>> compute_layer_progressive_ratio(0, 10, 0.1, 1.0, "exponential")
        0.1
        >>> compute_layer_progressive_ratio(9, 10, 0.1, 1.0, "exponential")
        1.0
    """
    if num_layers <= 1:
        return start_ratio
    
    # Normalize layer position to [0, 1]
    t = layer_idx / (num_layers - 1)
    
    if growth_mode == "linear":
        # Linear interpolation: r(t) = start + t * (end - start)
        ratio = start_ratio + t * (end_ratio - start_ratio)
    elif growth_mode == "exponential":
        # Exponential growth: r(t) = start * (end/start)^t
        if start_ratio <= 0:
            return start_ratio
        ratio_mult = end_ratio / start_ratio
        ratio = start_ratio * (ratio_mult ** t)
    else:
        raise ValueError(f"Unknown growth_mode: {growth_mode}")
    
    return float(ratio)


# ============================================================================
# Layer Group Projection Size Computation
# ============================================================================

def compute_layer_group_ratio(
    layer_idx: int,
    boundary_layer: int,
    feature_ratio: float,
    head_ratio: float
) -> float:
    """
    Compute projection ratio based on layer group (feature extraction vs head).
    
    This strategy splits the network into two groups:
    - Feature extraction layers (layer_idx < boundary_layer): use feature_ratio
    - Head layers (layer_idx >= boundary_layer): use head_ratio
    
    Args:
        layer_idx: Current layer index (0-based)
        boundary_layer: Layer index threshold separating groups
        feature_ratio: Projection ratio for feature extraction layers (0.0-1.0)
        head_ratio: Projection ratio for head layers (0.0-1.0)
        
    Returns:
        Projection ratio for the given layer (0.0-1.0)
        
    Examples:
        >>> # Split at layer 5, feature extraction uses 30%, head uses 80%
        >>> compute_layer_group_ratio(3, 5, 0.3, 0.8)
        0.3
        >>> compute_layer_group_ratio(5, 5, 0.3, 0.8)
        0.8
        >>> compute_layer_group_ratio(10, 5, 0.3, 0.8)
        0.8
    """
    if layer_idx < boundary_layer:
        return float(feature_ratio)
    else:
        return float(head_ratio)


# ============================================================================
# Layer Power-Law Projection Size Computation
# ============================================================================

def compute_layer_power_law_size(
    d_i: int,
    all_dims: List[int],
    global_ratio: float,
    alpha: float
) -> int:
    """
    Compute projection size using power-law redistribution of budget within a layer.
    
    This function operates PER-LAYER: The budget is calculated for all tensors 
    within the same layer, then redistributed using a power-law formula.
    
    This strategy maintains a per-layer compression budget (global_ratio) while
    adapting per-tensor projection sizes based on their relative dimension using
    a power-law distribution.
    
    Formula:
        B = global_ratio * sum(all_dims)  # Total projection budget FOR THIS LAYER
        weight_i = d_i ** alpha
        m_i = B * weight_i / sum(weight_j)
    
    Where:
        - B: Total projection capacity budget FOR THIS LAYER
        - d_i: Dimension of current tensor
        - all_dims: List of dimensions of ALL tensors IN THE SAME LAYER
        - alpha: Power-law exponent controlling redistribution
          * alpha < 1: Larger tensors compressed more, smaller tensors less (default: 0.85)
          * alpha = 1: Uniform ratio (equivalent to fixed strategy)
          * alpha > 1: Larger tensors compressed less, smaller tensors more
    
    This ensures:
        1. Average compression = global_ratio (per-layer budget preserved)
        2. Larger tensors within a layer get more absolute capacity but lower ratio
        3. Smaller tensors within a layer get less absolute capacity but higher ratio
    
    Args:
        d_i: Dimension of current parameter (d_last)
        all_dims: List of dimensions of all tensors IN THE SAME LAYER
        global_ratio: Per-layer compression budget (0.0-1.0, e.g., 0.25 for 25% average compression)
        alpha: Power-law exponent (typically 0.7-0.95, default 0.85)
        
    Returns:
        Projection size m_i for the current parameter
        
    Examples:
        >>> # Three tensors in ONE LAYER with dims [256, 512, 1024], global_ratio=0.25, alpha=0.85
        >>> all_dims = [256, 512, 1024]
        >>> # Total budget for this layer: B = 0.25 * (256 + 512 + 1024) = 448
        >>> # Weights: 256^0.85=164, 512^0.85=311, 1024^0.85=590
        >>> # Sum weights: 1065
        >>> # Tensor 0: m_0 = 448 * 164/1065 = 69  → ratio=69/256=0.27 (higher than 0.25)
        >>> # Tensor 1: m_1 = 448 * 311/1065 = 131 → ratio=131/512=0.26
        >>> # Tensor 2: m_2 = 448 * 590/1065 = 248 → ratio=248/1024=0.24 (lower than 0.25)
        >>> compute_layer_power_law_size(256, all_dims, 0.25, 0.85)
        69
        >>> compute_layer_power_law_size(1024, all_dims, 0.25, 0.85)
        248
        
        >>> # With alpha=1.0 (uniform ratio, equivalent to fixed strategy)
        >>> compute_layer_power_law_size(256, all_dims, 0.25, 1.0)
        64
        >>> compute_layer_power_law_size(1024, all_dims, 0.25, 1.0)
        256
    """
    if not all_dims or d_i <= 0:
        return 1
    
    # Total projection budget
    total_dim = sum(all_dims)
    B = global_ratio * total_dim
    
    # Compute power-law weights
    weights = [d ** alpha for d in all_dims]
    total_weight = sum(weights)
    
    if total_weight < 1e-12:
        # Fallback to uniform distribution
        return max(1, int(B * d_i / total_dim))
    
    # Compute projection size for current dimension
    weight_i = d_i ** alpha
    m_i = B * weight_i / total_weight
    
    return max(1, int(m_i))


# ============================================================================
# Dimension Extraction
# ============================================================================

def last_dim_from_tensor(W: torch.Tensor) -> int:
    """
    Extract logical input dimension from a parameter tensor.
    
    - Linear (out, in) -> in
    - Conv2d (out, in, kh, kw) -> in * kh * kw
    - Others -> last axis
    
    Args:
        W: Parameter tensor
        
    Returns:
        Logical input dimension
    """
    if W.ndim == 2:
        return int(W.shape[1])
    if W.ndim == 4:
        _, in_c, kh, kw = W.shape
        return int(in_c * kh * kw)
    return int(W.shape[-1])


# ============================================================================
# Rank Estimation - Tensor Mode (Stable Rank via Power Iteration)
# ============================================================================

@torch.no_grad()
def spectral_norm_powerit(W: torch.Tensor, n_iter: int = 10) -> float:
    """
    Approximate top singular value via power iteration (SVD-free).
    
    Works on 2D matrix M:
      - Linear -> (out, in)
      - Conv -> (out, in*kh*kw)
    
    Args:
        W: Parameter tensor (2D or 4D)
        n_iter: Number of power iterations
        
    Returns:
        Approximate largest singular value
    """
    # Reshape to 2D if needed
    if W.ndim == 4:
        out, in_c, kh, kw = W.shape
        M = W.reshape(out, in_c * kh * kw)
    elif W.ndim == 2:
        M = W
    else:
        M = W.reshape(W.shape[0], -1)
    
    A = M.detach().to(dtype=torch.float32)
    device = A.device
    
    # Initialize random vector
    v = torch.randn(A.shape[1], dtype=torch.float32, device=device)
    v = v / (v.norm() + 1e-12)
    
    # Power iteration
    for _ in range(n_iter):
        u = A @ v
        u = u / (u.norm() + 1e-12)
        v = A.t() @ u
        v = v / (v.norm() + 1e-12)
    
    # Rayleigh quotient approximates sigma_max (use abs to avoid tiny negatives)
    return float(max((u @ (A @ v)).abs().item(), 1e-12)) if A.numel() > 0 else 1e-12


@torch.no_grad()
def stable_rank_tensor(W: torch.Tensor, n_iter: int = 8) -> float:
    """
    Compute stable rank: ||W||_F^2 / ||W||_2^2.
    
    Uses power iteration for spectral norm (no full SVD needed).
    Cheap and robust for large tensors.
    
    Args:
        W: Parameter tensor
        n_iter: Number of power iterations for spectral norm
        
    Returns:
        Stable rank estimate
    """
    fro2 = float(W.float().pow(2).sum().item())
    sigma_max = max(spectral_norm_powerit(W, n_iter=n_iter), 1e-12)
    return float(fro2 / (sigma_max * sigma_max + 1e-12))


# ============================================================================
# Rank Estimation - Layer Mode (Effective Rank via SVD + Entropy)
# ============================================================================

@torch.no_grad()
def svdvals_2d(W: torch.Tensor) -> torch.Tensor:
    """
    Compute singular values for a 2D weight matrix.
    
    - Linear (out, in) -> direct SVD
    - Conv (out, in, kh, kw) -> reshape to (out, in*kh*kw) then SVD
    - 1D tensors (bias, LayerNorm) -> return empty (skip)
    
    Args:
        W: Parameter tensor
        
    Returns:
        Singular values (empty tensor if not 2D/4D)
    """
    if W.ndim == 4:
        out, in_c, kh, kw = W.shape
        M = W.reshape(out, in_c * kh * kw)
    elif W.ndim == 2:
        M = W
    else:
        return torch.empty(0, dtype=torch.float64)
    
    return torch.linalg.svdvals(M.to(torch.float64, non_blocking=True).cpu())


@torch.no_grad()
def effective_rank_from_svals(S: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank from singular values.
    
    Uses variance spectrum (λ_i = σ_i^2):
        r_eff = exp(-Σ p_i log p_i), where p_i = λ_i / Σ λ_i
    
    This matches the intrinsic dimension analysis implementation.
    
    Args:
        S: Singular values
        eps: Small constant for numerical stability
        
    Returns:
        Effective rank
    """
    if S.numel() == 0:
        return 0.0
    
    # Variance spectrum
    S = S.to(torch.float64)
    v = S ** 2
    vsum = torch.sum(v)
    
    if vsum <= 0:
        return 0.0
    
    # Probability distribution from variance spectrum
    p = (v / vsum).clamp_min(eps)
    
    # Entropy-based effective rank
    H = -torch.sum(p * torch.log(p))
    return float(torch.exp(H).item())


@torch.no_grad()
def effective_rank_layer(layer_params: Dict[str, torch.Tensor]) -> float:
    """
    Compute weighted average effective rank across layer's 2D tensors.
    
    Mirrors the intrinsic dimension analysis methodology:
    - Per-tensor SVD and entropy-based effective rank
    - Weighted by tensor numel
    - Skips 1D tensors (bias, LayerNorm)
    
    Args:
        layer_params: Dictionary of parameter name -> tensor for a layer
        
    Returns:
        Weighted effective rank for the layer
    """
    ranks, weights = [], []
    
    for name, W in layer_params.items():
        # Only process 2D (Linear) and 4D (Conv) tensors
        if W.ndim not in (2, 4):
            continue
        
        S = svdvals_2d(W)
        if S.numel() == 0:
            continue
        
        # Numerical trimming: drop tiny singular values
        smax = float(S.max().item())
        if smax <= 0:
            continue
        
        rel_tol = 1e-12 * max(W.shape)
        S = S[S > rel_tol * smax]
        
        if S.numel() == 0:
            continue
        
        # Compute effective rank from variance spectrum
        r_eff = effective_rank_from_svals(S)
        ranks.append(r_eff)
        weights.append(int(W.numel()))
    
    if not ranks:
        return 0.0
    
    # Weighted average by parameter count
    total_weight = sum(weights) + 1e-12
    return float(sum(r * w for r, w in zip(ranks, weights)) / total_weight)


# ============================================================================
# Projection Size Policies
# ============================================================================

def proj_size_for_tensor(
    W: torch.Tensor,
    strategy: Strategy,
    cfg: ProjSizeCfg,
    layer_idx: int = 0,
    num_layers: int = 1,
    all_dims: List[int] | None = None  # For layer_power_law strategy
) -> int:
    """
    Compute projection size for a single tensor.
    
    Args:
        W: Parameter tensor
        strategy: "fixed", "random", "rank", "layer_progressive", "layer_group", or "layer_power_law"
        cfg: Configuration for projection size estimation
        layer_idx: Current layer index (for layer_progressive/layer_group strategies)
        num_layers: Total number of layers (for layer_progressive/layer_group strategies)
        all_dims: List of all dimensions (for layer_power_law strategy)
        
    Returns:
        Projection size m
    """
    d_last = last_dim_from_tensor(W)
    m_max = max(cfg.m_min, int(cfg.f_max * d_last))
    
    if strategy == "fixed":
        m_tgt = int(cfg.ratio * d_last)
    elif strategy == "random":
        rng = cfg.rng or random
        m_tgt = rng.randint(cfg.m_min, m_max)
    elif strategy == "rank":
        r = stable_rank_tensor(W)
        m_tgt = int(cfg.beta * r)
    elif strategy == "layer_progressive":
        # Compute progressive ratio and apply to d_last
        prog_ratio = compute_layer_progressive_ratio(
            layer_idx, num_layers, cfg.start_proj_ratio, cfg.end_proj_ratio, cfg.growth_mode
        )
        m_tgt = int(prog_ratio * d_last)
    elif strategy == "layer_group":
        # Compute group-based ratio and apply to d_last
        group_ratio = compute_layer_group_ratio(
            layer_idx, cfg.group_boundary_layer, cfg.feature_proj_ratio, cfg.head_proj_ratio
        )
        m_tgt = int(group_ratio * d_last)
    elif strategy == "layer_power_law":
        # Compute power-law redistributed size based on global budget
        if all_dims is None:
            raise ValueError("layer_power_law strategy requires all_dims parameter")
        m_tgt = compute_layer_power_law_size(
            d_last, all_dims, cfg.global_ratio, cfg.power_law_alpha
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return clip_and_pow2(m_tgt, cfg.m_min, m_max, cfg.pow2_round, cfg.pow2_mode)


def proj_size_for_layer(
    layer_params: Dict[str, torch.Tensor],
    strategy: Strategy,
    cfg: ProjSizeCfg,
    layer_idx: int = 0,
    num_layers: int = 1,
    all_dims: List[int] | None = None  # For layer_power_law strategy
) -> int:
    """
    Compute a single projection size for an entire layer group.
    
    Uses the largest d_last among tensors for capacity bound.
    
    Args:
        layer_params: Dictionary of parameter name -> tensor for a layer
        strategy: "fixed", "random", "rank", "layer_progressive", "layer_group", or "layer_power_law"
        cfg: Configuration for projection size estimation
        layer_idx: Current layer index (for layer_progressive/layer_group strategies)
        num_layers: Total number of layers (for layer_progressive/layer_group strategies)
        all_dims: List of all dimensions (for layer_power_law strategy)
        
    Returns:
        Projection size m (shared across layer)
    """
    # Find largest input dimension among 2D/4D tensors
    d_candidates = [
        last_dim_from_tensor(W)
        for W in layer_params.values()
        if W.ndim in (2, 4)
    ]
    
    if not d_candidates:
        # No 2D/4D tensors, fallback to m_min
        return cfg.m_min
    
    d_last = max(d_candidates)
    m_max = max(cfg.m_min, int(cfg.f_max * d_last))
    
    if strategy == "fixed":
        m_tgt = int(cfg.ratio * d_last)
    elif strategy == "random":
        rng = cfg.rng or random
        m_tgt = rng.randint(cfg.m_min, m_max)
    elif strategy == "rank":
        r = effective_rank_layer(layer_params)
        m_tgt = int(cfg.beta * r)
    elif strategy == "layer_progressive":
        # Compute progressive ratio and apply to d_last
        prog_ratio = compute_layer_progressive_ratio(
            layer_idx, num_layers, cfg.start_proj_ratio, cfg.end_proj_ratio, cfg.growth_mode
        )
        m_tgt = int(prog_ratio * d_last)
    elif strategy == "layer_group":
        # Compute group-based ratio and apply to d_last
        group_ratio = compute_layer_group_ratio(
            layer_idx, cfg.group_boundary_layer, cfg.feature_proj_ratio, cfg.head_proj_ratio
        )
        m_tgt = int(group_ratio * d_last)
    elif strategy == "layer_power_law":
        # Compute power-law redistributed size based on global budget
        if all_dims is None:
            raise ValueError("layer_power_law strategy requires all_dims parameter")
        m_tgt = compute_layer_power_law_size(
            d_last, all_dims, cfg.global_ratio, cfg.power_law_alpha
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return clip_and_pow2(m_tgt, cfg.m_min, m_max, cfg.pow2_round, cfg.pow2_mode)


# ============================================================================
# Unified Entry Point
# ============================================================================

def proj_size_for(
    unit: torch.Tensor | Dict[str, torch.Tensor],
    mode: Mode,
    strategy: Strategy,
    cfg: ProjSizeCfg,
    layer_idx: int = 0,
    num_layers: int = 1,
    all_dims: List[int] | None = None  # For layer_power_law strategy
) -> int:
    """
    Unified entry point for projection size estimation.
    
    Args:
        unit: Either a single tensor (mode="tensor") or dict of tensors (mode="layer")
        mode: "tensor" or "layer"
        strategy: "fixed", "random", "rank", "layer_progressive", "layer_group", or "layer_power_law"
        cfg: Configuration for projection size estimation
        layer_idx: Current layer index (for layer_progressive/layer_group strategies)
        num_layers: Total number of layers (for layer_progressive/layer_group strategies)
        all_dims: List of all dimensions (for layer_power_law strategy)
        
    Returns:
        Projection size m
        
    Examples:
        >>> cfg = ProjSizeCfg(ratio=0.25, beta=2.5, pow2_round=True)
        >>> 
        >>> # Tensor mode with fixed strategy
        >>> weight = torch.randn(512, 256)
        >>> m = proj_size_for(weight, mode="tensor", strategy="fixed", cfg=cfg)
        >>> print(f"Projection size: {m}")
        >>> 
        >>> # Layer mode with rank strategy
        >>> layer = {"attn.weight": torch.randn(512, 256), "attn.bias": torch.randn(512)}
        >>> m = proj_size_for(layer, mode="layer", strategy="rank", cfg=cfg)
        >>> print(f"Layer projection size: {m}")
        >>> 
        >>> # Global power-law strategy
        >>> all_layer_dims = [256, 512, 768, 1024]
        >>> m = proj_size_for(weight, mode="tensor", strategy="layer_power_law", 
        ...                   cfg=cfg, all_dims=all_layer_dims)
        >>> print(f"Power-law projection size: {m}")
    """
    if mode == "tensor":
        if not isinstance(unit, torch.Tensor):
            raise TypeError("tensor mode expects a single torch.Tensor")
        return proj_size_for_tensor(unit, strategy, cfg, layer_idx, num_layers, all_dims)
    elif mode == "layer":
        if not isinstance(unit, dict):
            raise TypeError("layer mode expects Dict[str, torch.Tensor]")
        return proj_size_for_layer(unit, strategy, cfg, layer_idx, num_layers, all_dims)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'tensor' or 'layer'")


# ============================================================================
# Usage Example (for documentation)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating different modes and strategies.
    """
    print("=== Projection Size Estimator Demo ===\n")
    
    # Configuration
    cfg = ProjSizeCfg(
        m_min=16,
        f_max=1.0,
        ratio=0.25,
        beta=2.5,
        pow2_round=True,
        pow2_mode="ceil"
    )
    
    # Create sample tensors
    linear_weight = torch.randn(512, 256)  # (out, in)
    conv_weight = torch.randn(64, 32, 3, 3)  # (out, in, kh, kw)
    bias = torch.randn(512)  # 1D tensor
    
    print("1. Tensor Mode - Fixed Strategy")
    m = proj_size_for(linear_weight, mode="tensor", strategy="fixed", cfg=cfg)
    print(f"   Linear weight {linear_weight.shape}: d_last=256, m={m}\n")
    
    print("2. Tensor Mode - Rank Strategy (Stable Rank)")
    m = proj_size_for(linear_weight, mode="tensor", strategy="rank", cfg=cfg)
    r_stable = stable_rank_tensor(linear_weight)
    print(f"   Linear weight: stable_rank={r_stable:.2f}, m={m}\n")
    
    print("3. Layer Mode - Fixed Strategy")
    layer_params = {
        "attn.q_proj.weight": torch.randn(512, 256),
        "attn.k_proj.weight": torch.randn(512, 256),
        "attn.v_proj.weight": torch.randn(512, 256),
        "attn.bias": torch.randn(512)
    }
    m = proj_size_for(layer_params, mode="layer", strategy="fixed", cfg=cfg)
    print(f"   Layer with 3 weights + bias: d_last_max=256, m={m}\n")
    
    print("4. Layer Mode - Rank Strategy (Effective Rank)")
    m = proj_size_for(layer_params, mode="layer", strategy="rank", cfg=cfg)
    r_eff = effective_rank_layer(layer_params)
    print(f"   Layer effective_rank={r_eff:.2f}, m={m}\n")
    
    print("5. Tensor Mode - Random Strategy")
    cfg_random = ProjSizeCfg(m_min=16, f_max=1.0, pow2_round=True, rng=random.Random(42))
    m = proj_size_for(linear_weight, mode="tensor", strategy="random", cfg=cfg_random)
    print(f"   Random projection size (with seed 42): m={m}\n")
    
    print("6. Layer Progressive Strategy - Linear Growth")
    cfg_progressive = ProjSizeCfg(
        start_proj_size=64, 
        end_proj_size=512, 
        growth_mode="linear",
        pow2_round=True
    )
    print("   Progressive sizes (12 layers, linear growth):")
    for i in range(12):
        m = proj_size_for(linear_weight, mode="tensor", strategy="layer_progressive", 
                         cfg=cfg_progressive, layer_idx=i, num_layers=12)
        print(f"     Layer {i:2d}: m={m:4d}")
    print()
    
    print("7. Layer Progressive Strategy - Exponential Growth")
    cfg_exp = ProjSizeCfg(
        start_proj_size=64, 
        end_proj_size=512, 
        growth_mode="exponential",
        pow2_round=True
    )
    print("   Progressive sizes (12 layers, exponential growth):")
    for i in range(12):
        m = proj_size_for(linear_weight, mode="tensor", strategy="layer_progressive", 
                         cfg=cfg_exp, layer_idx=i, num_layers=12)
        print(f"     Layer {i:2d}: m={m:4d}")
    print()
    
    print("=== Demo Complete ===")
