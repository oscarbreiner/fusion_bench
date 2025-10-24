# projection_size_estimator.py
"""
Projection Size Estimator for Fastfood Subspace Merging

This module provides adaptive projection size estimation for subspace merging operations.
Supports two modes:
  - tensor: Per-tensor projection size estimation
  - layer: Per-layer projection size estimation (shared across layer's parameters)

Four strategies:
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
Strategy = Literal["fixed", "random", "rank", "layer_progressive", "layer_group"]
Pow2Mode = Literal["ceil", "floor", "nearest"]
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
    num_layers: int = 1
) -> int:
    """
    Compute projection size for a single tensor.
    
    Args:
        W: Parameter tensor
        strategy: "fixed", "random", "rank", or "layer_progressive"
        cfg: Configuration for projection size estimation
        layer_idx: Current layer index (for layer_progressive strategy)
        num_layers: Total number of layers (for layer_progressive strategy)
        
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
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return clip_and_pow2(m_tgt, cfg.m_min, m_max, cfg.pow2_round, cfg.pow2_mode)


def proj_size_for_layer(
    layer_params: Dict[str, torch.Tensor],
    strategy: Strategy,
    cfg: ProjSizeCfg,
    layer_idx: int = 0,
    num_layers: int = 1
) -> int:
    """
    Compute a single projection size for an entire layer group.
    
    Uses the largest d_last among tensors for capacity bound.
    
    Args:
        layer_params: Dictionary of parameter name -> tensor for a layer
        strategy: "fixed", "random", "rank", or "layer_progressive"
        cfg: Configuration for projection size estimation
        layer_idx: Current layer index (for layer_progressive strategy)
        num_layers: Total number of layers (for layer_progressive strategy)
        
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
    num_layers: int = 1
) -> int:
    """
    Unified entry point for projection size estimation.
    
    Args:
        unit: Either a single tensor (mode="tensor") or dict of tensors (mode="layer")
        mode: "tensor" or "layer"
        strategy: "fixed", "random", "rank", or "layer_progressive"
        cfg: Configuration for projection size estimation
        layer_idx: Current layer index (for layer_progressive strategy)
        num_layers: Total number of layers (for layer_progressive strategy)
        
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
        >>> # Layer progressive strategy
        >>> m = proj_size_for(weight, mode="tensor", strategy="layer_progressive", 
        ...                   cfg=cfg, layer_idx=5, num_layers=12)
        >>> print(f"Progressive projection size at layer 5/12: {m}")
    """
    if mode == "tensor":
        if not isinstance(unit, torch.Tensor):
            raise TypeError("tensor mode expects a single torch.Tensor")
        return proj_size_for_tensor(unit, strategy, cfg, layer_idx, num_layers)
    elif mode == "layer":
        if not isinstance(unit, dict):
            raise TypeError("layer mode expects Dict[str, torch.Tensor]")
        return proj_size_for_layer(unit, strategy, cfg, layer_idx, num_layers)
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
