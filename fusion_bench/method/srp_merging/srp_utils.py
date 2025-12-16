"""
Utility functions for structured random projection (SRP) based subspace merging.
Supports FWHT/SRHT/DCT/DHT/Fastfood transforms and model-merge helpers (TIES/EMA/etc.).

Main API:
- create_projection_ops(...): Build projection (fwd) and lifting (lift) operators
- Supports transforms: FWHT, SRHT (Hadamard), DCT-II, DHT, Fastfood
- Mathematically correct scaling for all transforms

Additional utilities:
- Orthonormal transforms: fwht_inplace_ortho, dct_ortho, dht_ortho
- Aggregation functions: zero_aware_aggregate, ties_merge_subspace, ema_merge_subspace
- Helper functions: layer_key, compute_global_dim, normalize_weights

Legacy compatibility:
- Transform aliases with warnings: {"hadamard","wht"} → "srht"
"""

from __future__ import annotations
import math
import hashlib
import warnings
from typing import List, Tuple, Callable, Literal, Dict

import numpy as np
from scipy.fft import dct as sp_dct, idct as sp_idct

import torch
from torch import Tensor

# Public exports
__all__ = [
    "EPS",
    "create_projection_ops",
    "create_multi_sketch_ops",
    "fwht_inplace_ortho",
    "dct_ortho",
    "idct_ortho",
    "dht_ortho",
    "idht_ortho",
    "zero_aware_aggregate",
    "layer_key",
    "compute_global_dim",
    "normalize_weights",
    # TIES / EMA / advanced merges
    "resolve_zero_signs",
    "resolve_sign",
    "ties_disjoint_merge",
    "ties_merge_subspace",
    "random_merge_subspace",
    "ema_adaptive_beta",
    "ema_merge_subspace",
    "energy_equalize",
    "variance_aware_weights",
    "subspace_whiten",
    "conflict_gating",
    "elect_then_avg",
    "soft_signmax",
    "signmax_mad_normalized",
    "align_weighted_mean",
    "coherence_penalized_weights",
    "orthogonal_deflation_merge",
    "odm_ema_tuning_free",
    "subspace_boosting",
    "generate_tall_masks",
    "apply_consensus_mask",
    # TSV-Merge inspired methods
    "merge_consensus_whiten",
    "merge_spectral_denoising",
    "merge_geometric_median",
    "merge_stack_and_whiten",
]

# Constants
EPS = 1e-12

# ============================================================================
# Core helpers (dimension + seeding)
# ============================================================================

def next_pow2(n: int) -> int:
    """Next power of two >= n."""
    return 1 << (n - 1).bit_length()

def seed_from_string(s: str) -> int:
    """Deterministic seed from string."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")

def _rng_from_seed(seed_key: str, device: torch.device | None = None) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed_from_string(seed_key))
    return g


# ============================================================================
# Orthonormal transforms (FWHT / DCT-II / DHT)
#   - Each F is orthonormal (F^T = F^{-1})
#   - Apply along the LAST dimension
# ============================================================================

@torch.no_grad()
def fwht_inplace_ortho(x: Tensor) -> Tensor:
    # orthonormal FWHT on last dim, preserve leading dims
    *lead, n = x.shape
    if n <= 1:
        return x
    h = 1
    y = x.reshape(-1, n).contiguous()  # keep batch as first dim
    while h < n:
        y = y.view(-1, n // (2*h), 2, h)
        # 🚀 OPTIMIZATION: Avoid unnecessary .clone() - in-place ops safe here
        a = y[..., 0, :]
        b = y[..., 1, :]
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.view(-1, n)
        h *= 2
    y.mul_(1.0 / math.sqrt(n))
    return y.view(*lead, n)

@torch.no_grad()
def dct_ortho(x: Tensor) -> Tensor:
    """
    GPU-accelerated orthonormal DCT-II transform along last dimension.
    Uses torch.fft for efficiency - stays on GPU, no CPU transfers.
    """
    N = x.shape[-1]
    # Use even-odd decomposition via FFT
    # DCT-II can be computed as real part of FFT of even extension
    x_doubled = torch.cat([x, x.flip(-1)], dim=-1)
    X_fft = torch.fft.fft(x_doubled, dim=-1)
    X = X_fft[..., :N].real
    
    # Apply DCT-II orthonormal scaling
    scale = torch.ones(N, dtype=x.dtype, device=x.device)
    scale[0] = 1.0 / math.sqrt(N)
    scale[1:] = math.sqrt(2.0 / N)
    return X * scale

@torch.no_grad()
def idct_ortho(x: Tensor) -> Tensor:
    """
    GPU-accelerated inverse DCT-II (i.e., DCT-III) along last dimension.
    Inverse operation of dct_ortho.
    """
    N = x.shape[-1]
    # Apply inverse scaling (transpose of forward scaling)
    scale = torch.ones(N, dtype=x.dtype, device=x.device)
    scale[0] = math.sqrt(N)
    scale[1:] = math.sqrt(N / 2.0)
    x_scaled = x * scale
    
    # Reconstruct via inverse FFT
    X_doubled = torch.zeros(x.shape[:-1] + (2 * N,), dtype=torch.complex64, device=x.device)
    X_doubled[..., :N] = x_scaled.to(torch.complex64)
    X_doubled[..., N:] = x_scaled.flip(-1).to(torch.complex64)
    x_rec = torch.fft.ifft(X_doubled, dim=-1)[..., :N].real
    return x_rec.to(x.dtype)

@torch.no_grad()
def dht_ortho(x: Tensor) -> Tensor:
    """
    GPU-accelerated orthonormal Discrete Hartley Transform (self-inverse).
    Uses torch.fft: H(x) = Re(FFT(x)) - Im(FFT(x)), staying on GPU.
    """
    N = x.shape[-1]
    X_fft = torch.fft.fft(x, dim=-1)
    # Hartley transform: real minus imaginary parts
    X = (X_fft.real - X_fft.imag) / math.sqrt(N)
    return X

@torch.no_grad()
def idht_ortho(x: Tensor) -> Tensor:
    """Inverse DHT (identical to forward DHT - self-inverse property)."""
    return dht_ortho(x)


# ============================================================================
# Transform factory and projection ops
# ============================================================================

TransformType = Literal["fwht", "srht", "dct", "dht", "fastfood", "none"]

def _normalize_transform_type(t: str | None) -> TransformType | None:
    """
    Normalize transform type string, supporting legacy names with warnings.
    
    Canonical names: "fwht" | "srht" | "dct" | "dht" | "fastfood" | "none" | None
    - "none" or None: Skip projection, merge directly in original space
    Legacy aliases: "hadamard", "wht" → "srht"
    """
    # Handle None or "none" - no transformation
    if t is None or (isinstance(t, str) and t.strip().lower() in {"none", "null", ""}):
        return None
    
    t_norm = t.strip().lower()
    
    # Legacy name mapping with warnings (do NOT alias 'fastfood' anymore)
    legacy_map = {
        "hadamard": "srht",
        "wht": "srht",
    }
    
    if t_norm in legacy_map:
        canon = legacy_map[t_norm]
        warnings.warn(
            f"transform_type='{t}' is deprecated; using '{canon}'. "
            "Please update your configs to one of: fwht | srht | dct | dht | fastfood | none.",
            category=UserWarning,
            stacklevel=3,
        )
        return canon  # type: ignore[return-value]
    
    # Validate canonical names
    if t_norm not in {"fwht", "srht", "dct", "dht", "fastfood", "none"}:
        raise ValueError(
            f"Unknown transform_type='{t}'. Allowed: fwht | srht | dct | dht | fastfood | none (or None)"
        )
    
    return t_norm if t_norm != "none" else None  # type: ignore[return-value]

class OrthoTransform:
    """
    Wraps an orthonormal transform F with:
      - forward: F(x)
      - inverse: F^{-1}(x) == F^T(x)
      - length L (possibly padded vs original D)
      - pad policy for hadamard/fastfood (power-of-2 only)
    """
    def __init__(self, kind: TransformType, D: int, device: torch.device):
        self.kind = kind
        if kind in ("fwht", "srht", "fastfood"):
            self.L = next_pow2(D)   # Hadamard requires power-of-2
            self.F = fwht_inplace_ortho
            self.Finv = fwht_inplace_ortho  # orthonormal => self-inverse
            self.needs_pad = (self.L != D)
        elif kind == "dct":
            self.L = D
            self.F = dct_ortho
            self.Finv = idct_ortho
            self.needs_pad = False
        elif kind == "dht":
            self.L = D
            self.F = dht_ortho
            self.Finv = idht_ortho
            self.needs_pad = False
        else:
            raise ValueError(f"Unknown transform kind: {kind}")
        self.device = device

    def pad(self, x: Tensor, D: int) -> Tensor:
        if self.needs_pad and D < self.L:
            return torch.nn.functional.pad(x, (0, self.L - D))
        return x

def _rand_signs(L: int, device: torch.device) -> Tensor:
    # Diagonal ±1 vector for random sign flips (D in SRHT: P F D x)
    return (torch.randint(0, 2, (L,), dtype=torch.int8, device=device) * 2 - 1).float()

@torch.no_grad()
def create_projection_ops(
    global_dim: int,
    proj_dim: int,
    *,
    transform_type: str | None,   # accepts canonical or legacy names, or None for no transform
    seed_key: str,
    device: torch.device,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """
    Build projection (fwd) and adjoint lifting (lift) operators.

    Modes:
      - None/"none": No transformation - identity operators (merge in original space).
                     y = x,    lift(y) = y  (no projection, no lifting)
      - "fwht": Full orthonormal transform (no subsampling). Requires proj_dim == L.
                y = F (D ⊙ x),    lift(y) = D ⊙ F^{-1}(y)
      - "srht": Subsampled Hadamard sketch (Hadamard + sign flips + row sampling).
                y = √(L/m) * P F (D ⊙ x),    lift(y) = (1/√(L/m)) * (D ⊙ F^{-1}(P^T y))
      - "dct":  Subsampled orthonormal DCT-II sketch (same structure, no padding).
      - "dht":  Subsampled orthonormal Hartley sketch (same structure, no padding).
      - "fastfood": Two Hadamards with permutation & Gaussian diag before subsampling.
                    y = √(L/m) * P H (G ⊙ (Π [ H (B ⊙ x) ]))

    Args:
        global_dim: Original input dimension D.
        proj_dim:   Target projection dimension m (for fwht, must equal L; ignored for None).
        transform_type: "fwht" | "srht" | "dct" | "dht" | "fastfood" | "none" | None.
        seed_key:    Deterministic seed key for reproducibility.
        device:      torch.device for created tensors.

    Returns:
        (fwd, lift) callables.
    """
    kind = _normalize_transform_type(transform_type)
    
    # Handle None transform type - identity operators (no projection)
    if kind is None:
        @torch.no_grad()
        def identity_fwd(x: Tensor) -> Tensor:
            """Identity forward: no transformation, returns input as-is."""
            return x.contiguous()
        
        @torch.no_grad()
        def identity_lift(y: Tensor) -> Tensor:
            """Identity lift: no transformation, returns input as-is."""
            return y.contiguous()
        
        return identity_fwd, identity_lift

    D = int(global_dim)
    tfm = OrthoTransform(kind, D, device)
    L = tfm.L
    m = int(proj_dim)

    # Randomness: signs, generator
    g = _rng_from_seed(seed_key, device=device)
    B = (torch.randint(0, 2, (L,), generator=g, dtype=torch.int8, device=device) * 2 - 1).to(torch.float32)

    if kind == "fwht":
        if m != L:
            raise ValueError(f"FWHT mode requires proj_dim == L ({L}), got {m}")
        scale = 1.0
        row_idx = torch.arange(L, device=device)  # identity P
        Pi = None
        invPi = None
        G = None
    else:
        if not (1 <= m <= L):
            raise ValueError(f"proj_dim (m) must satisfy 1 <= m <= L ({L}), got {m}")
        scale = math.sqrt(L / m)
        row_idx = torch.randperm(L, generator=g, device=device)[:m]  # P: row selector
        # Fastfood-specific randomness:
        if kind == "fastfood":
            Pi = torch.randperm(L, generator=g, device=device)
            invPi = torch.empty(L, dtype=torch.long, device=device)
            invPi[Pi] = torch.arange(L, device=device)
            G = torch.randn(L, generator=g, device=device, dtype=torch.float32)
        else:
            Pi = None
            invPi = None
            G = None

    @torch.no_grad()
    def fwd(xD: Tensor) -> Tensor:
        assert xD.shape[-1] == D, f"Expected last dim {D}, got {xD.shape[-1]}"
        x = tfm.pad(xD, D).to(torch.float32, copy=False)  # [..., L]
        if kind == "fastfood":
            # Fastfood: y = √(L/m) * P * H * (G ⊙ (Π [ H (B ⊙ x) ]))
            x = x * B
            x = tfm.F(x)                  # H (B ⊙ x)
            x = x.index_select(-1, Pi)    # Π H (B ⊙ x)
            x = x * G                     # G ⊙ (...)
            x = tfm.F(x)                  # H G Π H B x
            x = x.index_select(-1, row_idx)
            return (scale * x).contiguous()
        else:
            # SRHT/DCT/DHT (single transform + subsample)
            x = x * B                    # D ⊙ x
            x = tfm.F(x)                 # F (D ⊙ x)
            if kind == "fwht":
                return x.contiguous()    # full transform output [..., L]
            x = x.index_select(dim=-1, index=row_idx)   # P F D x  -> [..., m]
            return (scale * x).contiguous()

    @torch.no_grad()
    def lift(y: Tensor) -> Tensor:
        if kind == "fastfood":
            # Adjoint of fastfood pipeline:
            # x̂ = B ⊙ H ( Π^T [ G ⊙ ( H ( P^T( √(m/L) y ) ) ) ] )
            y_full = (y.to(torch.float32, copy=False) / scale)
            buf = torch.zeros(y_full.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
            buf.index_copy_(dim=-1, index=row_idx, source=y_full)       # P^T y
            z = tfm.Finv(buf)                                           # H^T = H
            z = z * G                                                   # G ⊙ ...
            z = z.index_select(-1, invPi)                               # Π^T
            z = tfm.Finv(z)                                             # H
            z = z * B
            return z[..., :D].contiguous()

        if kind == "fwht":
            y_full = y.to(torch.float32, copy=False)
        else:
            y_full = (y.to(torch.float32, copy=False) / scale)
            buf = torch.zeros(y_full.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
            buf.index_copy_(dim=-1, index=row_idx, source=y_full)  # P^T y
            y_full = buf

        y_full = tfm.Finv(y_full)    # F^{-1}(...)
        y_full.mul_(B)               # D ⊙ ...
        return y_full[..., :D].contiguous()

    return fwd, lift

# ============================================================================
# Multi-Sketch Projection Operators
# ============================================================================

@torch.no_grad()
def create_multi_sketch_ops(
    global_dim: int,
    proj_dim: int,
    *,
    num_sketches: int,
    transform_type: str | None,
    seed_key: str,
    device: torch.device,
) -> Tuple[List[Callable[[Tensor], Tensor]], List[Callable[[Tensor], Tensor]]]:
    """
    Build multiple independent projection and lifting operators for multi-sketching.
    
    Multi-sketching uses J independent random projections A_1, A_2, ..., A_J to
    capture different subspaces of the parameter space. Each sketch has a different
    random seed, so they probe orthogonal directions.
    
    Benefits:
    - Single projection A = s·P·H·D only captures m < d dimensions
    - Everything orthogonal to those rows (nullspace) is lost
    - Multiple sketches with different random seeds cover more of the full space
    - Reconstruction averages across sketches to reduce information loss
    
    Args:
        global_dim: Original input dimension D
        proj_dim: Target projection dimension m (per sketch)
        num_sketches: Number of independent sketches J
        transform_type: "fwht" | "srht" | "dct" | "dht" | "fastfood" | "none" | None
        seed_key: Base seed key (each sketch gets seed_key_j)
        device: torch.device for created tensors
        
    Returns:
        (fwd_ops, lift_ops) where each is a list of J callables
        
    Example usage:
        >>> fwd_ops, lift_ops = create_multi_sketch_ops(512, 128, num_sketches=3, ...)
        >>> # Project with each sketch
        >>> y_sketches = [fwd_j(x) for fwd_j in fwd_ops]
        >>> # ... merge each sketch separately ...
        >>> # Lift and ensemble
        >>> x_lifted = [lift_j(y_j) for lift_j, y_j in zip(lift_ops, y_merged_sketches)]
        >>> x_final = torch.stack(x_lifted, dim=0).mean(dim=0)
    """
    fwd_ops = []
    lift_ops = []
    
    for j in range(num_sketches):
        # Create unique seed for each sketch
        sketch_seed = f"{seed_key}_sketch_{j}"
        
        # Create independent projection operators
        fwd_j, lift_j = create_projection_ops(
            global_dim=global_dim,
            proj_dim=proj_dim,
            transform_type=transform_type,
            seed_key=sketch_seed,
            device=device,
        )
        
        fwd_ops.append(fwd_j)
        lift_ops.append(lift_j)
    
    return fwd_ops, lift_ops

# ============================================================================
# TIES / EMA / Advanced aggregation (unchanged)
# ============================================================================

@torch.no_grad()
def resolve_zero_signs(sign_to_mult: Tensor, method: str = "majority") -> Tensor:
    majority_sign = torch.sign(sign_to_mult.sum())
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult

@torch.no_grad()
def resolve_sign(v: Tensor) -> Tensor:
    sign_to_mult = torch.sign(v.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

@torch.no_grad()
def ties_disjoint_merge(v: Tensor, merge_func: str, sign_to_mult: Tensor) -> Tensor:
    merge_func = merge_func.split("-")[-1]
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, v > 0, v < 0)
        selected_entries = v * rows_to_keep
    else:
        rows_to_keep = v != 0
        selected_entries = v * rows_to_keep
    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"TIES merge method {merge_func} is not defined.")
    return disjoint_aggs

@torch.no_grad()
def ties_merge_subspace(U: Tensor, ties_merge_func: str = "sum", weights: List[float] | None = None) -> Tensor:
    K = U.shape[0]
    if K == 0:
        raise ValueError("No task vectors to merge")
    if K == 1:
        w = weights[0] if weights else 1.0
        return w * U[0]
    orig_shape = U.shape[1:]
    U_flat = U.view(K, -1)
    if weights is not None:
        w_tensor = torch.tensor(weights, dtype=U_flat.dtype, device=U_flat.device).view(K, 1)
        U_flat = w_tensor * U_flat
    print("RESOLVING SIGN (TIES)")
    final_signs = resolve_sign(U_flat)
    print(f"DISJOINT AGGREGATION (TIES): {ties_merge_func}")
    merged_tv = ties_disjoint_merge(U_flat, ties_merge_func, final_signs)
    return merged_tv.view(orig_shape)

@torch.no_grad()
def ema_adaptive_beta(z_acc: Tensor, z_new: Tensor, gamma: float = 1.2, w_c: float = 0.6, w_s: float = 0.4, eps: float = 1e-8) -> float:
    if z_acc.norm() < eps:
        return 0.0
    c = (z_acc @ z_new) / (z_acc.norm() * z_new.norm() + eps)
    s = z_acc.norm() / (z_new.norm() + eps)
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
    K = U.shape[0]
    if K == 0:
        raise ValueError("No task vectors to merge")
    if K == 1:
        w = weights[0] if weights else 1.0
        return w * U[0]
    orig_shape = U.shape[1:]
    U_flat = U.view(K, -1)
    if task_order == "random":
        indices = torch.randperm(K).tolist()
    elif task_order == "custom":
        if custom_order is None:
            raise ValueError("custom_order must be provided when task_order='custom'")
        if len(custom_order) != K or set(custom_order) != set(range(K)):
            raise ValueError(f"custom_order must be a permutation of [0,1,...,{K-1}], got {custom_order}")
        indices = custom_order
    elif task_order == "cosine_similarity":
        indices = [0]
        z_acc = U_flat[0].clone()
        remaining = set(range(1, K))
        while remaining:
            best_idx = None
            best_sim = -2.0
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
                z_acc = torch.stack([U_flat[i] for i in indices]).mean(dim=0)
            else:
                best_idx = next(iter(remaining))
                indices.append(best_idx)
                remaining.remove(best_idx)
    else:
        indices = list(range(K))

    z_acc = torch.zeros_like(U_flat[0])
    for task_idx in indices:
        z_new = U_flat[task_idx]
        if weights is not None:
            z_new = weights[task_idx] * z_new
        beta = ema_adaptive_beta(z_acc, z_new, ema_gamma, ema_w_c, ema_w_s)
        z_acc = beta * z_acc + (1 - beta) * z_new
    return z_acc.view(orig_shape)

@torch.no_grad()
def energy_equalize(U: Tensor, eps: float = 1e-8) -> Tensor:
    K, M = U.shape
    norms = U.norm(dim=1) + eps
    target = norms.median()
    scale = (target / norms).view(K, 1)
    return U * scale

@torch.no_grad()
def variance_aware_weights(U: Tensor, eps: float = 1e-8) -> Tensor:
    var = U.var(dim=0, unbiased=False)
    w = 1.0 / (var + eps)
    w = w / (w.mean() + eps)
    return w

@torch.no_grad()
def subspace_whiten(U: Tensor, eps: float = 1e-6) -> Tensor:
    K, M = U.shape
    Uc = U - U.mean(dim=0, keepdim=True)
    G = (Uc @ Uc.t()) / max(M, 1)
    eigvals, eigvecs = torch.linalg.eigh(G)
    inv_sqrt = (eigvals.clamp_min(eps)).rsqrt()
    W = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.t()
    return W @ Uc

@torch.no_grad()
def conflict_gating(U: Tensor, tau: float = 0.6, eps: float = 1e-8) -> Tensor:
    sgn = torch.sign(U)
    mean_sign = sgn.sum(dim=0) / ((sgn != 0).sum(dim=0).clamp_min(1))
    mask = (mean_sign.abs() >= tau).float()
    return U * mask.unsqueeze(0)

@torch.no_grad()
def elect_then_avg(U: Tensor) -> Tensor:
    sgn_sum = torch.sign(U).sum(dim=0)
    elected = torch.where(sgn_sum >= 0, 1.0, -1.0)
    match = (torch.sign(U) == elected.unsqueeze(0))
    denom = match.sum(dim=0).clamp_min(1).float()
    return (U * match).sum(dim=0) / denom

@torch.no_grad()
def soft_signmax(U: Tensor, T: float = 0.25) -> Tensor:
    K, M = U.shape
    sgn_sum = torch.sign(U).sum(dim=0)
    elected = torch.where(sgn_sum >= 0, 1.0, -1.0)
    same = (torch.sign(U) == elected.unsqueeze(0))
    scores = U.abs() / max(T, 1e-6)
    scores[~same] = -float("inf")
    W = torch.softmax(scores, dim=0)
    W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    return (W * U).sum(dim=0)

@torch.no_grad()
def signmax_mad_normalized(U: Tensor, T: float = 0.5, eps: float = 1e-8) -> Tensor:
    """
    SignMax with MAD normalization and temperature-controlled softmax voting.
    
    Prevents high-magnitude experts from dominating by normalizing each expert
    by its Median Absolute Deviation (MAD) before voting. More robust than
    standard normalization when experts have different scales or outliers.
    
    Algorithm:
        1. Normalize each expert by MAD: U_norm[k] = U[k] / (MAD(U[k]) + eps)
        2. Elect sign by majority voting on normalized experts
        3. Compute temperature-scaled scores for sign-consistent experts
        4. Weight original (unnormalized) experts by softmax of scores
    
    Args:
        U: Stacked expert tensors [K, M] where K = num_experts, M = num_parameters
        T: Temperature parameter for softmax (lower = more selective)
           - Small T (e.g., 0.1): Pick single dominant expert (arg-max like)
           - Medium T (e.g., 0.25): Balanced weighting
           - Large T (e.g., 1.0): Nearly uniform weighting
        eps: Small constant to avoid division by zero
        
    Returns:
        Merged expert tensor [M] with MAD-normalized voting and temperature-scaled weighting
        
    Benefits vs standard SignMax:
        - MAD normalization is robust to outliers (vs mean/std)
        - Equal voting power for all experts regardless of magnitude
        - Temperature control over selection vs averaging trade-off
        - Fully vectorized (no Python loops)
        
    Example:
        >>> U = torch.randn(5, 1000)  # 5 experts, 1000 parameters
        >>> merged = signmax_mad_normalized(U, T=0.25)  # Balanced temperature
    """
    K, M = U.shape
    
    # Step 1: Compute MAD for each expert (vectorized)
    # MAD = median(|X - median(X)|)
    medians = U.median(dim=1, keepdim=True)[0]  # [K, 1]
    abs_deviations = (U - medians).abs()  # [K, M]
    mad = abs_deviations.median(dim=1, keepdim=True)[0]  # [K, 1]
    
    # Normalize by MAD
    U_norm = U / (mad + eps)  # [K, M]
    
    # Step 2: Elect sign by majority voting on normalized experts
    sgn_sum = torch.sign(U_norm).sum(dim=0)  # [M]
    elected = torch.where(sgn_sum >= 0, 1.0, -1.0)  # [M]
    
    # Step 3: Identify sign-consistent experts
    same = (torch.sign(U_norm) == elected.unsqueeze(0))  # [K, M]
    
    # Step 4: Compute temperature-scaled scores based on vote margin
    # Vote margin = how strongly each expert agrees with elected sign (signed magnitude)
    # For elected=+1: margin = U_norm (positive values favor election)
    # For elected=-1: margin = -U_norm (negative values favor election, flipped to positive)
    # Softmax temperature: softmax(margin/T) - smaller T = peakier, larger T = more uniform
    vote_margin = U_norm * elected.unsqueeze(0)  # [K, M] - positive when agreeing with election
    scores = vote_margin / max(T, 1e-6)  # [K, M]
    scores[~same] = -float("inf")  # Mask out sign-inconsistent experts
    
    # Step 5: Softmax weighting
    W = torch.softmax(scores, dim=0)  # [K, M]
    W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Step 6: Apply weights to ORIGINAL (unnormalized) experts
    # This preserves the actual parameter magnitudes
    return (W * U).sum(dim=0)  # [M]

@torch.no_grad()
def align_weighted_mean(U: Tensor, gamma: float = 2.0, clip_neg: bool = True, eps: float = 1e-8) -> Tensor:
    K, M = U.shape
    c = U.mean(dim=0)
    c_norm = c.norm() + eps
    u_norms = U.norm(dim=1) + eps
    cos = (U @ c) / (u_norms * c_norm)
    if clip_neg:
        cos = cos.clamp_min(0.0)
    w = (cos.clamp_min(0.0)) ** gamma + eps
    w = w / w.sum()
    return (w.view(K, 1) * U).sum(dim=0)

@torch.no_grad()
def coherence_penalized_weights(U: Tensor, lam: float = 1e-3) -> Tensor:
    K, M = U.shape
    G = U @ U.t()
    I = torch.eye(K, device=U.device, dtype=U.dtype)
    b = torch.ones(K, 1, device=U.device, dtype=U.dtype)
    w = torch.linalg.solve(G + lam * I, b).squeeze(1)
    w = torch.clamp(w, min=0.0)
    w = w / (w.sum() + 1e-8)
    return (w.view(K, 1) * U).sum(dim=0)

@torch.no_grad()
def orthogonal_deflation_merge(U: Tensor, weight_by_residual: bool = True, eps: float = 1e-8) -> Tensor:
    K, M = U.shape
    Q = []
    alphas = []
    for k in range(K):
        r = U[k]
        for q in Q:
            r = r - (r @ q) * q
        nr = r.norm()
        if nr > eps:
            q = r / nr
            Q.append(q)
            alphas.append(nr if weight_by_residual else 1.0)
    if not Q:
        return torch.zeros(M, dtype=U.dtype, device=U.device)
    A = torch.tensor(alphas, dtype=U.dtype, device=U.device).view(-1, 1)
    Qmat = torch.stack(Q, dim=0)
    return (A * Qmat).sum(dim=0)

# ---------------- Tuning-free ODM-EMA ----------------

@torch.no_grad()
def _tf_svd_energy_basis(H: Tensor) -> Tensor | None:
    if H is None or H.numel() == 0 or H.shape[1] == 0:
        return None
    U, S, _ = torch.linalg.svd(H, full_matrices=False)
    if S.numel() == 0:
        return None
    e = S.pow(2)
    tot = float(e.sum())
    if tot <= 0.0:
        return None
    cume = torch.cumsum(e, dim=0)
    r = int(torch.searchsorted(cume, 0.5 * tot)) + 1
    q = max(1, min(r, 16, U.shape[1]))
    return U[:, :q]

@torch.no_grad()
def _tf_odm_residual(z_new: Tensor, history: list[Tensor]) -> Tensor:
    if not history:
        return z_new
    H = torch.stack(history, dim=1)
    Q = _tf_svd_energy_basis(H)
    if Q is None:
        return z_new
    return z_new - Q @ (Q.T @ z_new)

@torch.no_grad()
def odm_ema_tuning_free(U: Tensor) -> Tensor:
    K = U.shape[0]
    if K == 0:
        raise ValueError("No task vectors to merge.")
    if K == 1:
        return U[0]
    orig_shape = U.shape[1:]
    Z = U.reshape(K, -1).to(torch.float32)
    D = Z.shape[1]
    z_acc = torch.zeros(D, dtype=Z.dtype, device=Z.device)
    history: list[Tensor] = []
    for t in range(K):
        z_new = Z[t]
        r_t = _tf_odm_residual(z_new, history)
        if t == 0:
            z_acc = r_t
        else:
            w_old = math.sqrt(t / (t + 1.0))
            w_new = 1.0 / math.sqrt(t + 1.0)
            z_acc = w_old * z_acc + w_new * r_t
        if r_t.norm() > 1e-6:
            history.append(r_t.detach().clone())
            if len(history) > 32:
                history.pop(0)
    return z_acc.reshape(orig_shape)

# ============================================================================
# Subspace Boosting (SB)
# ============================================================================

@torch.no_grad()
def subspace_boosting(param: Tensor, beta: float, eps: float = 1e-12) -> Tensor:
    """
    Restore rank to collapsed task-vector matrices by boosting neglected singular values.
    
    When merging many experts, task-vector matrices often collapse in rank with most
    energy concentrated in a few singular directions. This function boosts the smaller
    singular values to restore a fuller-rank update.
    
    Algorithm:
        1. Compute SVD: T = U Σ V^T
        2. Compute cumulative energy: energy(k) = Σ(σ_i, i=1..k) / Σ(σ_i, all)
        3. Find cutoff index: k* = min{k : energy(k) >= beta}
        4. Boost singular values: σ_i' = max(σ_i, σ_k*)
        5. Reconstruct: T' = U Σ' V^T
    
    Args:
        param: Merged task-vector weight matrix (2D tensor)
        beta: Fraction of cumulative singular value energy to preserve (0.0-1.0)
              Typical range: 0.00-0.02
              - Small beta: aggressive boosting (more rank restored)
              - Large beta: mild boosting
        eps: Small constant to avoid division by zero
    
    Returns:
        Rank-boosted parameter tensor with same shape as input
    
    Example:
        >>> merged_weight = merge_task_vectors([tv1, tv2, tv3])  # Shape: (out_dim, in_dim)
        >>> boosted_weight = subspace_boosting(merged_weight, beta=0.01)
        >>> # Now boosted_weight has restored rank with smaller singular values boosted
    """
    if param.ndim != 2:
        # Only apply to 2D matrices (linear layer weights)
        return param
    
    if param.numel() == 0:
        return param
    
    # Compute SVD
    U, S, Vh = torch.linalg.svd(param, full_matrices=False)
    
    if S.numel() == 0:
        return param
    
    # Compute cumulative energy
    total_energy = S.sum() + eps
    cumulative_energy = torch.cumsum(S, dim=0)
    normalized_energy = cumulative_energy / total_energy
    
    # Find cutoff index (first index where cumulative energy >= beta)
    # Clamp beta to valid range
    beta_clamped = max(0.0, min(1.0, beta))
    above_threshold = (normalized_energy >= beta_clamped).nonzero(as_tuple=False)
    
    if above_threshold.numel() == 0:
        # No singular values meet threshold, return original
        return param
    
    cutoff_idx = above_threshold[0].item()
    cutoff_value = S[cutoff_idx]
    
    # Boost singular values (clamp to cutoff)
    S_boosted = torch.clamp(S, min=cutoff_value)
    
    # Reconstruct boosted matrix
    # param_boosted = U @ diag(S_boosted) @ Vh
    # Efficient: (U * S_boosted) @ Vh
    param_boosted = (U * S_boosted.unsqueeze(0)) @ Vh
    
    return param_boosted

# ============================================================================
# TALL Masks (Task-Adaptive Low-rank Learning)
# ============================================================================

@torch.no_grad()
def generate_tall_masks(
    merged_delta: Dict[str, Tensor],
    individual_deltas: Dict[str, Dict[str, Tensor]],
    base_state: Dict[str, Tensor],
    tall_mask_lambda: float = 0.6,
) -> Dict[str, Dict[str, Tensor]]:
    """
    Generate task-specific TALL masks for parameter selection.
    
    TALL masks identify which parameters in the merged model are most relevant for each task
    by comparing distances in parameter space:
    
    For each task t:
        mask_t = |θ_0 - θ_t| > |θ_merged - θ_t| * λ
    
    Intuition:
        - If a parameter changed a lot from pretrained (|θ_0 - θ_t| large)
        - But is similar to the merged model (|θ_merged - θ_t| small)
        - Then this parameter is important for task t → include in mask
    
    Args:
        merged_delta: Merged task vector (after merging, boosting, etc.)
        individual_deltas: Dictionary mapping task names to their individual task vectors
        base_state: Base/pretrained model state dict
        tall_mask_lambda: Threshold parameter (typical: 0.2-0.6)
                         - Smaller λ: more permissive masks (more parameters kept)
                         - Larger λ: stricter masks (fewer parameters kept)
    
    Returns:
        Dictionary mapping task names to their TALL masks (same structure as state dict)
        Each mask is a binary tensor indicating which parameters are active for that task
    
    Example:
        >>> tall_masks = generate_tall_masks(merged_delta, individual_deltas, base_state, lambda=0.6)
        >>> # tall_masks['mnist']['vision_model.encoder.layers.0.self_attn.q_proj.weight'] = Tensor([True, False, ...])
    """
    tall_masks = {}
    
    # Compute merged model state: base + merged_delta
    merged_state = {k: base_state[k] + merged_delta[k] for k in merged_delta.keys()}
    
    for task_name, task_delta in individual_deltas.items():
        task_mask = {}
        
        # Compute fine-tuned state for this task: base + task_delta
        task_state = {k: base_state[k] + task_delta[k] for k in task_delta.keys()}
        
        for param_name in merged_delta.keys():
            if param_name not in task_delta:
                continue
            
            # Distance from pretrained to task-specific
            diff_base_task = (base_state[param_name] - task_state[param_name]).abs()
            
            # Distance from merged to task-specific
            diff_merged_task = (merged_state[param_name] - task_state[param_name]).abs()
            
            # TALL mask: keep parameters where task-specific change is large
            # but distance to merged model is small (scaled by lambda)
            mask = diff_base_task > (diff_merged_task * tall_mask_lambda)
            
            task_mask[param_name] = mask.float()
        
        tall_masks[task_name] = task_mask
    
    return tall_masks

@torch.no_grad()
def apply_consensus_mask(
    merged_delta: Dict[str, Tensor],
    tall_masks: Dict[str, Dict[str, Tensor]],
    consensus_threshold: int = 2,
) -> Dict[str, Tensor]:
    """
    Apply consensus masking to filter merged parameters based on cross-task agreement.
    
    Consensus masking removes parameters that are not important across multiple tasks:
    - Catastrophic weights: parameters not activated by any task (threshold > num_tasks)
    - Selfish weights: parameters activated by only one task (threshold = 2)
    - Keep shared weights: parameters activated by >= threshold tasks
    
    Algorithm:
        For each parameter position:
            count = number of tasks with mask=1 at this position
            consensus_mask = (count >= consensus_threshold)
        merged_delta = merged_delta ⊙ consensus_mask
    
    Args:
        merged_delta: Merged task vector to be filtered
        tall_masks: Dictionary of per-task TALL masks from generate_tall_masks()
        consensus_threshold: Minimum number of tasks that must agree (typical: 1-2)
                            - 0: No filtering (keep all parameters)
                            - 1: Remove only catastrophic weights (not used by any task)
                            - 2: Remove catastrophic + selfish weights (used by <2 tasks)
                            - >num_tasks: Remove everything (zero-shot baseline)
    
    Returns:
        Filtered merged_delta with consensus mask applied
        Parameters not meeting threshold are set to 0
    
    Example:
        >>> # Remove parameters not used by at least 2 tasks
        >>> filtered_delta = apply_consensus_mask(merged_delta, tall_masks, threshold=2)
    """
    consensus_mask = {}
    
    # Initialize consensus counter for each parameter
    for param_name in merged_delta.keys():
        consensus_mask[param_name] = torch.zeros_like(merged_delta[param_name])
    
    # Count how many tasks activate each parameter
    for task_name, task_mask in tall_masks.items():
        for param_name, mask in task_mask.items():
            if param_name in consensus_mask:
                consensus_mask[param_name] += mask
    
    # Apply threshold: keep only parameters activated by >= threshold tasks
    for param_name in consensus_mask.keys():
        consensus_mask[param_name] = (consensus_mask[param_name] >= consensus_threshold).float()
    
    # Apply consensus mask to merged delta
    filtered_delta = {}
    for param_name, delta in merged_delta.items():
        if param_name in consensus_mask:
            filtered_delta[param_name] = delta * consensus_mask[param_name]
        else:
            filtered_delta[param_name] = delta
    
    return filtered_delta

# ============================================================================
# Zero-Aware Aggregation Functions
# ============================================================================

@torch.no_grad()
def random_merge_subspace(U: Tensor) -> Tensor:
    """
    Randomly select one task vector per parameter position.
    
    Args:
        U: (K, m) or (K, d) tensor where K = num tasks
        
    Returns:
        (m,) or (d,) tensor - randomly selected task vector per position
    """
    K = U.shape[0]
    if K == 0:
        return torch.zeros_like(U[0])
    
    # Generate random indices for each position
    random_indices = torch.randint(0, K, (U.shape[1],), device=U.device)
    
    # Select random task vector for each position
    result = U[random_indices, torch.arange(U.shape[1], device=U.device)]
    
    return result


# ============================================================================
# Advanced Aggregation Methods (TSV-Merge Inspired)
# ============================================================================

@torch.no_grad()
def merge_consensus_whiten(U: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Approach 1: Consensus-Whitened Residuals (CWR)
    Fixes interference by orthogonalizing ONLY the disagreements (residuals),
    preserving the shared knowledge (consensus).
    
    Best for: General purpose. Best balance of stability (keeps consensus) and interference reduction.
    """
    # Handle both 2D (K, M) and 3D (K, take, M) shapes
    orig_shape = U.shape
    if U.ndim == 3:
        # Flatten batch dimension: (K, take, M) -> (K, take*M)
        K, take, M = U.shape
        U_flat = U.reshape(K, -1)
    else:
        U_flat = U
        K = U_flat.shape[0]
    
    # 1. Compute Consensus (Mean)
    consensus = U_flat.mean(dim=0)
    
    # 2. Compute Residuals
    residuals = U_flat - consensus.unsqueeze(0)
    
    # 3. Whiten Residuals (Row/Task Covariance)
    # Covariance is K x K (very small in SRP)
    cov = (residuals @ residuals.mT) / (residuals.shape[1] - 1 + eps)
    
    # Eigen-decomposition (safer than Cholesky for ill-conditioned matrices)
    vals, vecs = torch.linalg.eigh(cov)
    
    # Whitening Matrix: W = E * D^(-0.5) * E.T
    inv_sqrt = (vals.clamp_min(eps)).rsqrt().unsqueeze(0)
    W = (vecs * inv_sqrt) @ vecs.mT
    
    # 4. Rotate residuals to be orthogonal
    whitened_residuals = W @ residuals
    
    # 5. Aggregate: Consensus + Mean of Orthogonalized Residuals
    # Using mean() here maintains the scale of the original updates.
    result = consensus + whitened_residuals.mean(dim=0)
    
    # Restore original shape if needed
    if len(orig_shape) == 3:
        result = result.view(orig_shape[1], orig_shape[2])
    
    return result


@torch.no_grad()
def merge_spectral_denoising(U: Tensor, energy_thresh: float = 0.95) -> Tensor:
    """
    Approach 2: Spectral Denoising Merge (SDM)
    Uses SVD to identify and remove "noise" components (tail energy) 
    introduced by random projections or weak task interference.
    
    Best for: High noise / many tasks. When merging 10+ tasks where random noise accumulation is a risk.
    """
    # Handle both 2D (K, M) and 3D (K, take, M) shapes
    orig_shape = U.shape
    if U.ndim == 3:
        # Flatten batch dimension: (K, take, M) -> (K, take*M)
        K, take, M = U.shape
        U_flat = U.reshape(K, -1)
    else:
        U_flat = U
    
    # Transpose to (M, K) for standard SVD interpretation (Features x Samples)
    Z = U_flat.mT  # Use .mT instead of .T to avoid deprecation warning
    L, S, R = torch.linalg.svd(Z, full_matrices=False)
    
    # Compute cumulative energy of singular values (S is 1D)
    energies = S.pow(2)
    total_energy = energies.sum()
    cum_energy = torch.cumsum(energies, dim=0) / total_energy
    
    # Hard Threshold: Keep components explaining X% variance
    # Searchsorted finds the index where condition is first met
    k = int(torch.searchsorted(cum_energy, energy_thresh).item()) + 1
    k = min(k, len(S))
    
    # Reconstruct denoised signal
    # Z_clean = L_k * S_k * R_k.T
    Z_clean = (L[:, :k] * S[:k].unsqueeze(0)) @ R[:, :k].mT
    
    # Return mean of denoised tasks (average across columns)
    result = Z_clean.mean(dim=1)
    
    # Restore original shape if needed
    if len(orig_shape) == 3:
        result = result.view(orig_shape[1], orig_shape[2])
    
    return result


@torch.no_grad()
def merge_geometric_median(U: Tensor, max_iter: int = 20, eps: float = 1e-6) -> Tensor:
    """
    Approach 3: Geometric Median (Weiszfeld's Algorithm)
    Robust aggregation that ignores outliers (tasks that are far from the group).
    
    Best for: Outlier rejection. If one task model is "broken" or has exploded gradients.
    """
    # Handle both 2D (K, M) and 3D (K, take, M) shapes
    orig_shape = U.shape
    if U.ndim == 3:
        # Flatten batch dimension: (K, take, M) -> (K, take*M)
        K, take, M = U.shape
        U_flat = U.reshape(K, -1)
    else:
        U_flat = U
    
    # Initial guess: Euclidean mean
    gm = U_flat.mean(dim=0)
    
    for _ in range(max_iter):
        # Distances from current median to all points: ||u_i - gm||
        diff = U_flat - gm.unsqueeze(0)
        norms = torch.norm(diff, dim=1)
        
        # Weiszfeld weights = 1 / distance
        # Clamp distance to avoid division by zero
        weights = 1.0 / torch.clamp(norms, min=eps)
        weights = weights / weights.sum()
        
        # Weighted average update
        new_gm = (weights.unsqueeze(1) * U_flat).sum(dim=0)
        
        # Check convergence
        if torch.norm(new_gm - gm) < eps:
            break
        gm = new_gm
    
    # Restore original shape if needed
    if len(orig_shape) == 3:
        gm = gm.view(orig_shape[1], orig_shape[2])
        
    return gm


@torch.no_grad()
def merge_stack_and_whiten(
    U: Tensor, 
    mode: str = "procrustes", 
    eps: float = 1e-6,
    return_mean: bool = False,
    newton_iter: int = 5
) -> Tensor:
    """
    Approaches 4 & 5: Stack -> Whiten -> Aggregate
    Implements the TSV-Merge paper logic.
    
    Modes:
      - 'procrustes': SVD-based (Paper Exact). Best for: Maximum interference reduction.
      - 'newton_schulz': SVD-Free Iterative. Best for: Speed / GPU efficiency.
    
    Args:
        U: (K, M) or (K, take, M) task vectors in subspace
        mode: 'procrustes' or 'newton_schulz'
        eps: Small constant for numerical stability
        return_mean: If True, return mean of whitened vectors; if False, return sum (paper default)
        newton_iter: Number of Newton-Schulz iterations (only used if mode='newton_schulz')
    
    Returns:
        Aggregated vector (M,) or (take, M)
    """
    # Handle both 2D (K, M) and 3D (K, take, M) shapes
    orig_shape = U.shape
    if U.ndim == 3:
        # Flatten batch dimension: (K, take, M) -> (K, take*M)
        K, take, M = U.shape
        U_flat = U.reshape(K, -1)
    else:
        U_flat = U
        K = U_flat.shape[0]
    
    if mode == "procrustes":
        # Approach 4: Paper Exact via SVD
        # U = L @ S @ R.T  -->  U_white = L @ R.T
        L, S, R = torch.linalg.svd(U_flat, full_matrices=False)
        U_white = L @ R 
        
    elif mode == "newton_schulz":
        # Approach 5: SVD-Free Iterative Procrustes
        # 1. Normalize by Frobenius norm to ensure singular values < sqrt(3)
        # This guarantees convergence of the Newton iteration.
        norm = U_flat.norm(p='fro') + eps
        X = U_flat.div(norm)
        
        # Identity for iteration
        I = torch.eye(K, device=U.device, dtype=U.dtype)
        
        # 2. Iteration: X_{k+1} = 0.5 * X_k * (3I - X_k^T * X_k) (for Col Ortho)
        # We want Row Orthogonality (Tasks), so we adapt:
        # X_{k+1} = 0.5 * (3I - X_k * X_k^T) * X_k
        for _ in range(newton_iter):
            A = X @ X.mT  # Covariance-like
            X = 0.5 * (3 * I - A) @ X
            
        # X converges to the polar factor U_white (scaled by magnitude removed)
        U_white = X

    else:
        # Fallback to ZCA if requested specifically
        # (Standard ZCA: G^-0.5 @ U)
        G = U_flat @ U_flat.mT
        evals, evecs = torch.linalg.eigh(G)
        inv_sqrt = (evals.clamp_min(eps)).rsqrt().unsqueeze(0)
        W = (evecs * inv_sqrt) @ evecs.mT
        U_white = W @ U_flat

    # Aggregation Step
    # The paper (Eq 5) uses SUM.
    # However, MEAN is often safer for hyperparameter-free stability.
    if return_mean:
        result = U_white.mean(dim=0)
    else:
        # Paper default: Summing unit-strength orthogonal vectors
        result = U_white.sum(dim=0)
    
    # Restore original shape if needed
    if len(orig_shape) == 3:
        result = result.view(orig_shape[1], orig_shape[2])
    
    return result


@torch.no_grad()
def zero_aware_aggregate(
    U: Tensor,
    merge_func: str,
    weights: List[float] | None = None,
    **kwargs
) -> Tensor:
    K = U.shape[0]
    mf = merge_func.lower()
    valid_funcs = {
        "sum", "mean", "max", "signmax", "ema", "random",
        "ties_sum", "ties_mean", "ties_max",
        "energy_equalize", "variance_aware", "subspace_whiten",
        "conflict_gating", "elect_then_avg", "soft_signmax", "signmax_mad_normalized",
        "align_weighted_mean", "coherence_penalized", "orthogonal_deflation",
        "odm_ema_tuning_free",
        "consensus_whiten", "spectral_denoise", "geometric_median",
        "stack_whiten_procrustes", "stack_whiten_newton"
    }
    if mf not in valid_funcs:
        raise ValueError(f"merge_func={merge_func} not in {valid_funcs}")

    # --- New Advanced Methods (TSV-Merge Inspired) ---
    
    if mf == "consensus_whiten":
        return merge_consensus_whiten(U, eps=kwargs.get("eps", 1e-6))
    
    if mf == "spectral_denoise":
        return merge_spectral_denoising(U, energy_thresh=kwargs.get("sdm_energy", 0.95))
    
    if mf == "geometric_median":
        return merge_geometric_median(U, max_iter=kwargs.get("gm_iter", 20), eps=kwargs.get("eps", 1e-6))
    
    if mf == "stack_whiten_procrustes":
        # Defaults to SUM (return_mean=False) to match paper Eq. 5
        return merge_stack_and_whiten(U, mode="procrustes", return_mean=False, eps=kwargs.get("eps", 1e-6))
    
    if mf == "stack_whiten_newton":
        return merge_stack_and_whiten(
            U, 
            mode="newton_schulz", 
            return_mean=False, 
            newton_iter=kwargs.get("newton_iter", 5),
            eps=kwargs.get("eps", 1e-6)
        )

    # --- Existing Advanced Methods ---
    
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

    if mf.startswith("ties_"):
        ties_func = mf.split("_", 1)[1]
        return ties_merge_subspace(U, ties_merge_func=ties_func, weights=weights)

    if mf == "odm_ema_tuning_free":
        return odm_ema_tuning_free(U)

    if mf == "sum":
        # Sum should NOT use normalized weights - just sum the task vectors directly
        # This is the key difference from 'mean': sum adds all task vectors,
        # while mean computes the weighted average
        return U.sum(dim=0)

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
    if mf == "max":
        idx = absU.argmax(dim=0)
        out = torch.gather(U, dim=0, index=idx.unsqueeze(0)).squeeze(0)
        all_zero = (absU.sum(dim=0) == 0)
        if all_zero.any():
            out = out.masked_fill(all_zero, 0.0)
        
        # Log decisions if callback provided
        decision_callback = kwargs.get("decision_callback", None)
        if decision_callback is not None:
            # Count wins for each task (aggregate across all parameters in this tensor)
            winner_counts = torch.bincount(idx.flatten().cpu(), minlength=U.shape[0])
            decision_callback(winner_counts.tolist())
        
        return out

    if mf == "signmax":
        sgn = torch.sign(U)
        pos_count = (sgn > 0).sum(dim=0)
        neg_count = (sgn < 0).sum(dim=0)
        dom_pos = pos_count > neg_count
        dom_neg = (sgn < 0).sum(dim=0) > pos_count
        tie_or_none = ~(dom_pos | dom_neg)
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
        
        # Log decisions if callback provided
        decision_callback = kwargs.get("decision_callback", None)
        if decision_callback is not None:
            # Count wins for each task (aggregate across all parameters in this tensor)
            winner_counts = torch.bincount(idx.flatten().cpu(), minlength=U.shape[0])
            decision_callback(winner_counts.tolist())
        
        return out

    if mf == "random":
        # Randomly select one task vector per parameter position
        return random_merge_subspace(U)

    # Advanced functions need flattening
    orig_shape = U.shape[1:]
    U_flat = U.view(K, -1)

    if mf == "energy_equalize":
        U_eq = energy_equalize(U_flat, eps=kwargs.get("eps", 1e-8))
        return U_eq.mean(dim=0).view(orig_shape)

    if mf == "variance_aware":
        var_weights = variance_aware_weights(U_flat, eps=kwargs.get("eps", 1e-8))
        return (U_flat.mean(dim=0) * var_weights).view(orig_shape)

    if mf == "subspace_whiten":
        U_white = subspace_whiten(U_flat, eps=kwargs.get("eps", 1e-6))
        return U_white.mean(dim=0).view(orig_shape)

    if mf == "conflict_gating":
        tau = kwargs.get("conflict_tau", 0.6)
        U_gated = conflict_gating(U_flat, tau=tau, eps=kwargs.get("eps", 1e-8))
        return U_gated.mean(dim=0).view(orig_shape)

    if mf == "elect_then_avg":
        return elect_then_avg(U_flat).view(orig_shape)

    if mf == "soft_signmax":
        T = kwargs.get("soft_signmax_temperature", 0.25)
        return soft_signmax(U_flat, T=T).view(orig_shape)

    if mf == "signmax_mad_normalized":
        T = kwargs.get("signmax_mad_temperature", 0.5)
        return signmax_mad_normalized(U_flat, T=T, eps=kwargs.get("eps", 1e-8)).view(orig_shape)

    if mf == "align_weighted_mean":
        gamma = kwargs.get("awm_gamma", 2.0)
        clip_neg = kwargs.get("awm_clip_neg", True)
        return align_weighted_mean(U_flat, gamma=gamma, clip_neg=clip_neg, eps=kwargs.get("eps", 1e-8)).view(orig_shape)

    if mf == "coherence_penalized":
        lam = kwargs.get("cpw_lambda", 1e-3)
        return coherence_penalized_weights(U_flat, lam=lam).view(orig_shape)

    if mf == "orthogonal_deflation":
        weight_by_residual = kwargs.get("odm_weight_by_residual", True)
        return orthogonal_deflation_merge(U_flat, weight_by_residual=weight_by_residual, eps=kwargs.get("eps", 1e-8)).view(orig_shape)

# ============================================================================
# Helper Utilities
# ============================================================================

def layer_key(name: str) -> str:
    """
    Extract a layer key from a parameter name.
    
    For models with explicit layer indices (e.g., 'vision_model.encoder.layers.N'),
    includes the layer index to avoid grouping all layers together.
    
    Examples:
        'vision_model.encoder.layers.0.self_attn.k_proj.weight' -> 'vision_model.encoder.layers.0'
        'model.layer.0.attention.query.weight' -> 'model.layer.0'
        'embeddings.position_embedding.weight' -> 'embeddings.position_embedding'
    """
    parts = name.split(".")
    
    # Check if parts[2] == "layers" and parts[3] is a digit (common pattern)
    if len(parts) >= 4 and parts[2] == "layers" and parts[3].isdigit():
        return ".".join(parts[:4])  # Include layer index
    
    # Check if parts[1] == "layer" and parts[2] is a digit (alternative pattern)
    if len(parts) >= 3 and parts[1] == "layer" and parts[2].isdigit():
        return ".".join(parts[:3])  # Include layer index
    
    # Default fallback
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name

def compute_global_dim(state_dict: dict, keys: List[str]) -> int:
    total_dim = 0
    for k in keys:
        if k in state_dict:
            total_dim += state_dict[k].numel()
    return total_dim

def normalize_weights(weights: List[float] | None, num_tasks: int) -> List[float]:
    if weights is None:
        return [1.0 / num_tasks] * num_tasks
    if len(weights) != num_tasks:
        raise ValueError(f"Number of weights ({len(weights)}) != number of tasks ({num_tasks})")
    total = sum(weights) + EPS
    return [w / total for w in weights]