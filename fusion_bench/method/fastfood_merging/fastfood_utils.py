"""
Utility functions for FWHT/SRHT/DCT/DHT/Fastfood-based subspace projections
and model-merge helpers (TIES/EMA/etc.).

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
from typing import List, Tuple, Callable, Literal

import numpy as np
from scipy.fft import dct as sp_dct, idct as sp_idct

import torch
from torch import Tensor

# Public exports
__all__ = [
    "EPS",
    "create_projection_ops",
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
    "ema_adaptive_beta",
    "ema_merge_subspace",
    "energy_equalize",
    "variance_aware_weights",
    "subspace_whiten",
    "conflict_gating",
    "elect_then_avg",
    "soft_signmax",
    "align_weighted_mean",
    "coherence_penalized_weights",
    "orthogonal_deflation_merge",
    "odm_ema_tuning_free",
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
        a = y[..., 0, :].clone()
        b = y[..., 1, :]
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        y = y.view(-1, n)
        h *= 2
    y.mul_(1.0 / math.sqrt(n))
    return y.view(*lead, n)

@torch.no_grad()
def dct_ortho(x: Tensor) -> Tensor:
    """Orthonormal DCT-II transform (SciPy backend, CPU fallback)."""
    x_np = x.detach().cpu().numpy()
    X = sp_dct(x_np, type=2, norm="ortho", axis=-1)
    return torch.from_numpy(X).to(x.device, dtype=x.dtype)

@torch.no_grad()
def idct_ortho(x: Tensor) -> Tensor:
    # Inverse of DCT-II (ortho) is DCT-III (ortho)
    x_np = x.detach().cpu().numpy()
    X = sp_idct(x_np, type=3, norm="ortho", axis=-1)
    return torch.from_numpy(X).to(x.device, dtype=x.dtype)

@torch.no_grad()
def dht_ortho(x: Tensor) -> Tensor:
    """
    Real-valued Discrete Hartley Transform (self-inverse).
    Implemented via FFT: H(x) = Re(FFT(x)) − Im(FFT(x)).
    """
    x_np = x.detach().cpu().numpy()
    X_fft = np.fft.fft(x_np, axis=-1)
    X = np.real(X_fft) - np.imag(X_fft)
    X /= np.sqrt(x_np.shape[-1])  # orthonormal scaling
    return torch.from_numpy(X).to(x.device, dtype=x.dtype)

@torch.no_grad()
def idht_ortho(x: Tensor) -> Tensor:
    """Inverse DHT (identical to forward)."""
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
# Zero-Aware Aggregation Functions
# ============================================================================

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
        "sum", "mean", "max", "signmax", "ema",
        "ties_sum", "ties_mean", "ties_max",
        "energy_equalize", "variance_aware", "subspace_whiten",
        "conflict_gating", "elect_then_avg", "soft_signmax",
        "align_weighted_mean", "coherence_penalized", "orthogonal_deflation",
        "odm_ema_tuning_free"
    }
    if mf not in valid_funcs:
        raise ValueError(f"merge_func={merge_func} not in {valid_funcs}")

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
        if weights is None:
            return U.sum(dim=0)
        w = torch.tensor(weights, dtype=U.dtype, device=U.device).view(
            K, *([1] * (U.ndim - 1))
        )
        return (w * U).sum(dim=0)

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
        return out

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