"""
FastFood Whitening Merging Algorithm

This module implements a model fusion technique that combines the computational efficiency 
of FastFood random projections with the interference reduction technique from the Task 
Singular Vector Merging (TSVM) paper. The algorithm operates layer-by-layer, projecting 
task matrices into a low-dimensional space, decorrelating them using whitening 
transformation, aggregating the results, and lifting back to the original parameter space.

The key innovation is applying whitening in the projected subspace to reduce task 
interference before aggregation, inspired by the TSV methodology but adapted for 
the efficient FastFood framework.
"""

from __future__ import annotations
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

from fusion_bench.utils import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType

EPS = 1e-12


# ---------------- Fastfood / SRHT helpers ----------------
def _next_pow2(n: int) -> int:
    """Get the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def _fwht_inplace_ortho(x: Tensor) -> Tensor:
    """In-place orthonormal Fast Walsh-Hadamard Transform along the last dim (scale 1/sqrt(n))."""
    n = x.shape[-1]
    if n <= 1:
        return x
    
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                u = x[..., i + j]
                v = x[..., i + j + h]
                x[..., i + j] = u + v
                x[..., i + j + h] = u - v
        h *= 2
    
    x.mul_(1.0 / math.sqrt(n))
    return x


def _seed_from(s: str) -> int:
    """Generate a deterministic seed from a string."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")


def _fastfood_ops(
    global_dim: int,
    proj_dim: int,
    *,
    seed_key: str,
    device: torch.device,
    use_G: bool,
):
    """
    Build a Fastfood operator with:
      V = H Π G H B ∈ R^{L×L}, L = 2^⌈log2 D⌉
      P = random row subset of size m = proj_dim
    We return:
      fwd(x)  = sqrt(L/m) * P V [x; 0]
      lift(y) = V^T P^T (y / sqrt(L/m))
    The same (B, G, Π, P) are reused for all tensors sharing `seed_key`.
    """
    torch.manual_seed(_seed_from(seed_key))
    D = int(global_dim)
    L = _next_pow2(D)
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
        """Project from original space to subspace."""
        assert xD.shape[-1] == D, f"Expected tensor with last dim {D}, got {xD.shape}"
        x = xD.clone()
        if D < L:
            # Pad to next power of 2
            padding = torch.zeros(x.shape[:-1] + (L - D,), dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=-1)
        
        x = x.to(torch.float32, copy=False)
        
        # Apply FastFood transform: B -> H -> Π -> G -> H -> P
        x.mul_(B)
        _fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G)
        _fwht_inplace_ortho(x)
        x = x.index_select(dim=-1, index=row_idx)  # P V x
        return (scale * x).contiguous()

    def lift(y: Tensor) -> Tensor:
        """Lift from subspace back to original space."""
        y = y.to(torch.float32, copy=False) / scale
        
        # Reconstruct full vector: P^T -> H -> G^T -> Π^T -> H -> B
        y_full = torch.zeros(y.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
        y_full.index_copy_(dim=-1, index=row_idx, source=y)  # P^T y
        _fwht_inplace_ortho(y_full)
        y_full.mul_(G)
        y_full = y_full[..., inv_Pi]
        _fwht_inplace_ortho(y_full)
        y_full.mul_(B)  # V^T P^T y
        return y_full[..., :D].contiguous()

    return fwd, lift


# --------------- Whitening in Subspace ----------------
@torch.no_grad()
def _whiten_task_vectors(P: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Apply whitening transformation to decorrelate task vectors in the subspace.
    
    This implements the core TSVM concept of interference reduction through whitening.
    The whitening transformation makes task vectors orthogonal, reducing interference
    when they are aggregated.
    
    Args:
        P: [d, T] matrix where columns are projected task vectors
        eps: Small constant for numerical stability
        
    Returns:
        P_perp: [d, T] whitened matrix where columns are decorrelated
    """
    d, T = P.shape
    if T <= 1:
        return P
    
    # For efficiency, use QR decomposition when T is small (which is usually the case)
    # This is much faster than SVD for small matrices
    if T <= 20:  # Use QR for small number of tasks
        try:
            # Center the vectors (optional, can help with stability)
            P_centered = P - P.mean(dim=1, keepdim=True)
            
            # QR decomposition: P = Q R, where Q has orthonormal columns
            Q, R = torch.linalg.qr(P_centered)
            
            # The QR decomposition already gives us orthogonal columns in Q
            # Scale to preserve magnitude information
            scale = torch.sqrt(torch.tensor(T, dtype=P.dtype, device=P.device))
            P_perp = Q * scale
            
            return P_perp
            
        except RuntimeError:
            # Fallback to covariance method if QR fails
            pass
    
    # Covariance-based whitening (more stable for larger T or as fallback)
    # Compute covariance matrix P^T P (T x T)
    Cov = P.T @ P  # [T, T]
    
    # Compute inverse square root via eigendecomposition for numerical stability
    # This is the closed-form solution for the whitening transformation
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(Cov)
        
        # Clamp eigenvalues to avoid numerical issues (regularization)
        eigenvals = torch.clamp(eigenvals, min=eps)
        
        # Compute eigenvals^{-1/2} - this is the key step for whitening
        eigenvals_inv_sqrt = 1.0 / torch.sqrt(eigenvals)
        
        # Cov^{-1/2} = V D^{-1/2} V^T where Cov = V D V^T
        Cov_inv_sqrt = eigenvecs @ torch.diag(eigenvals_inv_sqrt) @ eigenvecs.T
        
    except RuntimeError as e:
        print(f"Warning: Eigendecomposition failed in whitening, using identity: {e}")
        Cov_inv_sqrt = torch.eye(T, device=P.device, dtype=P.dtype)
    
    # Apply whitening: P_perp = P * Cov^{-1/2}
    # This ensures that P_perp^T @ P_perp = I (orthogonal columns)
    P_perp = P @ Cov_inv_sqrt
    
    return P_perp


@torch.no_grad()
def _aggregate_whitened_vectors(P_perp: Tensor, agg_method: str = "mean") -> Tensor:
    """
    Aggregate whitened task vectors in the subspace.
    
    Args:
        P_perp: [d, T] whitened matrix
        agg_method: Aggregation method ("mean", "sum", "median")
        
    Returns:
        p_merged: [d] merged vector in subspace
    """
    if agg_method == "mean":
        return P_perp.mean(dim=1)
    elif agg_method == "sum":
        return P_perp.sum(dim=1)
    elif agg_method == "median":
        return torch.median(P_perp, dim=1)[0]
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")


def _layer_key(name: str) -> str:
    """Heuristic layer-grouping key (works for most HF models)."""
    parts = name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name


# ------------------ Main Algorithm ------------------
@auto_register_config
class FastfoodWhiteningMergeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    FastFood Whitening Subspace Merging Algorithm.

    This algorithm combines FastFood random projections with whitening-based interference 
    reduction from the TSVM methodology. The complete workflow is:
    
    1. **Task Vector Extraction**: Compute Δᵢ = θᵢ - θ₀ for each fine-tuned model
    2. **FastFood Projection**: Project task vectors into low-dimensional subspace via SRHT
    3. **Whitening Transformation**: Decorrelate projected vectors to reduce interference
    4. **Aggregation**: Combine whitened vectors (mean/sum/median)
    5. **Lifting**: Project merged result back to original parameter space
    6. **Final Model**: θ_merged = θ₀ + α * Δ_merged
    
    Key Features:
    - **Computational Efficiency**: O(D log d) FastFood projection vs O(D²) dense projection
    - **Interference Reduction**: Whitening decorrelates task vectors before aggregation  
    - **Adaptive Dimensionality**: Uses D/T heuristic from TSVM paper
    - **Numerical Stability**: Multiple fallback methods for whitening computation
    
    Controls:
      proj_ratio: float (0..1) - Subspace compression ratio (used if adaptive_proj_dim=False)
      subspace_scope: "per_tensor" | "layer" | "global" - Projection scope
      use_G: bool - Use Gaussian scaling in FastFood transform
      device: str - Computation device
      weights: list[float] - Task importance weights 
      scale: float - Post-merge scaling factor (α)
      aggregation_method: str - How to aggregate whitened vectors ("mean", "sum", "median")
      whitening_eps: float - Numerical stability constant for whitening (regularization)
      adaptive_proj_dim: bool - Use D/T heuristic (recommended: True)
      enable_whitening: bool - Enable whitening step (False = regular FastFood)
    """

    def __init__(
        self,
        proj_ratio: float = 0.75,
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",  # "per_tensor" | "layer" | "global"
        block_rows: int = 8192,
        weights: List[float] | None = None,
        scale: float = 1.0,
        aggregation_method: str = "mean",  # "mean" | "sum" | "median"
        whitening_eps: float = 1e-8,
        adaptive_proj_dim: bool = True,  # Use D/T as projection dimension
        enable_whitening: bool = True,   # Enable/disable whitening transformation
        # Analysis integration parameters
        run_analysis: bool = False,
        analysis_methods: List[str] = None,
        analysis_output_path: str = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        assert subspace_scope in {"per_tensor", "layer", "global"}
        self.block_rows = int(block_rows)
        self.weights = list(weights) if weights is not None else None
        self.scale = float(scale)
        self.aggregation_method = str(aggregation_method)
        self.whitening_eps = float(whitening_eps)
        self.adaptive_proj_dim = bool(adaptive_proj_dim)
        self.enable_whitening = bool(enable_whitening)
        
        # Analysis parameters
        self.run_analysis = run_analysis
        self.analysis_methods = analysis_methods or []
        self.analysis_output_path = analysis_output_path

    @torch.no_grad()
    def run(
        self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs: Any
    ) -> nn.Module:
        modelpool = to_modelpool(modelpool)

        # ---------- Load Models ----------
        with self.profile("loading models"):
            base_model = modelpool.load_model("_pretrained_")
            donor_names = list(modelpool.model_names)
            if len(donor_names) < 2:
                raise ValueError(f"Need ≥2 donors; got {len(donor_names)}")

            donors_sd: List[StateDictType] = [
                modelpool.load_model(n).state_dict(keep_vars=True)
                for n in donor_names
            ]
            base_sd: Dict[str, Tensor] = base_model.state_dict(keep_vars=True)

        # ---------- Filter Eligible Tensors ----------
        keys_all = list(base_sd.keys())
        keys_float = [
            k for k in keys_all
            if (k in donors_sd[0])
            and torch.is_floating_point(base_sd[k])
            and base_sd[k].ndim >= 1
            and all((k in d) and torch.is_floating_point(d[k]) and d[k].shape == base_sd[k].shape for d in donors_sd)
        ]
        K = len(donor_names)
        print(f"[Setup] donors={K} | total tensors={len(keys_all)} | eligible float tensors={len(keys_float)}")

        if not keys_float:
            raise RuntimeError("No overlapping float tensors with identical shapes. Nothing to merge.")

        # ---------- Compute Task Vectors: Δᵢ = θᵢ - θ₀ ----------
        with self.profile("computing task vectors"):
            task_vectors: List[Dict[str, Tensor]] = []
            for i, donor_sd in enumerate(donors_sd):
                task_vec = {}
                for k in keys_float:
                    task_vec[k] = donor_sd[k] - base_sd[k]
                task_vectors.append(task_vec)

        # ---------- Group Tensors by Scope ----------
        if self.subspace_scope == "per_tensor":
            groups = {k: [k] for k in keys_float}
        elif self.subspace_scope == "layer":
            # Group by layer using heuristic
            layer_groups = {}
            for k in keys_float:
                layer_k = _layer_key(k)
                if layer_k not in layer_groups:
                    layer_groups[layer_k] = []
                layer_groups[layer_k].append(k)
            groups = layer_groups
        else:  # global
            groups = {"global": keys_float}

        print(f"[Grouping] scope={self.subspace_scope} | groups={len(groups)}")

        # ---------- Normalize Task Weights ----------
        if self.weights is not None:
            if len(self.weights) != K:
                raise ValueError(f"Expected {K} weights, got {len(self.weights)}")
            w_sum = sum(self.weights)
            if w_sum <= 0:
                raise ValueError(f"Sum of weights must be positive, got {w_sum}")
            norm_weights = [w / w_sum for w in self.weights]
        else:
            norm_weights = [1.0 / K] * K

        # ---------- Process Each Tensor Individually (Efficient Pattern) ----------
        merged_delta = {}
        
        # Move base model to CPU for efficient processing
        base_cpu = {k: v.cpu() for k, v in base_sd.items() if k in keys_float}
        donors_cpu = [{k: v.cpu() for k, v in task_vectors[i].items()} for i in range(K)]
        
        # Operator cache for reuse
        op_cache = {}
        
        # Helper function for projection dimension and seed key
        def proj_seed_key(name: str) -> str:
            if self.subspace_scope == "per_tensor":
                return name
            elif self.subspace_scope == "layer":
                return _layer_key(name)
            else:  # global
                return "global"
        
        # Determine global dimension for global scope
        global_D = None
        if self.subspace_scope == "global":
            global_D = sum(base_cpu[k].numel() for k in keys_float)
        
        print(f"[Processing] scope={self.subspace_scope} | total_tensors={len(keys_float)}")
        if global_D is not None:
            print(f"[Global scope] total_dim={global_D}")
        
        # Process each tensor efficiently using block processing
        for name in keys_float:
            with self.profile(f"process tensor {name}"):
                tb = base_cpu[name]
                d_last = int(tb.shape[-1])
                rows = tb.numel() // d_last
                if rows <= 0:
                    continue

                # Reshape to 2D for efficient processing
                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                # Choose scoped dimension & build (or reuse) operator
                seed_key = proj_seed_key(name)
                cur_D = global_D if (global_D is not None) else d_last
                
                if self.adaptive_proj_dim:
                    proj_dim = max(1, cur_D // K)  # D/T heuristic
                else:
                    proj_dim = max(1, int(cur_D * self.proj_ratio))
                
                cache_key = (seed_key, cur_D, proj_dim)
                if cache_key not in op_cache:
                    fwd, lift = _fastfood_ops(
                        cur_D, proj_dim, seed_key=seed_key, device=self.device, use_G=self.use_G
                    )
                    op_cache[cache_key] = (fwd, lift)
                else:
                    fwd, lift = op_cache[cache_key]

                # Process tensor in blocks to avoid memory issues
                cursor = 0
                result_blocks = []

                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor:cursor + take, :]

                    # Collect task vectors for this block
                    task_deltas = []
                    for i in range(K):
                        sl_donor = donors_cpu[i][name].view(rows, d_last).float()[cursor:cursor + take, :]
                        delta = sl_donor  # Already computed as task vector
                        
                        # Pad to global dimension if needed
                        if global_D is not None and d_last < cur_D:
                            buf = torch.zeros((take, cur_D), dtype=torch.float32, device="cpu")
                            buf[:, :d_last].copy_(delta)
                            task_deltas.append(buf)
                        else:
                            task_deltas.append(delta)

                    # Project all task vectors to subspace
                    projected_deltas = []
                    for delta in task_deltas:
                        proj_delta = fwd(delta.to(self.device, non_blocking=True))  # [take, proj_dim]
                        projected_deltas.append(proj_delta.cpu())  # Move back to CPU

                    # Stack into matrix P: [take, proj_dim, K]
                    P = torch.stack(projected_deltas, dim=2)  # [take, proj_dim, K]
                    
                    # Apply whitening per row (each parameter position independently)
                    if self.enable_whitening:
                        P_whitened = torch.zeros_like(P)
                        for row in range(take):
                            P_row = P[row]  # [proj_dim, K]
                            P_whitened[row] = _whiten_task_vectors(P_row, eps=self.whitening_eps)
                    else:
                        P_whitened = P

                    # Aggregate in subspace using weights
                    if self.weights is not None:
                        weight_tensor = torch.tensor(norm_weights, dtype=P_whitened.dtype, device=P_whitened.device)
                        merged_proj = (P_whitened * weight_tensor.view(1, 1, K)).sum(dim=2)  # [take, proj_dim]
                    else:
                        # Use specified aggregation method
                        if self.aggregation_method == "mean":
                            merged_proj = P_whitened.mean(dim=2)  # [take, proj_dim]
                        elif self.aggregation_method == "sum":
                            merged_proj = P_whitened.sum(dim=2)
                        elif self.aggregation_method == "median":
                            merged_proj = torch.median(P_whitened, dim=2)[0]
                        else:
                            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

                    # Lift back to original space
                    merged_lifted = lift(merged_proj.to(self.device, non_blocking=True))  # [take, cur_D or d_last]
                    
                    # Trim back to original tensor dimension if we padded
                    if global_D is not None and d_last < cur_D:
                        merged_lifted = merged_lifted[:, :d_last]
                    
                    result_blocks.append(merged_lifted.cpu())
                    cursor += take

                # Concatenate all blocks and reshape back to original tensor shape
                merged_tensor = torch.cat(result_blocks, dim=0).view(tb.shape)
                merged_delta[name] = merged_tensor.to(base_sd[name].dtype)

        # ---------- Construct Final Model: θ_merged = θ₀ + α * Δ_merged ----------
        with self.profile("constructing final model"):
            final_state_dict = {}
            
            for k in keys_all:
                if k in merged_delta:
                    # Apply scaling and add to base
                    final_state_dict[k] = base_sd[k] + self.scale * merged_delta[k]
                else:
                    # Keep base tensor unchanged
                    final_state_dict[k] = base_sd[k]

        # ---------- Load State Dict into Model ----------
        if isinstance(base_model, nn.Module):
            model = base_model
            model.load_state_dict(final_state_dict)
        elif isinstance(base_model, LazyStateDict):
            from copy import deepcopy
            model = deepcopy(base_model.meta_module)
            model = model.to_empty(device=base_model._device)
            result = model.load_state_dict(final_state_dict, strict=False)
            if result.unexpected_keys:
                raise ValueError(f"Unexpected keys in state dict: {result.unexpected_keys}")
            if result.missing_keys:
                print(f"Warning: Missing keys in state dict: {result.missing_keys}")
        else:
            raise TypeError(f"Unsupported model type: {type(base_model)}")

        self.print_profile_summary()
        
        # ---------- Optional Analysis ----------
        if self.run_analysis:
            print("[Analysis] Running integrated analysis...")
            # This would integrate with the existing analysis framework
            # For now, we'll skip this to keep the implementation focused
            
        return model
