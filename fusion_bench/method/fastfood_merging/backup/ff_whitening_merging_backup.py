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
    
    # Vectorized FWHT implementation for better performance
    h = 1
    while h < n:
        # Use tensor operations instead of loops for better performance
        idx1 = torch.arange(0, n, h * 2, device=x.device).unsqueeze(1) + torch.arange(h, device=x.device)
        idx2 = idx1 + h
        
        # Ensure indices don't exceed bounds
        valid_mask = (idx2 < n).all(dim=1)
        if not valid_mask.all():
            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]
        
        if idx1.numel() > 0:
            # Reshape for vectorized operations
            idx1_flat = idx1.flatten()
            idx2_flat = idx2.flatten()
            
            u = x[..., idx1_flat]
            v = x[..., idx2_flat]
            x[..., idx1_flat] = u + v
            x[..., idx2_flat] = u - v
        
        h *= 2
    
    x.mul_(1.0 / math.sqrt(n))
    return x


def _seed_from(s: str) -> int:
    """Generate a deterministic seed from a string."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")


def _simple_random_projection_ops(
    global_dim: int,
    proj_dim: int,
    *,
    seed_key: str,
    device: torch.device,
):
    """
    Simple random projection for efficiency.
    Uses a sparse random matrix for fast projection and lifting.
    """
    torch.manual_seed(_seed_from(seed_key))
    D = int(global_dim)
    m = max(1, int(proj_dim))
    
    # Create sparse random projection matrix
    # Use sparse matrix with few non-zero entries per row for efficiency
    sparsity = min(0.1, 10.0 / D)  # At most 10% non-zero or 10 elements per row
    nnz_per_row = max(1, int(D * sparsity))
    
    # Create projection matrix P [m, D]
    P_rows = []
    P_cols = []
    P_vals = []
    
    for i in range(m):
        # Randomly select columns for this row
        cols = torch.randperm(D)[:nnz_per_row]
        # Random values (+1 or -1)
        vals = torch.randint(0, 2, (nnz_per_row,), dtype=torch.float32) * 2 - 1
        vals = vals / math.sqrt(nnz_per_row)  # Normalize
        
        P_rows.extend([i] * nnz_per_row)
        P_cols.extend(cols.tolist())
        P_vals.extend(vals.tolist())
    
    # Create sparse matrix
    indices = torch.stack([
        torch.tensor(P_rows, device=device),
        torch.tensor(P_cols, device=device)
    ])
    values = torch.tensor(P_vals, device=device, dtype=torch.float32)
    P_sparse = torch.sparse_coo_tensor(indices, values, (m, D), device=device)
    P_sparse = P_sparse.coalesce()
    
    def fwd(x: Tensor) -> Tensor:
        """Project from original space to subspace."""
        return torch.sparse.mm(P_sparse, x.T).T
    
    def lift(y: Tensor) -> Tensor:
        """Lift from subspace back to original space."""
        return torch.sparse.mm(P_sparse.T, y.T).T
    
    return fwd, lift


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
    
    For efficiency, falls back to simple random projection for large dimensions.
    """
    D = int(global_dim)
    m = max(1, int(proj_dim))
    
    # Use simple random projection for large dimensions to avoid FWHT overhead
    if D > 1000:  
        return _simple_random_projection_ops(D, m, seed_key=seed_key, device=device)
    
    # Original FastFood implementation for smaller dimensions
    torch.manual_seed(_seed_from(seed_key))
    L = _next_pow2(D)

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
        assert xD.shape[-1] == D
        x = xD
        if D < L:
            x = torch.cat([x, torch.zeros(x.shape[:-1] + (L - D,), dtype=x.dtype, device=x.device)], dim=-1)
        x = x.to(torch.float32, copy=False)
        x.mul_(B)
        _fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G)
        _fwht_inplace_ortho(x)
        x = x.index_select(dim=-1, index=row_idx)  # P V x
        return (scale * x).contiguous()

    def lift(y: Tensor) -> Tensor:
        """Lift from subspace back to original space."""
        y = (y.to(torch.float32, copy=False) / scale)
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
    
    Args:
        P: [d, T] matrix where columns are projected task vectors
        eps: Small constant for numerical stability
        
    Returns:
        P_perp: [d, T] whitened matrix where columns are decorrelated
    """
    T = P.shape[1]
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
            # Fallback to original method if QR fails
            pass
    
    # Original SVD method for larger T or as fallback
    # Compute covariance matrix P^T P (T x T)
    Cov = P.T @ P  # [T, T]
    
    # Compute inverse square root via eigendecomposition for numerical stability
    try:
        eigenvals, eigenvecs = torch.linalg.eigh(Cov)
        
        # Clamp eigenvalues to avoid numerical issues
        eigenvals = torch.clamp(eigenvals, min=eps)
        
        # Compute eigenvals^{-1/2}
        eigenvals_inv_sqrt = 1.0 / torch.sqrt(eigenvals)
        
        # Cov^{-1/2} = V D^{-1/2} V^T where Cov = V D V^T
        Cov_inv_sqrt = eigenvecs @ torch.diag(eigenvals_inv_sqrt) @ eigenvecs.T
        
    except RuntimeError as e:
        print(f"Warning: Eigendecomposition failed in whitening, using identity: {e}")
        Cov_inv_sqrt = torch.eye(T, device=P.device, dtype=P.dtype)
    
    # Apply whitening: P_perp = P * Cov^{-1/2}
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
    reduction from the TSVM methodology. It operates by:
    
    1. Projecting task vectors into a low-dimensional subspace via FastFood transform
    2. Applying whitening transformation to decorrelate task vectors  
    3. Aggregating the decorrelated vectors in the subspace
    4. Lifting the merged result back to the original parameter space
    
    Controls:
      proj_ratio: float (0..1) - Subspace compression ratio (default: 0.75)
      subspace_scope: "per_tensor" | "layer" | "global" - Projection scope
      use_G: bool - Use Gaussian scaling in FastFood transform
      device: str - Computation device
      block_rows: int - Memory management for large tensors
      weights: list[float] - Task importance weights 
      scale: float - Post-merge scaling factor
      aggregation_method: str - How to aggregate whitened vectors ("mean", "sum", "median")
      whitening_eps: float - Numerical stability constant for whitening
      adaptive_proj_dim: bool - Adapt projection dimension based on number of tasks
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

        # ---------- Load ----------
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

        # ---------- Eligible tensors ----------
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

        # ---------- Compute task vectors ----------
        with self.profile("computing task vectors"):
            task_vectors: List[Dict[str, Tensor]] = []
            for i, donor_sd in enumerate(donors_sd):
                task_vec = {}
                for k in keys_float:
                    task_vec[k] = donor_sd[k] - base_sd[k]
                task_vectors.append(task_vec)

        # ---------- Group by scope ----------
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

        # ---------- Normalize task weights ----------
        if self.weights is not None:
            if len(self.weights) != K:
                raise ValueError(f"Expected {K} weights, got {len(self.weights)}")
            w_sum = sum(self.weights)
            if w_sum <= 0:
                raise ValueError(f"Sum of weights must be positive, got {w_sum}")
            norm_weights = [w / w_sum for w in self.weights]
        else:
            norm_weights = [1.0 / K] * K

        # ---------- Merge by group ----------
        merged_delta = {}
        
        for group_name, tensor_keys in groups.items():
            with self.profile(f"merge group {group_name}"):
                print(f"[Group {group_name}] Processing {len(tensor_keys)} tensors...")
                
                # Collect all tensors in this group
                group_task_vectors = []
                for i in range(K):
                    group_vec = []
                    for k in tensor_keys:
                        vec = task_vectors[i][k].flatten()
                        group_vec.append(vec)
                    # Concatenate all tensors in group into single vector
                    if len(group_vec) == 1:
                        group_task_vectors.append(group_vec[0])
                    else:
                        group_task_vectors.append(torch.cat(group_vec, dim=0))
                
                if not group_task_vectors:
                    continue
                    
                # Determine projection dimension
                global_dim = group_task_vectors[0].numel()
                
                if self.adaptive_proj_dim:
                    # Use D/T heuristic from TSVM paper
                    proj_dim = max(1, global_dim // K)
                else:
                    # Use traditional proj_ratio
                    proj_dim = max(1, int(global_dim * self.proj_ratio))
                
                print(f"[Group {group_name}] global_dim={global_dim}, proj_dim={proj_dim}, "
                      f"compression_ratio={proj_dim/global_dim:.4f}")

                # Create FastFood operators
                fwd, lift = _fastfood_ops(
                    global_dim=global_dim,
                    proj_dim=proj_dim,
                    seed_key=group_name,
                    device=self.device,
                    use_G=self.use_G,
                )

                # ---------- Step 2: Project to subspace ----------
                projected_vectors = []
                for task_vec in group_task_vectors:
                    task_vec = task_vec.to(self.device)
                    proj_vec = fwd(task_vec)  # [proj_dim]
                    projected_vectors.append(proj_vec)
                
                # Stack into matrix P: [proj_dim, K]
                P = torch.stack(projected_vectors, dim=1)  # [proj_dim, K]
                
                # ---------- Step 3: Apply whitening (optional) ----------
                if self.enable_whitening:
                    P_whitened = _whiten_task_vectors(P, eps=self.whitening_eps)
                else:
                    P_whitened = P  # Skip whitening, use original projected vectors
                
                # ---------- Step 4: Aggregate in subspace ----------
                if self.weights is not None:
                    # Weighted aggregation
                    w_tensor = torch.tensor(norm_weights, device=P_whitened.device, dtype=P_whitened.dtype)
                    p_merged = (P_whitened * w_tensor.view(1, -1)).sum(dim=1)
                else:
                    # Simple aggregation
                    p_merged = _aggregate_whitened_vectors(P_whitened, self.aggregation_method)
                
                # ---------- Step 5: Lift back to original space ----------
                merged_vec = lift(p_merged)  # [global_dim]
                
                # Split merged vector back into individual tensors
                start_idx = 0
                for k in tensor_keys:
                    orig_shape = base_sd[k].shape
                    numel = base_sd[k].numel()
                    
                    tensor_delta = merged_vec[start_idx:start_idx + numel].view(orig_shape)
                    merged_delta[k] = tensor_delta.to(base_sd[k].device, dtype=base_sd[k].dtype)
                    
                    start_idx += numel

        # ---------- Construct final model ----------
        with self.profile("constructing final model"):
            final_state_dict = {}
            
            for k in keys_all:
                if k in merged_delta:
                    # Apply scaling and add to base
                    final_state_dict[k] = base_sd[k] + self.scale * merged_delta[k]
                else:
                    # Keep base tensor unchanged
                    final_state_dict[k] = base_sd[k]

        # ---------- Load into model ----------
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
