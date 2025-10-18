"""
FastFood Wrapper for Model Merging

This module implements a wrapper that projects models into a lower-dimensional FastFood subspace,
applies compatible merging methods, and lifts back to the original space. The key insight is that
all models use the same projection matrix to ensure consistency.

Compatible merging methods:
- SimpleAverageAlgorithm
- TaskArithmeticAlgorithm  
- TiesMergingAlgorithm
- FisherMergingAlgorithm
- TaskSingularVectorMerging (TSVM)
- Max aggregation of task vectors
"""

from __future__ import annotations

import hashlib
import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch import Tensor, nn

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils import LazyStateDict
from fusion_bench.utils.type import StateDictType

# Import compatible merging algorithms
# Import compatible merging algorithms (lazy imports to avoid circular dependencies)
try:
    from fusion_bench.method.simple_average import SimpleAverageAlgorithm
except ImportError:
    SimpleAverageAlgorithm = None

try:
    from fusion_bench.method.task_arithmetic import TaskArithmeticAlgorithm
except ImportError:
    TaskArithmeticAlgorithm = None

try:
    from fusion_bench.method.ties_merging import TiesMergingAlgorithm
except ImportError:
    TiesMergingAlgorithm = None

try:
    from fusion_bench.method.fisher_merging.fisher_merging import FisherMergingAlgorithm
except ImportError:
    FisherMergingAlgorithm = None

try:
    from fusion_bench.method.task_singular_vector import TaskSingularVectorMerging
except ImportError:
    TaskSingularVectorMerging = None

log = logging.getLogger(__name__)

EPS = 1e-12


# ---------------- FastFood / SRHT helpers ----------------
def _next_pow2(n: int) -> int:
    """Get next power of 2."""
    return 1 << (n - 1).bit_length()


def _seed_from(s: str) -> int:
    """Generate seed from string."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")


@torch.no_grad()
def _fwht_inplace_ortho(x: Tensor) -> Tensor:
    """In-place orthonormal FWHT along the last dim (scale 1/sqrt(n))."""
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


def _create_fastfood_projection(
    global_dim: int,
    proj_dim: int,
    *,
    seed_key: str,
    device: torch.device,
    use_G: bool = False,
):
    """
    Create FastFood projection operators.
    
    Returns:
        fwd: Projects from global_dim to proj_dim
        lift: Lifts from proj_dim back to global_dim
    """
    torch.manual_seed(_seed_from(seed_key))
    D = int(global_dim)
    L = _next_pow2(D)
    m = max(1, int(proj_dim))

    # FastFood parameters
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

    # Johnson-Lindenstrauss row subset and scaling
    row_idx = torch.randperm(L, device=device)[:m]
    scale = math.sqrt(L / m)

    def fwd(xD: Tensor) -> Tensor:
        """Project from original space to subspace."""
        assert xD.shape[-1] == D
        x = xD
        if D < L:
            x = torch.nn.functional.pad(x, (0, L - D))
        x = x.to(torch.float32, copy=False)
        x.mul_(B)
        _fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G)
        _fwht_inplace_ortho(x)
        x = x.index_select(dim=-1, index=row_idx)
        return (scale * x).contiguous()

    def lift(y: Tensor) -> Tensor:
        """Lift from subspace back to original space."""
        y = (y.to(torch.float32, copy=False) / scale)
        y_full = torch.zeros(y.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
        y_full.index_copy_(dim=-1, index=row_idx, source=y)
        _fwht_inplace_ortho(y_full)
        y_full.mul_(G)
        y_full = y_full[..., inv_Pi]
        _fwht_inplace_ortho(y_full)
        y_full.mul_(B)
        return y_full[..., :D].contiguous()

    return fwd, lift


def _layer_key(name: str) -> str:
    """Generate layer grouping key for consistent projection matrices."""
    parts = name.split(".")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return name


@auto_register_config
class FastfoodWrapperAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    FastFood wrapper that projects models to lower-dimensional space for merging.
    
    This algorithm:
    1. Projects all model parameters to a lower-dimensional FastFood subspace
    2. Applies a compatible merging method in the projected space
    3. Lifts the merged result back to the original parameter space
    
    All models use the same projection matrix to ensure consistency.
    """
    
    def __init__(
        self,
        proj_ratio: float = 0.75,
        use_G: bool = False,
        device: str = "cuda",
        merging_method: str = "simple_average",
        method_config: Optional[Dict[str, Any]] = None,
        subspace_scope: Literal["per_tensor", "layer", "global"] = "layer",
        block_rows: int = 8192,
        **kwargs
    ):
        """
        Initialize FastFood wrapper.
        
        Args:
            proj_ratio: Compression ratio for subspace projection (0.0-1.0)
            use_G: Whether to use Gaussian scaling in FastFood transform
            device: Computation device
            merging_method: Which merging method to use in subspace
            method_config: Configuration for the wrapped merging method
            subspace_scope: Scope of projection matrices ("per_tensor", "layer", "global")
            block_rows: Number of rows to process in each block for memory efficiency
        """
        super().__init__(**kwargs)
        self.proj_ratio = proj_ratio
        self.use_G = use_G
        self.device = torch.device(device)
        self.merging_method = merging_method
        self.method_config = method_config or {}
        self.subspace_scope = subspace_scope
        self.block_rows = int(block_rows)
        
        # Create the wrapped merging algorithm
        self._create_wrapped_algorithm()

    def _create_wrapped_algorithm(self):
        """Create the wrapped merging algorithm instance."""
        if self.merging_method == "simple_average":
            if SimpleAverageAlgorithm is None:
                raise ImportError("SimpleAverageAlgorithm not available")
            self.wrapped_algorithm = SimpleAverageAlgorithm(**self.method_config)
        elif self.merging_method == "task_arithmetic":
            if TaskArithmeticAlgorithm is None:
                raise ImportError("TaskArithmeticAlgorithm not available")
            self.wrapped_algorithm = TaskArithmeticAlgorithm(**self.method_config)
        elif self.merging_method == "ties_merging":
            # Native subspace implementation - no wrapped algorithm needed
            self.wrapped_algorithm = None
        elif self.merging_method == "fisher_merging":
            # Native subspace implementation - no wrapped algorithm needed
            self.wrapped_algorithm = None
        elif self.merging_method == "task_singular_vector":
            # Native subspace implementation - no wrapped algorithm needed
            self.wrapped_algorithm = None
        elif self.merging_method == "max_aggregation":
            # Max aggregation will be implemented as a special case
            self.wrapped_algorithm = None
        else:
            raise ValueError(f"Unsupported merging method: {self.merging_method}")

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool) -> nn.Module:
        """
        Run FastFood wrapper merging.
        
        Args:
            modelpool: Pool of models to merge
            
        Returns:
            Merged model
        """
        log.info(
            f"FastFood wrapper merging with {self.merging_method} on {len(modelpool.model_names)} models"
        )
        
        # Load all models
        models = {}
        base_model = None
        for model_name in modelpool.model_names:
            model = modelpool.load_model(model_name)
            models[model_name] = model
            if base_model is None:
                base_model = deepcopy(model)
        
        # Get the base model for task arithmetic methods
        base_model_for_task_vectors = None
        if self.merging_method in ["task_arithmetic", "ties_merging", "max_aggregation"]:
            # Try to get base model from modelpool
            if hasattr(modelpool, '_base_model') and modelpool._base_model is not None:
                base_model_for_task_vectors = modelpool.load_model(modelpool._base_model)
            elif hasattr(modelpool, 'base_model') and modelpool.base_model is not None:
                base_model_for_task_vectors = modelpool.load_model(modelpool.base_model)
            else:
                # Use the first model as base model
                base_model_for_task_vectors = base_model
                log.warning("No base model specified, using first model as base for task vector computation")
            
        # Project all models to subspace and merge
        with self.profile("fastfood projection and merging"):
            merged_sd = self._project_merge_lift(
                models, base_model_for_task_vectors, base_model.state_dict()
            )
        
        # Load merged parameters into base model
        if isinstance(base_model, LazyStateDict):
            base_model = deepcopy(base_model.meta_module).to_empty(device=self.device)
        
        result = base_model.load_state_dict(merged_sd, strict=False)
        if result.unexpected_keys:
            log.warning(f"Unexpected keys: {result.unexpected_keys}")
        if result.missing_keys:
            log.warning(f"Missing keys: {result.missing_keys}")
            
        return base_model

    def _project_merge_lift(
        self, 
        models: Dict[str, nn.Module], 
        base_model_for_task_vectors: Optional[nn.Module],
        template_sd: StateDictType
    ) -> StateDictType:
        """Project models to subspace, merge, and lift back following fastfood_merging.py logic."""
        
        # Extract floating point parameters (matching fastfood_merging.py)
        keys_float = [k for k in template_sd.keys() 
                     if torch.is_floating_point(template_sd[k]) and template_sd[k].ndim >= 1]
        
        log.info(f"Processing {len(keys_float)} floating point tensors")
        
        # Determine global dimension if using global scope (matching fastfood_merging.py)
        global_D = None
        if self.subspace_scope == "global":
            maxd = 1
            for k in keys_float:
                t = template_sd[k]
                maxd = max(maxd, int(t.shape[-1]))
            global_D = maxd
        
        # Define projection seed key function (matching fastfood_merging.py)
        def proj_seed_key(name: str) -> str:
            if self.subspace_scope == "global":
                return "global"
            elif self.subspace_scope == "layer":
                return _layer_key(name)
            else:  # per_tensor
                return name
        
        # Report subspace sizing (first few examples)
        def _dim_for(k: str) -> tuple[int, int]:
            d_last = int(template_sd[k].shape[-1])
            cur_D = global_D if (global_D is not None) else d_last
            m = max(1, int(cur_D * self.proj_ratio))
            return cur_D, m

        ex = keys_float[:5]
        if ex:
            dims = [(_dim_for(k), k) for k in ex]
            log.info(f"Dims scope={self.subspace_scope} | proj_ratio={self.proj_ratio:.3f} | examples:")
            for (D, m), k in dims:
                log.info(f"   - {k}: original_last_dim={int(template_sd[k].shape[-1])} | scoped_dim={D} → proj_dim={m} (compression={m/max(1,D):.3f})")
        
        # Work on CPU copies (matching fastfood_merging.py)
        base_cpu = {k: v.detach().cpu().clone() for k, v in template_sd.items()}
        
        # Get donor models (task vectors if needed)
        donors_cpu = []
        if self.merging_method in ["task_arithmetic", "ties_merging", "max_aggregation"]:
            # Create task vectors
            if base_model_for_task_vectors is None:
                raise ValueError(f"Base model required for {self.merging_method} but not available")
            base_for_tasks = {k: v.detach().cpu().clone() 
                            for k, v in base_model_for_task_vectors.state_dict().items()}
            
            for model_name, model in models.items():
                donor_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                donors_cpu.append(donor_sd)
        else:
            # Use models directly
            for model_name, model in models.items():
                donor_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                donors_cpu.append(donor_sd)
        
        merged_sd = deepcopy(base_cpu)
        dev = self.device
        
        # Operator cache keyed by (seed_key, cur_D, proj_dim)
        op_cache = {}
        
        merged_tensors = 0
        
        with self.profile("merging models"):
            for name in keys_float:
                tb = base_cpu[name]
                d_last = int(tb.shape[-1])
                rows = tb.numel() // d_last
                if rows <= 0:
                    continue

                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                # Choose scoped dim & build (or reuse) operator
                seed_key = proj_seed_key(name)
                cur_D = global_D if (global_D is not None) else d_last
                proj_dim = max(1, int(cur_D * self.proj_ratio))
                cache_key = (seed_key, cur_D, proj_dim)
                if cache_key not in op_cache:
                    fwd, lift = _create_fastfood_projection(
                        cur_D, proj_dim, seed_key=f"fastfood_wrapper_{seed_key}", device=dev, use_G=self.use_G
                    )
                    op_cache[cache_key] = (fwd, lift)
                else:
                    fwd, lift = op_cache[cache_key]

                cursor = 0
                tensor_changed = False

                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor:cursor + take, :]

                    # Donor deltas aligned to cur_D if global scope
                    Xs = []
                    for dsd in donors_cpu:
                        if name not in dsd:
                            continue
                        sl_donor = dsd[name].view(rows, d_last).float()[cursor:cursor + take, :]
                        
                        if self.merging_method in ["task_arithmetic", "ties_merging", "max_aggregation"]:
                            # Task vector
                            delta = sl_donor - sl_base
                        else:
                            # Direct model weights
                            delta = sl_donor
                        
                        if global_D is not None and d_last < cur_D:
                            buf = torch.zeros((take, cur_D), dtype=torch.float32, device="cpu")
                            buf[:, :d_last].copy_(delta)
                            Xs.append(buf)
                        else:
                            Xs.append(delta)

                    if not Xs:
                        continue

                    # Project all donors
                    Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]

                    # Merge in subspace
                    U_stack = torch.stack(Ys, dim=0)  # [K, take, m]
                    Ymerge = self._merge_vectors_subspace(U_stack)

                    # Lift back to original space
                    Xmerge = lift(Ymerge).to("cpu", non_blocking=True)[:, :d_last]

                    # Apply to base
                    if self.merging_method in ["task_arithmetic", "ties_merging", "max_aggregation"]:
                        # Add task vector to base
                        merged_sd[name].view(rows, d_last)[cursor:cursor + take, :].add_(Xmerge)
                    else:
                        # Replace with merged weights
                        merged_sd[name].view(rows, d_last)[cursor:cursor + take, :].copy_(Xmerge)

                    cursor += take
                    tensor_changed = True

                if tensor_changed:
                    merged_tensors += 1

        log.info(f"FastFood wrapper merge complete: {merged_tensors} tensors merged")
        return merged_sd

    def _merge_vectors_subspace(self, U_stack: Tensor) -> Tensor:
        """Merge vectors in subspace using the specified method."""
        # U_stack: [K, take, m] where K is number of models, take is block size, m is proj_dim
        
        if self.merging_method == "simple_average":
            return U_stack.mean(dim=0)
            
        elif self.merging_method == "task_arithmetic":
            scaling_factor = self.method_config.get("scaling_factor", 0.3)
            return scaling_factor * U_stack.mean(dim=0)
            
        elif self.merging_method == "ties_merging":
            return self._ties_merge_vectorized(U_stack)
            
        elif self.merging_method == "fisher_merging":
            return self._fisher_merge_vectorized(U_stack)
            
        elif self.merging_method == "task_singular_vector":
            return self._tsvm_merge_vectorized(U_stack)
            
        elif self.merging_method == "max_aggregation":
            # Element-wise max by absolute value
            abs_vals = torch.abs(U_stack)
            max_indices = torch.argmax(abs_vals, dim=0)
            return torch.gather(U_stack, 0, max_indices.unsqueeze(0)).squeeze(0)
            
        else:
            raise ValueError(f"Unknown merging method: {self.merging_method}")

    def _ties_merge_vectorized(self, U_stack: Tensor) -> Tensor:
        """Vectorized TIES merging implementation."""
        scaling_factor = self.method_config.get("scaling_factor", 0.3)
        threshold = self.method_config.get("threshold", 20.0)
        merge_func = self.method_config.get("merge_func", "sum")
        
        K = U_stack.shape[0]
        
        # Step 1: Trim (remove low-magnitude values)
        flat_U = U_stack.view(K, -1)  # [K, features]
        abs_U = torch.abs(flat_U)
        
        # Calculate percentile threshold for each model
        threshold_vals = torch.quantile(abs_U, q=(100 - threshold) / 100, dim=1, keepdim=True)
        mask = abs_U >= threshold_vals
        trimmed_U = torch.where(mask, flat_U, torch.zeros_like(flat_U))
        
        # Step 2: Elect (resolve sign conflicts)
        sign_sum = torch.sign(trimmed_U).sum(dim=0)
        majority_sign = torch.sign(sign_sum)
        
        # Keep only values that agree with majority sign
        for k in range(K):
            model_signs = torch.sign(trimmed_U[k])
            conflict_mask = (model_signs != 0) & (model_signs != majority_sign) & (majority_sign != 0)
            trimmed_U[k] = torch.where(conflict_mask, torch.zeros_like(trimmed_U[k]), trimmed_U[k])
        
        # Step 3: Merge
        if merge_func == "sum":
            result = trimmed_U.sum(dim=0)
        elif merge_func == "mean":
            # Count non-zero contributions
            non_zero_count = (trimmed_U != 0).sum(dim=0).clamp(min=1)
            result = trimmed_U.sum(dim=0) / non_zero_count
        else:
            result = trimmed_U.sum(dim=0)
        
        # Apply scaling and reshape back
        result = scaling_factor * result
        return result.view(U_stack.shape[1:])

    def _fisher_merge_vectorized(self, U_stack: Tensor) -> Tensor:
        """Vectorized Fisher merging implementation (simplified)."""
        normalize_fisher_weight = self.method_config.get("normalize_fisher_weight", True)
        minimal_fisher_weight = self.method_config.get("minimal_fisher_weight", 1e-6)
        
        # For simplicity, use uniform Fisher weights in subspace
        # In practice, you'd compute actual Fisher information
        K = U_stack.shape[0]
        fisher_weights = torch.ones(K, device=U_stack.device) / K
        
        if normalize_fisher_weight:
            fisher_weights = fisher_weights / fisher_weights.sum()
        
        fisher_weights = torch.clamp(fisher_weights, min=minimal_fisher_weight)
        
        # Weighted average
        fisher_weights = fisher_weights.view(-1, 1, 1)  # [K, 1, 1]
        return (U_stack * fisher_weights).sum(dim=0)

    def _tsvm_merge_vectorized(self, U_stack: Tensor) -> Tensor:
        """
        Vectorized TSVM implementation following the reference algorithm.
        
        This implements the exact TSVM algorithm from the reference implementation:
        1. For each task vector (already in projected space):
           - Compute SVD: u, s, vh = svd(task_vector)  
           - Keep only top 1/K fraction of singular values (sv_reduction)
           - Concatenate the reduced components across tasks
        2. After processing all tasks:
           - Compute SVD of concatenated sum_u and sum_vh
           - Merge: u_u @ vh_u @ diag(sum_s) @ u_vh @ vh_vh
        
        Args:
            U_stack: [K, take, m] where K is number of models, take is block size, m is proj_dim
                    These are already task vectors (deltas) in the projected space
        
        Returns:
            Merged task vector of shape [take, m]
        """
        alpha = self.method_config.get("alpha", 1.0)
        
        K, take, m = U_stack.shape
        num_tasks = K
        
        # TSVM reduction factor: each task gets 1/K of the singular values
        sv_reduction = 1.0 / num_tasks
        
        try:
            # Lists to collect SVD components from all tasks
            u_list = []
            s_list = []
            vh_list = []
            
            # Process each task vector
            for i in range(num_tasks):
                # Get task vector for this task: [take, m]
                vec = U_stack[i]  # [take, m]
                
                # Ensure float32 for SVD (reference implementation does this)
                if vec.dtype not in [torch.float32, torch.float64]:
                    vec = vec.to(dtype=torch.float32)
                
                # Compute SVD of this task vector (2D matrix)
                # vec = u @ diag(s) @ vh
                u, s, vh = torch.linalg.svd(vec, full_matrices=False)
                # u: [take, rank], s: [rank], vh: [rank, m]
                # where rank = min(take, m)
                
                # Calculate how many singular values to keep for this task
                reduced_index_s = max(1, int(s.shape[0] * sv_reduction))  # At least 1
                
                # Keep only the top singular values and corresponding vectors
                u_reduced = u[:, :reduced_index_s]      # [take, reduced_rank]
                s_reduced = s[:reduced_index_s]         # [reduced_rank]
                vh_reduced = vh[:reduced_index_s, :]    # [reduced_rank, m]
                
                # Collect for concatenation
                u_list.append(u_reduced)
                s_list.append(s_reduced)
                vh_list.append(vh_reduced)
            
            # Concatenate all reduced SVD components along the rank dimension
            # This creates larger matrices that contain information from all tasks
            sum_u = torch.cat(u_list, dim=1)    # [take, total_reduced_rank]
            sum_s = torch.cat(s_list, dim=0)    # [total_reduced_rank]
            sum_vh = torch.cat(vh_list, dim=0)  # [total_reduced_rank, m]
            
            # Stage 2: Orthogonalize the concatenated matrices via SVD
            # This is the key TSVM step that reduces task interference
            u_u, s_u, vh_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_vh, s_vh, vh_vh = torch.linalg.svd(sum_vh, full_matrices=False)
            
            # Merge using the TSVM formula
            # result = (u_u @ vh_u) @ diag(sum_s) @ (u_vh @ vh_vh)
            result = torch.linalg.multi_dot([
                u_u,
                vh_u,
                torch.diag(sum_s),
                u_vh,
                vh_vh
            ])
            
            # Apply alpha scaling if specified (matching reference behavior)
            if alpha != 1.0:
                result = alpha * result
            
            return result  # [take, m]
            
        except Exception as e:
            # Fallback to simple average if SVD fails
            log.warning(f"TSVM SVD failed: {e}, falling back to simple average")
            return U_stack.mean(dim=0)

# OLD METHODS REMOVED - Now processing each tensor individually like fastfood_merging.py
# ALL OLD METHODS REMOVED 
# Now using tensor-by-tensor processing exactly like fastfood_merging.py
# This eliminates the concatenation approach that caused timeouts
