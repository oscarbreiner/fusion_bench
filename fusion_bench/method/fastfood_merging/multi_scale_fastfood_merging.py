# multi_scale_fastfood_merging.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

import torch
from torch import nn, Tensor

from fusion_bench.utils import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType

# Import utilities from fastfood_utils
from .fastfood_utils import (
    EPS,
    create_fastfood_ops,
    zero_aware_aggregate,
    layer_key,
)

# Keep backward compatibility
_fastfood_ops = create_fastfood_ops
_zero_aware_aggregate = zero_aware_aggregate
_layer_key = layer_key


@torch.no_grad()
def _multi_scale_aggregate(
    merged_results: List[Tensor],
    proj_dims: List[int],
    postlift_func: str,
    weights: List[float] | None = None,
    **kwargs
) -> Tensor:
    """
    Aggregate multiple post-lift merge results from different projection dimensions.
    
    Args:
        merged_results: List of merged tensors from different projection dimensions
        proj_dims: List of projection dimensions used (for weighting/debugging)
        postlift_func: Aggregation function for post-lift results
        weights: Optional weights for different projection scales
        **kwargs: Additional parameters for aggregation functions
        
    Returns:
        Final aggregated tensor
    """
    if len(merged_results) == 0:
        raise ValueError("No merge results to aggregate")
    if len(merged_results) == 1:
        return merged_results[0]
    
    # Stack all results: [num_scales, ...]
    U_stack = torch.stack(merged_results, dim=0)
    
    # Use the same zero-aware aggregation but with scale-based weights if needed
    scale_weights = None
    if weights is not None and len(weights) == len(merged_results):
        scale_weights = weights
    elif postlift_func in ["weighted_sum", "weighted_sum_proj", "weighted_sum_inv_proj"] and proj_dims is not None:
        if postlift_func == "weighted_sum" or postlift_func == "weighted_sum_proj":
            # Weight by projection dimension (higher dim = more information preserved)
            scale_weights = [float(dim) for dim in proj_dims]
        elif postlift_func == "weighted_sum_inv_proj":
            # Weight by inverse projection dimension (lower dim = higher weight)
            scale_weights = [1.0 / float(dim) for dim in proj_dims]
        
        # Normalize weights to sum to 1
        total_weight = sum(scale_weights)
        scale_weights = [w / total_weight for w in scale_weights]
    
    # Map postlift_func to standard aggregation functions
    if postlift_func in ["weighted_sum", "weighted_sum_proj", "weighted_sum_inv_proj"]:
        # Use sum with weights (all weighted_sum variants)
        return _zero_aware_aggregate(U_stack, "sum", scale_weights, **kwargs)
    elif postlift_func == "weighted_mean":
        # Use mean with weights
        return _zero_aware_aggregate(U_stack, "mean", scale_weights, **kwargs)
    else:
        # Use the specified function directly
        return _zero_aware_aggregate(U_stack, postlift_func, scale_weights, **kwargs)


@auto_register_config
class MultiScaleFastfoodMergeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Multi-scale task-vector merging via Fastfood/SRHT subspaces.
    
    This algorithm projects task vectors into multiple subspaces of different dimensions,
    performs aggregation within each subspace, lifts back to original space, then
    performs a second aggregation across all post-lift results.
    
    Controls:
      proj_dims: List of projection dimensions (or ratios if < 1.0)
      subspace_func: Aggregation function within each subspace
      postlift_func: Aggregation function across post-lift results  
      subspace_scope: "per_tensor" | "layer" | "global"
      use_G: bool (FastFood Gaussian scaling)
      block_rows: int (memory management)
      weights: Optional task importance weights
      scale_weights: Optional weights for different projection scales
      scale: Global scaling factor
      
    All other parameters mirror the original FastfoodSubspaceMergeAlgorithm.
    """

    def __init__(
        self,
        proj_dims: List[float] = [0.25, 0.50, 0.75],  # Projection dimensions/ratios
        subspace_func: str = "signmax",               # Aggregation within subspace
        postlift_func: str = "weighted_mean",         # Aggregation across scales
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",               # "per_tensor" | "layer" | "global"
        block_rows: int = 8192,
        weights: List[float] | None = None,           # Task weights
        scale_weights: List[float] | None = None,     # Scale weights
        scale: float = 1.0,
        
        # EMA-specific parameters for subspace aggregation
        ema_task_order: str = "given",
        ema_gamma: float = 1.2,
        ema_w_c: float = 0.6,
        ema_w_s: float = 0.4,
        ema_custom_order: List[str] | None = None,
        
        # Analysis integration parameters
        run_analysis: bool = False,
        analysis_methods: List[str] = None,
        analysis_output_path: str = None,
        
        # Compatibility parameters
        ties_trim_pct: float = 0.0,
        tadrop_tau: float = 0.0,
        use_pareto: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        
        # Multi-scale specific parameters
        if not proj_dims:
            raise ValueError("proj_dims cannot be empty")
        
        # Handle string-to-list conversion for Hydra compatibility
        if isinstance(proj_dims, str):
            # Parse comma-separated string: "0.25,0.5,0.75" -> [0.25, 0.5, 0.75]
            self.proj_dims = [float(x.strip()) for x in proj_dims.split(',')]
        else:
            self.proj_dims = list(proj_dims)
        self.subspace_func = str(subspace_func).lower()
        self.postlift_func = str(postlift_func).lower()
        
        # Core FastFood parameters
        self.use_G = bool(use_G)
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        self.block_rows = int(block_rows)
        
        # Weights and scaling
        # Handle weights string-to-list conversion for Hydra compatibility
        if weights is not None:
            if isinstance(weights, str):
                # Parse comma-separated string: "1.0,2.0,3.0" -> [1.0, 2.0, 3.0]
                self.weights = [float(x.strip()) for x in weights.split(',')]
            else:
                self.weights = list(weights)
        else:
            self.weights = None
        
        # Handle scale_weights string-to-list conversion for Hydra compatibility
        if scale_weights is not None:
            if isinstance(scale_weights, str):
                # Parse comma-separated string: "1.0,2.0,3.0" -> [1.0, 2.0, 3.0]
                self.scale_weights = [float(x.strip()) for x in scale_weights.split(',')]
            else:
                self.scale_weights = list(scale_weights)
        else:
            self.scale_weights = None
            
        self.scale = float(scale)
        
        # EMA parameters for subspace aggregation
        self.ema_task_order = str(ema_task_order)
        self.ema_gamma = float(ema_gamma)
        self.ema_w_c = float(ema_w_c)
        self.ema_w_s = float(ema_w_s)
        # Handle ema_custom_order string-to-list conversion for Hydra compatibility
        if ema_custom_order is not None:
            if isinstance(ema_custom_order, str):
                # Parse comma-separated string: "task1,task2,task3" -> ["task1", "task2", "task3"]
                self.ema_custom_order = [x.strip() for x in ema_custom_order.split(',')]
            else:
                self.ema_custom_order = list(ema_custom_order)
        else:
            self.ema_custom_order = None
        
        # Analysis parameters
        self.run_analysis = run_analysis
        self.analysis_methods = analysis_methods or []
        self.analysis_output_path = analysis_output_path
        
        # Validate parameters        
        if self.scale_weights is not None and len(self.scale_weights) != len(self.proj_dims):
            raise ValueError(f"scale_weights length ({len(self.scale_weights)}) must match proj_dims length ({len(self.proj_dims)})")

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
        print(f"[MultiScale Setup] donors={K} | total tensors={len(keys_all)} | eligible float tensors={len(keys_float)}")

        if not keys_float:
            raise RuntimeError("No overlapping float tensors with identical shapes. Nothing to merge.")

        # ---------- Task Weights ----------
        if self.weights is None:
            w = [1.0 / K] * K
        else:
            if len(self.weights) != K:
                raise ValueError("`weights` length must match number of donors.")
            s = sum(self.weights) + EPS
            w = [wi / s for wi in self.weights]

        # ---------- EMA Custom Order Mapping ----------
        ema_custom_indices = None
        if self.subspace_func == "ema" and self.ema_task_order == "custom":
            if self.ema_custom_order is None:
                raise ValueError("ema_custom_order must be provided when ema_task_order='custom'")
            
            name_to_idx = {name: i for i, name in enumerate(donor_names)}
            try:
                ema_custom_indices = [name_to_idx[task_name] for task_name in self.ema_custom_order]
            except KeyError as e:
                available_names = list(donor_names)
                raise ValueError(f"Task name {e} in ema_custom_order not found in donor names. Available: {available_names}")
            
            if len(ema_custom_indices) != K:
                raise ValueError(f"ema_custom_order must include all {K} tasks, got {len(ema_custom_indices)}")
            if set(ema_custom_indices) != set(range(K)):
                raise ValueError(f"ema_custom_order must be a permutation of all task indices, got {ema_custom_indices}")
            
            print(f"[MultiScale EMA] Custom task order: {self.ema_custom_order} -> indices {ema_custom_indices}")

        # ---------- Seed scoping function ----------
        def proj_seed_key(param_name: str) -> str:
            if self.subspace_scope == "global":
                return "__GLOBAL__"
            if self.subspace_scope == "layer":
                return _layer_key(param_name)
            return param_name  # per_tensor

        # ---------- Determine global D if needed ----------
        global_D = None
        if self.subspace_scope == "global":
            maxd = 1
            for k in keys_float:
                t = base_sd[k]
                maxd = max(maxd, int(t.shape[-1]))
            global_D = maxd

        # ---------- Report multi-scale configuration ----------
        def _compute_proj_dims(k: str) -> Tuple[int, List[int]]:
            d_last = int(base_sd[k].shape[-1])
            cur_D = global_D if (global_D is not None) else d_last
            
            actual_dims = []
            for dim in self.proj_dims:
                if dim < 1.0:  # Ratio
                    actual_dim = max(1, int(cur_D * dim))
                else:  # Absolute dimension
                    actual_dim = max(1, int(dim))
                actual_dims.append(min(actual_dim, cur_D))  # Cap at cur_D
            
            return cur_D, actual_dims

        # Show example configurations
        ex = keys_float[:3]
        if ex:
            print(f"[MultiScale Config] scope={self.subspace_scope} | proj_dims={self.proj_dims} | subspace_func={self.subspace_func} | postlift_func={self.postlift_func}")
            for k in ex:
                cur_D, actual_dims = _compute_proj_dims(k)
                compressions = [f"{dim}/{cur_D}({dim/max(1,cur_D):.3f})" for dim in actual_dims]
                print(f"   - {k}: original_last_dim={int(base_sd[k].shape[-1])} | scoped_dim={cur_D} → proj_dims={actual_dims} | compressions={compressions}")

        # ---------- Work on CPU copies ----------
        base_cpu = {k: v.detach().cpu().clone() for k, v in base_sd.items()}
        donors_cpu = [{k: v.detach().cpu().clone() for k, v in d.items()} for d in donors_sd]
        dev = self.device

        # ---------- Multi-scale merge ----------
        merged_tensors = 0
        changed_params = 0

        # Operator cache keyed by (seed_key, cur_D, proj_dim)
        op_cache: Dict[Tuple[str, int, int], Tuple[Any, Any]] = {}

        with self.profile("multi-scale merging"):
            for name in keys_float:
                tb = base_cpu[name]
                d_last = int(tb.shape[-1])
                rows = tb.numel() // d_last
                if rows <= 0:
                    continue

                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                # Determine scoped dimension and actual projection dimensions
                seed_key = proj_seed_key(name)
                cur_D, actual_proj_dims = _compute_proj_dims(name)

                cursor = 0
                tensor_changed = False

                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor:cursor + take, :]

                    # Prepare donor deltas
                    Xs: List[Tensor] = []
                    for dsd in donors_cpu:
                        sl_donor = dsd[name].view(rows, d_last).float()[cursor:cursor + take, :]
                        delta = sl_donor - sl_base
                        if global_D is not None and d_last < cur_D:
                            buf = torch.zeros((take, cur_D), dtype=torch.float32, device="cpu")
                            buf[:, :d_last].copy_(delta)
                            Xs.append(buf)
                        else:
                            Xs.append(delta)

                    # ---------- Multi-scale processing ----------
                    scale_results = []
                    
                    for proj_dim in actual_proj_dims:
                        # Get or create FastFood operator for this scale
                        cache_key = (seed_key, cur_D, proj_dim)
                        if cache_key not in op_cache:
                            fwd, lift = _fastfood_ops(
                                cur_D, proj_dim, seed_key=seed_key, device=dev, use_G=self.use_G
                            )
                            op_cache[cache_key] = (fwd, lift)
                        else:
                            fwd, lift = op_cache[cache_key]

                        # Project all donors to this subspace
                        Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]

                        # Aggregate in subspace
                        U_stack = torch.stack(Ys, dim=0)  # [K, take, proj_dim]
                        Ymerged = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.subspace_func,
                            weights=w if self.subspace_func in {"sum", "mean", "ema"} else None,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        )

                        # Lift back to original space
                        Xmerged_full = lift(Ymerged).to("cpu", non_blocking=True)  # [take, cur_D]
                        Xmerged = Xmerged_full[:, :d_last]
                        
                        scale_results.append(Xmerged)

                    # ---------- Aggregate across scales ----------
                    if len(scale_results) == 1:
                        Xfinal = scale_results[0]
                    else:
                        Xfinal = _multi_scale_aggregate(
                            scale_results,
                            actual_proj_dims,
                            self.postlift_func,
                            weights=self.scale_weights,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        )

                    # Write back with scaling
                    upd = (self.scale * Xfinal).to(sl_base.dtype)
                    sl_base.add_(upd)

                    # Track changes
                    tensor_changed = tensor_changed or bool(upd.abs().max().item() > 0)

                    cursor += take
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                merged_tensors += 1
                if tensor_changed:
                    changed_params += 1

        # ---------- Stats and validation ----------
        bad, total_float = [], 0
        for n, p in base_cpu.items():
            if not torch.is_floating_point(p):
                continue
            total_float += 1
            if torch.isnan(p).any() or torch.isinf(p).any():
                bad.append(n)

        # Scale drift analysis
        drift_hi, drift_lo = [], []
        for k in keys_float:
            pre = base_sd[k].float()
            now = base_cpu[k].float()
            nb = float(pre.norm().item()) + 1e-12
            na = float(now.norm().item())
            ratio = na / nb
            if ratio > 10.0:
                drift_hi.append((k, ratio))
            elif ratio < 0.1:
                drift_lo.append((k, ratio))

        drift_hi.sort(key=lambda x: x[1], reverse=True)
        drift_lo.sort(key=lambda x: x[1])

        # Print summary
        print("\n=== Multi-Scale Merge Summary ===")
        print(f"[MultiScale] donors={K} | scales={len(self.proj_dims)} | eligible_tensors={len(keys_float)} | processed={merged_tensors} | changed_tensors={changed_params}")
        print(f"[MultiScale] projection_dimensions={self.proj_dims} | subspace_func={self.subspace_func} | postlift_func={self.postlift_func}")
        
        if bad:
            print(f"[MultiScale] ⚠️ NaN/Inf in {len(bad)}/{total_float} float tensors (showing up to 10):")
            for n in bad[:10]:
                print("  -", n)
        else:
            print(f"[MultiScale] ✓ No NaN/Inf across {total_float} float tensors.")

        print(f"[MultiScale] Large ↑ drift (>10x): {len(drift_hi)} | Large ↓ drift (<0.1x): {len(drift_lo)}")
        for name, r in drift_hi[:5]:
            print(f"   HI  {r:.3f}  {name}")
        for name, r in drift_lo[:5]:
            print(f"   LO  {r:.3f}  {name}")

        self.print_profile_summary()

        # ---------- Load merged state back ----------
        if isinstance(base_model, nn.Module):
            model = base_model
            model.load_state_dict(
                {k: v if not torch.is_floating_point(v) else v for k, v in base_cpu.items()},
                strict=False,
            )
        elif isinstance(base_model, LazyStateDict):
            model = base_model.meta_module.to_empty(device=base_model._device)
            result = model.load_state_dict({k: v for k, v in base_cpu.items()}, strict=False)
            if result.unexpected_keys:
                raise ValueError(f"Unexpected keys: {result.unexpected_keys}")
        else:
            raise TypeError(f"Unsupported model type: {type(base_model)}")

        if merged_tensors == 0:
            raise RuntimeError("No tensors were processed; check eligibility filters.")
        if changed_params == 0:
            print("⚠️ Note: processed tensors but no numeric changes detected.")

        # ---------- Run Analysis (if enabled) ----------
        if self.run_analysis and self.analysis_methods:
            print("\n=== Running Integrated Analysis ===")
            self._run_integrated_analysis(modelpool, model)

        return model

    def _run_integrated_analysis(self, modelpool: BaseModelPool, merged_model: nn.Module):
        """Run integrated analysis methods after multi-scale merging."""
        print(f"[MultiScale Analysis] Running {len(self.analysis_methods)} analysis methods")
        
        # Create unique method identifier
        method_id = self._create_method_identifier()
        
        for analysis_method in self.analysis_methods:
            try:
                print(f"[MultiScale Analysis] Running {analysis_method}")
                
                if analysis_method == "merged_task_vector":
                    self._run_merged_task_vector_analysis(modelpool, method_id)
                elif analysis_method == "task_vector_similarity":
                    self._run_task_vector_similarity_analysis(modelpool, method_id)
                elif analysis_method == "task_vector_layer":
                    self._run_task_vector_layer_analysis(modelpool, method_id)
                else:
                    print(f"[MultiScale Analysis] Warning: Unknown analysis method '{analysis_method}'")
                    
            except Exception as e:
                print(f"[MultiScale Analysis] Error in {analysis_method}: {e}")
                import traceback
                traceback.print_exc()
    
    def _create_method_identifier(self) -> str:
        """Create unique method identifier for analysis outputs."""
        parts = ["multiscale_fastfood"]
        
        # Add projection dimensions info
        proj_str = "_".join([f"{d}" for d in self.proj_dims])
        parts.append(f"dims[{proj_str}]")
        
        # Add function info
        parts.append(f"sub_{self.subspace_func}")
        parts.append(f"post_{self.postlift_func}")
        
        if self.subspace_scope != 'global':
            parts.append(f"scope_{self.subspace_scope}")
            
        return "_".join(parts)
    
    def _run_merged_task_vector_analysis(self, modelpool: BaseModelPool, method_id: str):
        """Run merged task vector analysis for multi-scale method."""
        try:
            from fusion_bench.method.analysis.merged_task_vector_analysis import MergedTaskVectorAnalysis
            
            analyzer = MergedTaskVectorAnalysis(
                merging_methods=["multi_scale_fastfood_merging"],
                trainable_only=True,
                output_path=self.analysis_output_path,
                device=str(self.device)
            )
            
            print(f"[MultiScale Analysis] Running merged task vector analysis for {method_id}")
            analyzer.run(modelpool)
            
        except ImportError as e:
            print(f"[MultiScale Analysis] Could not import MergedTaskVectorAnalysis: {e}")
    
    def _run_task_vector_similarity_analysis(self, modelpool: BaseModelPool, method_id: str):
        """Run task vector similarity analysis."""
        try:
            from fusion_bench.method.analysis.task_vector_cos_similarity import TaskVectorCosSimilarity
            
            analyzer = TaskVectorCosSimilarity(
                plot_heatmap=True,
                trainable_only=True,
                method_name=method_id,
                device=str(self.device),
                output_path=self.analysis_output_path
            )
            
            print(f"[MultiScale Analysis] Running task vector similarity analysis for {method_id}")
            analyzer.run(modelpool)
            
        except ImportError as e:
            print(f"[MultiScale Analysis] Could not import TaskVectorCosSimilarity: {e}")
    
    def _run_task_vector_layer_analysis(self, modelpool: BaseModelPool, method_id: str):
        """Run layer-wise task vector analysis."""
        try:
            from fusion_bench.method.analysis.task_vector_layer_analysis import TaskVectorLayerAnalysis
            
            analyzer = TaskVectorLayerAnalysis(
                trainable_only=True,
                method_name=method_id,
                device=str(self.device),
                output_path=self.analysis_output_path
            )
            
            print(f"[MultiScale Analysis] Running layer-wise task vector analysis for {method_id}")
            analyzer.run(modelpool)
            
        except ImportError as e:
            print(f"[MultiScale Analysis] Could not import TaskVectorLayerAnalysis: {e}")