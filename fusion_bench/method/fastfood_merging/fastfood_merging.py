# fastfood_merging.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import fnmatch

import torch
from torch import nn, Tensor
import math

from fusion_bench.utils import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_sub, state_dict_mul

# Import utilities from fastfood_utils
from .fastfood_utils import (
    EPS,
    create_fastfood_ops,
    zero_aware_aggregate,
    layer_key,
    normalize_weights,
    compute_global_dim
)

# Keep backward compatibility by re-exporting under old names
_fastfood_ops = create_fastfood_ops
_zero_aware_aggregate = zero_aware_aggregate
_layer_key = layer_key


# ------------------ Algorithm ------------------
@auto_register_config
class FastfoodSubspaceMergeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Task-vector merging via Fastfood/SRHT subspaces.

    Controls:
      subspace_scope: "per_tensor" | "per_flat_tensor" | "layer" | "global"
                      - "per_tensor": row-wise projection on 2D tensors (rows x in_dim)
                      - "per_flat_tensor": flatten 2D tensors to (out·in) and apply one projection
                      - "layer": layer-wise projection
                      - "global": global projection across all parameters
      merge_where:    "subspace" | "postlift"
      merge_func:     "sum" | "mean" | "max" | "signmax" | "ema" | "ties_sum" | "ties_mean" | "ties_max"
      proj_ratio:     float (0..1)
      use_G:          bool
      block_rows:     int
      weights:        list[float] (donor weights; normalized internally)
      scale:          float (post-merge scale on Δ*)
      
      # Task Arithmetic Reconstruction mode:
      use_task_arithmetic_reconstruction: bool
      task_arithmetic_scaling: float
      report_reconstruction_error: bool
      
      # Weight matching parameters:
      use_weight_matching: bool
      weight_matching_max_iter: int
      weight_matching_seed: int
      weight_matching_verbose: bool
      weight_matching_input_shapes: tuple
      
      # EMA-specific parameters (when merge_func="ema"):
      ema_task_order: "given" | "random" | "cosine_similarity" | "custom"
      ema_gamma: float
      ema_w_c: float
      ema_w_s: float
      ema_custom_order: list[str]
      
      # TSV-style linear/non-linear separation:
      only_project_linear: bool (if True, only 2D tensors are projected; 1D tensors use mean in original space)
      
      ties_trim_pct, tadrop_tau, use_pareto: kept for API compatibility (unused)
    """

    def __init__(
        self,
        proj_ratio: float = 0.10,
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",  # "per_tensor" | "per_flat_tensor" | "layer" | "global"
        merge_where: str = "subspace",   # "subspace" | "postlift"
        merge_func: str = "sum",         # "sum" | "mean" | "max" | "signmax" | "ema"
        block_rows: int = 8192,
        weights: List[float] | None = None,
        scale: float = 1.0,
        # Task Arithmetic Reconstruction mode parameters
        use_task_arithmetic_reconstruction: bool = False,
        task_arithmetic_scaling: float = 1.0,
        report_reconstruction_error: bool = False,
        # EMA-specific parameters
        ema_task_order: str = "given",   # "given" | "random" | "cosine_similarity" | "custom"
        ema_gamma: float = 1.2,          # sigmoid scaling factor
        ema_w_c: float = 0.6,            # cosine alignment weight
        ema_w_s: float = 0.4,            # scale ratio weight
        ema_custom_order: List[str] | None = None,  # task names in desired order (when ema_task_order="custom")
        # Weight matching parameters (optional preprocessing)
        use_weight_matching: bool = False,  # Enable weight matching preprocessing
        weight_matching_max_iter: int = 100,  # Max iterations for weight matching
        weight_matching_seed: int = 0,  # Seed for weight matching permutation order
        weight_matching_verbose: bool = True,  # Verbose output for weight matching
        weight_matching_input_shapes: Tuple[Tuple[int, ...], ...] | None = None,  # Input shapes for spec generation
        # Analysis integration parameters
        run_analysis: bool = False,
        analysis_methods: List[str] = None,
        analysis_output_path: str = None,
        # TSV-style linear/non-linear separation
        only_project_linear: bool = False,  # Only project 2D tensors; merge 1D tensors in original space
        # Kept in signature for compatibility (not used)
        ties_trim_pct: float = 0.0,
        tadrop_tau: float = 0.0,
        use_pareto: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        assert merge_where in {"subspace", "postlift"}
        self.merge_where = merge_where
        self.merge_func = str(merge_func).lower()
        self.block_rows = int(block_rows)
        self.weights = list(weights) if weights is not None else None
        self.scale = float(scale)
        
        # Task Arithmetic Reconstruction parameters
        self.use_task_arithmetic_reconstruction = bool(use_task_arithmetic_reconstruction)
        self.task_arithmetic_scaling = float(task_arithmetic_scaling)
        self.report_reconstruction_error = bool(report_reconstruction_error)
        
        # EMA parameters
        self.ema_task_order = str(ema_task_order)
        self.ema_gamma = float(ema_gamma)
        self.ema_w_c = float(ema_w_c)
        self.ema_w_s = float(ema_w_s)
        self.ema_custom_order = list(ema_custom_order) if ema_custom_order is not None else None
        
        # Weight matching parameters
        self.use_weight_matching = bool(use_weight_matching)
        self.weight_matching_max_iter = int(weight_matching_max_iter)
        self.weight_matching_seed = int(weight_matching_seed)
        self.weight_matching_verbose = bool(weight_matching_verbose)
        self.weight_matching_input_shapes = weight_matching_input_shapes
        
        # TSV-style linear/non-linear separation
        self.only_project_linear = bool(only_project_linear)
        
        # Analysis parameters
        self.run_analysis = run_analysis
        self.analysis_methods = analysis_methods or []
        self.analysis_output_path = analysis_output_path

    # ------------------- main -------------------
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

        # ---------- Weight Matching Preprocessing (Optional) ----------
        if self.use_weight_matching:
            print("\n=== Weight Matching Preprocessing ===")
            print(f"[Weight Matching] Aligning {len(donor_names)} donors to base model")
            print(f"[Weight Matching] max_iter={self.weight_matching_max_iter}, seed={self.weight_matching_seed}")
            
            with self.profile("weight_matching"):
                # Import weight matching utilities from local package
                try:
                    from .weight_matching.core import get_permutation_spec
                    from .weight_matching.weight_matching import weight_matching
                    from .weight_matching.core import apply_perm
                except ImportError as e:
                    raise ImportError(
                        f"Weight matching modules not found. Error: {e}\n"
                        "Make sure weight_matching package is properly installed."
                    )
                
                # Get permutation specification from base model
                # Determine input shapes - use provided or default to vision model
                try:
                    if self.weight_matching_input_shapes is not None:
                        input_shapes = self.weight_matching_input_shapes
                    else:
                        # Default to standard vision model input (batch=1, RGB, 224x224)
                        input_shapes = ((1, 3, 224, 224),)
                        print(f"[Weight Matching] Using default input shapes: {input_shapes}")
                    
                    spec = get_permutation_spec(
                        base_model,
                        input_shapes,
                        verbose=False
                    )
                    print(f"[Weight Matching] Found {len(spec)} permutation groups")
                except Exception as e:
                    print(f"[Weight Matching] Warning: Could not generate permutation spec: {e}")
                    print(f"[Weight Matching] This may be due to incompatible model architecture or input shapes.")
                    print(f"[Weight Matching] Skipping weight matching preprocessing")
                    spec = None
                
                if spec is not None:
                    # Apply weight matching to align each donor to the base
                    for i, donor_name in enumerate(donor_names):
                        print(f"[Weight Matching] Aligning donor {i+1}/{len(donor_names)}: {donor_name}")
                        
                        # Run weight matching algorithm (matches original PLeaS implementation)
                        perm = weight_matching(
                            spec=spec,
                            state_as=base_sd,  # Reference model (base)
                            state_bs=donors_sd[i],  # Model to align (donor)
                            max_iter=self.weight_matching_max_iter,
                            init_perm=None,  # Start with identity permutation
                            inplace=True,  # Modify donors_sd[i] in place
                            skip_suffixes=("running_mean", "running_var"),  # Skip BN stats
                            skip_missing=True,  # Skip missing parameters gracefully
                            verbose=self.weight_matching_verbose,
                            seed=self.weight_matching_seed,
                            return_costs=False,
                        )
                        
                        print(f"[Weight Matching] ✓ Aligned {donor_name}")
                    
                    print("[Weight Matching] All donors aligned to base model")
                    print("=" * 50 + "\n")

        # ---------- Task Arithmetic Reconstruction Mode (Optional) ----------
        if self.use_task_arithmetic_reconstruction:
            print("\n=== Task Arithmetic Reconstruction Mode ===")
            print(f"[Task Arithmetic] Merging {len(donor_names)} task vectors with scaling factor {self.task_arithmetic_scaling}")
            print(f"[Task Arithmetic] Will project/lift the MERGED TASK VECTOR (not the whole model)")
            
            with self.profile("task_arithmetic"):
                # Calculate task vectors: Δᵢ = θᵢ - θ₀ (convert to plain dicts to avoid type issues)
                base_dict = {k: v.data if hasattr(v, 'data') else v for k, v in base_sd.items()}
                donors_dict = [{k: v.data if hasattr(v, 'data') else v for k, v in d.items()} for d in donors_sd]
                
                task_vector = None
                for i, donor_name in enumerate(donor_names):
                    print(f"[Task Arithmetic] Processing task {i+1}/{len(donor_names)}: {donor_name}")
                    
                    # Compute task vector for this donor
                    donor_tv = state_dict_sub(donors_dict[i], base_dict)
                    
                    # Accumulate task vectors
                    if task_vector is None:
                        task_vector = donor_tv
                    else:
                        task_vector = state_dict_add(task_vector, donor_tv)
                
                # Scale the aggregated task vector
                task_vector = state_dict_mul(task_vector, self.task_arithmetic_scaling)
                
                print(f"[Task Arithmetic] Merged task vector computed: θ_tv = λ·Σᵢ(θᵢ - θ₀)")
                
                # Now test projection regularization: project/lift the TASK VECTOR only
                if self.report_reconstruction_error or self.proj_ratio < 1.0:
                    print(f"\n[Projection Test] Projecting task vector with proj_ratio={self.proj_ratio}")
                    
                    # Prepare task vector for projection
                    tv_cpu = {k: v.detach().cpu().clone() for k, v in task_vector.items()}
                    dev = self.device
                    
                    # Track reconstruction error
                    total_reconstruction_error = 0.0
                    total_norm = 0.0
                    reconstruction_errors = []
                    
                    # Identify eligible parameters
                    keys_float_ta = [
                        k for k in tv_cpu.keys()
                        if torch.is_floating_point(tv_cpu[k])
                        and tv_cpu[k].ndim >= 1
                    ]
                    
                    # operator cache
                    op_cache_ta: Dict[Tuple[str, int, int], Tuple[Any, Any]] = {}
                    
                    def proj_seed_key_ta(param_name: str) -> str:
                        if self.subspace_scope == "global":
                            return "global"
                        elif self.subspace_scope == "layer":
                            return layer_key(param_name)
                        else:
                            return param_name
                    
                    # Project each task vector tensor down and up
                    for name in keys_float_ta:
                        tb = tv_cpu[name]
                        d_last = int(tb.shape[-1])
                        rows = tb.numel() // d_last
                        if rows <= 0:
                            continue
                        
                        original_tv = tb.view(rows, d_last).float()
                        
                        # Determine dimension based on scope
                        seed_key = proj_seed_key_ta(name)
                        
                        if self.subspace_scope == "per_tensor":
                            cur_D = d_last
                        elif self.subspace_scope == "layer":
                            # For layer scope, use actual dimension (conservative)
                            cur_D = d_last
                        else:  # global
                            # For global scope, compute max dimension across eligible params
                            if 'global_D_ta' not in locals():
                                global_D_ta = max(tv_cpu[k].shape[-1] for k in keys_float_ta)
                                print(f"[Projection Test] Global dimension: {global_D_ta}")
                            cur_D = global_D_ta
                        
                        # Projection size based on actual dimension
                        proj_dim = max(1, int(cur_D * self.proj_ratio))
                        cache_key = (seed_key, cur_D, proj_dim)
                        
                        if cache_key not in op_cache_ta:
                            fwd, lift = create_fastfood_ops(
                                cur_D, proj_dim, seed_key=seed_key, device=dev, use_G=self.use_G, k_min=self.k_min,
                                correct_for_pow2_padding=self.correct_for_pow2_padding
                            )
                            op_cache_ta[cache_key] = (fwd, lift)
                        else:
                            fwd, lift = op_cache_ta[cache_key]
                        
                        # Process in blocks
                        reconstructed_tv = torch.zeros_like(original_tv)
                        br = min(self.block_rows, rows)
                        cursor = 0
                        
                        while cursor < rows:
                            take = min(rows - cursor, br)
                            sl = original_tv[cursor:cursor + take, :]
                            
                            # Only pad if using shared operator sized for larger dimension
                            if cur_D > d_last:
                                sl_input = torch.nn.functional.pad(sl, (0, cur_D - d_last))
                            else:
                                sl_input = sl
                            
                            # Project down and up
                            Y = fwd(sl_input.to(dev, non_blocking=True))
                            X_rec = lift(Y).to("cpu", non_blocking=True)
                            
                            # Extract only original dimension if padded
                            if cur_D > d_last:
                                X_rec = X_rec[:, :d_last]
                            
                            reconstructed_tv[cursor:cursor + take, :] = X_rec
                            cursor += take
                        
                        # Compute reconstruction error for this task vector tensor
                        diff = reconstructed_tv - original_tv
                        tensor_error = float(diff.pow(2).sum().item())
                        tensor_norm = float(original_tv.pow(2).sum().item())
                        
                        total_reconstruction_error += tensor_error
                        total_norm += tensor_norm
                        
                        if tensor_norm > 0:
                            relative_error = tensor_error / tensor_norm
                            reconstruction_errors.append((name, relative_error))
                        
                        # Update task vector with reconstructed version
                        tv_cpu[name] = reconstructed_tv.view(tb.shape).to(tb.dtype)
                    
                    # Report reconstruction error
                    if total_norm > 0:
                        global_relative_error = total_reconstruction_error / total_norm
                        print(f"\n[Reconstruction Error] Global relative error: {global_relative_error:.6e}")
                        print(f"[Reconstruction Error] Total squared error: {total_reconstruction_error:.6e}")
                        print(f"[Reconstruction Error] Total norm: {total_norm:.6e}")
                        
                        # Sort and show top errors
                        reconstruction_errors.sort(key=lambda x: x[1], reverse=True)
                        print(f"\n[Reconstruction Error] Top 10 tensors by relative error:")
                        for i, (name, rel_err) in enumerate(reconstruction_errors[:10], 1):
                            print(f"  {i}. {name}: {rel_err:.6e}")
                    
                    # Now add reconstructed task vector to base model: θ* = θ₀ + lift(project(Δ*))
                    merged_sd = state_dict_add(base_dict, tv_cpu)
                    base_cpu = {k: v.detach().cpu().clone() for k, v in merged_sd.items()}
                else:
                    # No projection - just add original task vector to base
                    merged_sd = state_dict_add(base_dict, task_vector)
                    base_cpu = {k: v.detach().cpu().clone() for k, v in merged_sd.items()}
                
                print("=" * 50 + "\n")
                
                # Skip normal merging loop - we're done
                print("[Task Arithmetic] Reconstruction test complete, skipping normal Fastfood merging")
                merged_tensors = len(keys_float_ta) if self.report_reconstruction_error or self.proj_ratio < 1.0 else 0
                changed_params = merged_tensors
                
                # Jump to loading section
                if isinstance(base_model, nn.Module):
                    model = base_model
                    model.load_state_dict(base_cpu)
                elif isinstance(base_model, LazyStateDict):
                    from copy import deepcopy
                    model = deepcopy(base_model.meta_module)
                    model = model.to_empty(device=base_model._device)
                    result = model.load_state_dict(base_cpu, strict=False)
                    if result.unexpected_keys:
                        raise ValueError(f"Unexpected keys in state dict: {result.unexpected_keys}")
                    if result.missing_keys:
                        print(f"Warning: Missing keys in state dict: {result.missing_keys}")
                else:
                    raise TypeError(f"Unsupported model type: {type(base_model)}")
                
                self.print_profile_summary()
                return model

                # ---------- Eligible tensors ----------
        keys_all = list(base_sd.keys())
        keys_float = [
            k for k in keys_all
            if (k in donors_sd[0])
            and torch.is_floating_point(base_sd[k])
            and base_sd[k].ndim >= 1
            and all((k in d) and torch.is_floating_point(d[k]) and d[k].shape == base_sd[k].shape for d in donors_sd)
        ]
        
        # Separate linear (2D) and non-linear (1D or other) weights if only_project_linear is enabled
        if self.only_project_linear:
            keys_linear = [k for k in keys_float if base_sd[k].ndim == 2]
            keys_nonlinear = [k for k in keys_float if base_sd[k].ndim != 2]
            print(f"[only_project_linear] linear (2D) tensors={len(keys_linear)} | non-linear tensors={len(keys_nonlinear)}")
        else:
            keys_linear = keys_float
            keys_nonlinear = []
        
        K = len(donor_names)
        print(f"[Setup] donors={K} | total tensors={len(keys_all)} | eligible float tensors={len(keys_float)}")

        if not keys_float:
            raise RuntimeError("No overlapping float tensors with identical shapes. Nothing to merge.")

        # ---------- Weights ----------
        if self.weights is None:
            w = [1.0 / K] * K
        else:
            if len(self.weights) != K:
                raise ValueError("`weights` length must match number of donors.")
            s = sum(self.weights) + EPS
            w = [wi / s for wi in self.weights]

        # ---------- EMA Custom Order Mapping ----------
        ema_custom_indices = None
        if self.merge_func == "ema" and self.ema_task_order == "custom":
            if self.ema_custom_order is None:
                raise ValueError("ema_custom_order must be provided when ema_task_order='custom'")
            
            # Map task names to indices
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
            
            print(f"[EMA] Custom task order: {self.ema_custom_order} -> indices {ema_custom_indices}")

        # ---------- Seed scoping ----------
        def proj_seed_key(param_name: str) -> str:
            if self.subspace_scope == "global":
                return "__GLOBAL__"
            if self.subspace_scope == "layer":
                return _layer_key(param_name)
            # per_tensor and per_flat_tensor both use param_name as key
            # (they differ in how the tensor is processed, not the seed)
            return param_name

        # Determine global D (max last-dim) if needed
        global_D = None
        if self.subspace_scope == "global":
            maxd = 1
            for k in keys_float:
                t = base_sd[k]
                maxd = max(maxd, int(t.shape[-1]))
            global_D = maxd

        # Report subspace sizing (first few examples)
        def _dim_for(k: str) -> Tuple[int, int]:
            t = base_sd[k]
            if self.subspace_scope == "per_flat_tensor" and t.ndim == 2:
                # For per_flat_tensor, use total number of elements
                cur_D = t.numel()
            else:
                d_last = int(t.shape[-1])
                cur_D = global_D if (global_D is not None) else d_last
            m = max(1, int(cur_D * self.proj_ratio))
            return cur_D, m

        ex = keys_float[:5]
        if ex:
            dims = [(_dim_for(k), k) for k in ex]
            print("[Dims] scope={} | proj_ratio={:.3f} | examples:".format(self.subspace_scope, self.proj_ratio))
            for (D, m), k in dims:
                print(f"   - {k}: original_last_dim={int(base_sd[k].shape[-1])} | scoped_dim={D} → proj_dim={m} (compression={m/max(1,D):.3f})")
        
                # Show EMA parameters if using EMA
        if self.merge_func == "ema":
            print(f"[EMA] task_order={self.ema_task_order} | gamma={self.ema_gamma:.3f} | w_c={self.ema_w_c:.3f} | w_s={self.ema_w_s:.3f}")

        # ---------- Work on CPU copies ----------
        base_cpu = {k: v.detach().cpu().clone() for k, v in base_sd.items()}
        donors_cpu = [{k: v.detach().cpu().clone() for k, v in d.items()} for d in donors_sd]
        dev = self.device

        # ---------- Merge ----------
        merged_tensors = 0
        changed_params = 0

        # operator cache keyed by (seed_key, cur_D, proj_dim)
        op_cache: Dict[Tuple[str, int, int], Tuple[Any, Any]] = {}

        # Small sample for lift error (subspace only)
        lift_err_num = 0.0
        lift_err_den = 0.0

        with self.profile("merging models"):
            # ---------- Process non-linear weights (1D) with mean in original space ----------
            if self.only_project_linear and keys_nonlinear:
                print(f"[Merging non-linear weights] Processing {len(keys_nonlinear)} non-linear tensors with mean in original space")
                for name in keys_nonlinear:
                    # Use running mean like TSV does
                    result = base_cpu[name].clone().float()
                    for i, dsd in enumerate(donors_cpu):
                        donor_val = dsd[name].float()
                        result += (donor_val - result) / (i + 2)  # i+2 because base is index 0, first donor is index 1
                    
                    # Apply scale and write back
                    update = (result - base_cpu[name].float()) * self.scale
                    base_cpu[name].add_(update.to(base_cpu[name].dtype))
                    
                    merged_tensors += 1
                    if update.abs().max().item() > 0:
                        changed_params += 1
            
            # ---------- Process linear weights (2D) with projection ----------
            for name in keys_linear:
                tb = base_cpu[name]
                d_last = int(tb.shape[-1])
                rows = tb.numel() // d_last
                if rows <= 0:
                    continue

                # Special case: per_flat_tensor - flatten entire 2D weight and apply single projection
                if self.subspace_scope == "per_flat_tensor":
                    # Flatten entire tensor to (out_dim * in_dim)
                    flat_dim = tb.numel()
                    vb_flat = tb.view(-1).float()  # [out*in]
                    
                    # Create projection for flattened dimension
                    seed_key = proj_seed_key(name)
                    cur_D = flat_dim
                    proj_dim = max(1, int(cur_D * self.proj_ratio))
                    cache_key = (seed_key, cur_D, proj_dim)
                    
                    if cache_key not in op_cache:
                        fwd, lift = _fastfood_ops(
                            cur_D, proj_dim, seed_key=seed_key, device=dev, use_G=self.use_G
                        )
                        op_cache[cache_key] = (fwd, lift)
                    else:
                        fwd, lift = op_cache[cache_key]
                    
                    # Collect flattened donor deltas
                    Xs: List[Tensor] = []
                    for dsd in donors_cpu:
                        donor_flat = dsd[name].view(-1).float()
                        delta = donor_flat - vb_flat
                        Xs.append(delta)
                    
                    # Project all donors
                    Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]
                    
                    # Mix in chosen space
                    if self.merge_where == "postlift":
                        # Reconstruct donors first, then aggregate in original space
                        Xhats = [lift(Y).to("cpu", non_blocking=False) for Y in Ys]
                        U_stack = torch.stack(Xhats, dim=0)  # [K, flat_dim]
                        Xmerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=w if self.merge_func in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"} else None,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        ).to("cpu", non_blocking=False)
                    else:
                        # Aggregate in subspace, then lift once
                        U_stack = torch.stack(Ys, dim=0)  # [K, proj_dim]
                        Ymerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=w if self.merge_func in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"} else None,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        )
                        Xmerge = lift(Ymerge).to("cpu", non_blocking=False)  # [flat_dim]
                        
                        # Accumulate lift reconstruction error stats
                        if lift_err_den < 1e8:
                            X0 = Xs[0].to(dev, non_blocking=False)
                            Y0 = Ys[0]
                            X0_rec = lift(Y0).to("cpu", non_blocking=False)
                            diff = (X0_rec.to(torch.float32) - Xs[0].to(torch.float32))
                            lift_err_num += float(diff.pow(2).sum().item())
                            lift_err_den += float(Xs[0].pow(2).sum().item())
                    
                    # Apply scale and reshape back to original shape
                    upd = (self.scale * Xmerge).view_as(tb).to(tb.dtype)
                    
                    # Trust-region: prevent blow-ups
                    max_ratio = 2.0
                    upd_norm = upd.norm().item()
                    base_norm = tb.norm().item() + 1e-12
                    if upd_norm > max_ratio * base_norm:
                        scale_factor = (max_ratio * base_norm) / (upd_norm + 1e-12)
                        upd = upd * scale_factor
                    
                    # Write back
                    tb.add_(upd)
                    
                    merged_tensors += 1
                    if upd.abs().max().item() > 0:
                        changed_params += 1
                    
                    continue  # Skip row-wise processing for this tensor
                
                # Standard processing: row-wise projection
                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                # choose scoped dim & build (or reuse) operator
                seed_key = proj_seed_key(name)
                cur_D = global_D if (global_D is not None) else d_last
                proj_dim = max(1, int(cur_D * self.proj_ratio))
                cache_key = (seed_key, cur_D, proj_dim)
                if cache_key not in op_cache:
                    fwd, lift = _fastfood_ops(
                        cur_D, proj_dim, seed_key=seed_key, device=dev, use_G=self.use_G
                    )
                    op_cache[cache_key] = (fwd, lift)
                else:
                    fwd, lift = op_cache[cache_key]

                cursor = 0
                tensor_changed = False

                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor:cursor + take, :]

                    # donor deltas aligned to cur_D if global scope
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

                    # project all donors
                    Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]

                    # mix in chosen space
                    if self.merge_where == "postlift":
                        # reconstruct donors first, then aggregate in original space
                        Xhats_raw = [lift(Y).to("cpu", non_blocking=False) for Y in Ys]
                        # Extract only original dimension if padded
                        Xhats = [X[:, :d_last] if cur_D > d_last else X for X in Xhats_raw]
                        U_stack = torch.stack(Xhats, dim=0)  # [K, take, d]
                        Xmerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=w if self.merge_func in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"} else None,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        ).to("cpu", non_blocking=False)
                    else:
                        # aggregate in subspace, then lift once
                        U_stack = torch.stack(Ys, dim=0)  # [K, take, m]
                        Ymerge = _zero_aware_aggregate(
                            U_stack,
                            merge_func=self.merge_func,
                            weights=w if self.merge_func in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"} else None,
                            ema_task_order=self.ema_task_order,
                            ema_gamma=self.ema_gamma,
                            ema_w_c=self.ema_w_c,
                            ema_w_s=self.ema_w_s,
                            ema_custom_order=ema_custom_indices,
                        )
                        Xmerge_full = lift(Ymerge).to("cpu", non_blocking=False)  # [take, cur_D or d_last]
                        # Extract only original dimension if padded
                        Xmerge = Xmerge_full[:, :d_last] if cur_D > d_last else Xmerge_full

                        # accumulate lift reconstruction error stats on a tiny slice
                        if take > 0 and (lift_err_den < 1e8):  # guard cost
                            # pick the first donor to estimate lift error
                            X0 = Xs[0].to(dev, non_blocking=False)
                            Y0 = Ys[0]
                            X0_rec = lift(Y0).to("cpu", non_blocking=False)
                            # Extract only original dimension if padded
                            if cur_D > d_last:
                                X0_rec = X0_rec[:, :d_last]
                                X0_orig = Xs[0][:, :d_last]
                            else:
                                X0_orig = Xs[0]
                            diff = (X0_rec.to(torch.float32) - X0_orig.to(torch.float32))
                            lift_err_num += float(diff.pow(2).sum().item())
                            lift_err_den += float(X0_orig.pow(2).sum().item())

                    # write back (scale)
                    upd = (self.scale * Xmerge).to(sl_base.dtype)
                    
                    # Trust-region: prevent blow-ups by limiting update magnitude
                    max_ratio = 2.0
                    upd_norm = upd.norm().item()
                    base_norm = sl_base.norm().item() + 1e-12
                    if upd_norm > max_ratio * base_norm:
                        scale_factor = (max_ratio * base_norm) / (upd_norm + 1e-12)
                        upd = upd * scale_factor
                    
                    # Simplified write-back: pinned tensors support in-place CPU ops
                    vb[cursor:cursor + take, :].add_(upd)

                    # did anything change?
                    tensor_changed = tensor_changed or bool(upd.abs().max().item() > 0)

                    cursor += take

                merged_tensors += 1
                if tensor_changed:
                    changed_params += 1
            
            # Clear cache once after all merging is done
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---------- Stats / sanity ----------
        # scan NaN/Inf
        bad, total_float = [], 0
        for n, p in base_cpu.items():
            if not torch.is_floating_point(p):
                continue
            total_float += 1
            if torch.isnan(p).any() or torch.isinf(p).any():
                bad.append(n)

        # scale drift vs original base
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

        # print summary
        print("\n=== Merge Summary ===")
        print(f"[Summary] donors={K} | eligible_tensors={len(keys_float)} | processed={merged_tensors} | changed_tensors={changed_params}")
        if bad:
            print(f"[Summary] ⚠️ NaN/Inf in {len(bad)}/{total_float} float tensors (showing up to 10):")
            for n in bad[:10]:
                print("  -", n)
        else:
            print(f"[Summary] ✓ No NaN/Inf across {total_float} float tensors.")

        if self.merge_where == "subspace" and lift_err_den > 0:
            rel = math.sqrt(lift_err_num) / (math.sqrt(lift_err_den) + EPS)
            print(f"[Summary] Lift reconstruction rel. error (Fro): {rel:.6f}")
        else:
            print("[Summary] Lift reconstruction error: N/A (postlift mixing or no samples).")

        print(f"[Summary] Large ↑ drift (>10x): {len(drift_hi)} | Large ↓ drift (<0.1x): {len(drift_lo)}")
        for name, r in drift_hi[:8]:
            print(f"   HI  {r:.3f}  {name}")
        for name, r in drift_lo[:8]:
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
            print("⚠️ Note: processed tensors but no numeric changes detected (donor deltas may be zero).")

        # ---------- Run Analysis (if enabled) ----------
        if self.run_analysis and self.analysis_methods:
            print("\n=== Running Integrated Analysis ===")
            self._run_integrated_analysis(modelpool, model)

        return model

    def _run_integrated_analysis(self, modelpool: BaseModelPool, merged_model: nn.Module):
        """
        Run integrated analysis methods after merging with reused Fastfood projections.
        
        This method runs analysis using the same Fastfood operators that were used
        for merging, ensuring consistency and avoiding redundant computation.
        
        Args:
            modelpool: The original model pool
            merged_model: The merged model result
        """
        print(f"[Analysis] Running {len(self.analysis_methods)} analysis methods")
        
        # Create unique method identifier for analysis outputs
        method_id = self._create_method_identifier()
        
        for analysis_method in self.analysis_methods:
            try:
                print(f"[Analysis] Running {analysis_method}")
                
                if analysis_method == "merged_task_vector":
                    self._run_merged_task_vector_analysis(modelpool, method_id)
                elif analysis_method == "task_vector_similarity":
                    self._run_task_vector_similarity_analysis(modelpool, method_id)
                elif analysis_method == "task_vector_layer":
                    self._run_task_vector_layer_analysis(modelpool, method_id)
                else:
                    print(f"[Analysis] Warning: Unknown analysis method '{analysis_method}'")
                    
            except Exception as e:
                print(f"[Analysis] Error in {analysis_method}: {e}")
                import traceback
                traceback.print_exc()
    
    def _create_method_identifier(self) -> str:
        """Create unique method identifier for analysis outputs."""
        parts = []
        
        if self.proj_ratio != 0.95:
            parts.append(f"proj{self.proj_ratio}")
        if self.merge_func != 'signmax':
            parts.append(f"{self.merge_func}")
        if self.subspace_scope != 'global':
            parts.append(f"{self.subspace_scope}")
        if self.merge_where != 'subspace':
            parts.append(f"{self.merge_where}")
        
        # Add EMA-specific parameters if using EMA
        if self.merge_func == 'ema':
            parts.append(f"g{self.ema_gamma}")
            parts.append(f"wc{self.ema_w_c}")
            parts.append(f"ws{self.ema_w_s}")
            if self.ema_task_order != 'given':
                order_abbrev = {
                    'random': 'rand',
                    'cosine_similarity': 'cos',
                    'custom': 'cust'
                }
                parts.append(f"{order_abbrev.get(self.ema_task_order, self.ema_task_order)}")
        
        # Add TIES identifier for TIES variants
        if self.merge_func.startswith('ties_'):
            parts.append("ties")
        
        if parts:
            return f"fastfood_{'_'.join(parts)}"
        else:
            return "fastfood_default"
    
    def _run_merged_task_vector_analysis(self, modelpool: BaseModelPool, method_id: str):
        """Run merged task vector analysis with reused Fastfood projections."""
        try:
            from fusion_bench.method.analysis.merged_task_vector_analysis import MergedTaskVectorAnalysis
            
            # Create analyzer with matching parameters
            analyzer = MergedTaskVectorAnalysis(
                merging_methods=["fastfood_merging"],
                proj_ratio=self.proj_ratio,
                use_G=self.use_G,
                merge_func=self.merge_func,
                subspace_scope=self.subspace_scope,
                merge_where=self.merge_where,
                trainable_only=True,
                output_path=self.analysis_output_path,
                device=str(self.device)
            )
            
            print(f"[Analysis] Running merged task vector analysis for {method_id}")
            analyzer.run(modelpool)
            
        except ImportError as e:
            print(f"[Analysis] Could not import MergedTaskVectorAnalysis: {e}")
    
    def _run_task_vector_similarity_analysis(self, modelpool: BaseModelPool, method_id: str):
        """Run task vector similarity analysis in both original and subspace."""
        try:
            from fusion_bench.method.analysis.task_vector_cos_similarity import TaskVectorCosSimilarity
            
            # Create analyzer with matching parameters
            analyzer = TaskVectorCosSimilarity(
                plot_heatmap=True,
                trainable_only=True,
                method_name=method_id,
                proj_ratio=self.proj_ratio,
                use_G=self.use_G,
                analyze_subspace=True,
                device=str(self.device),
                output_path=self.analysis_output_path
            )
            
            print(f"[Analysis] Running task vector similarity analysis for {method_id}")
            analyzer.run(modelpool)
            
        except ImportError as e:
            print(f"[Analysis] Could not import TaskVectorCosSimilarity: {e}")
    
    def _run_task_vector_layer_analysis(self, modelpool: BaseModelPool, method_id: str):
        """Run layer-wise task vector analysis."""
        try:
            from fusion_bench.method.analysis.task_vector_layer_analysis import TaskVectorLayerAnalysis
            
            # Create analyzer
            analyzer = TaskVectorLayerAnalysis(
                trainable_only=True,
                method_name=method_id,
                device=str(self.device),
                output_path=self.analysis_output_path
            )
            
            print(f"[Analysis] Running layer-wise task vector analysis for {method_id}")
            analyzer.run(modelpool)
            
        except ImportError as e:
            print(f"[Analysis] Could not import TaskVectorLayerAnalysis: {e}")
