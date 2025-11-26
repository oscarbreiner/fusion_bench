# fastfood_merging.py
from __future__ import annotations
from typing import Any, Dict, List
import math

import torch
from torch import nn, Tensor

from fusion_bench.utils import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType

# Import utilities from srp_utils
from ..srp_utils import (
    EPS,
    create_projection_ops,
    zero_aware_aggregate,
    layer_key,
    normalize_weights,
    compute_global_dim
)


# ------------------ Algorithm ------------------
@auto_register_config
class FastfoodSubspaceMergeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Task-vector merging via Fastfood/SRHT subspaces with layer-wise projection sizes.

    Controls:
      layer_proj_mode: "custom" | "inverse" | "forward" | "feature_split"
        - custom: provide explicit layer -> proj_ratio mapping via layer_projections
        - inverse: weight by inverse layer number (earlier layers get smaller projections)
        - forward: weight by layer number (later layers get smaller projections)
        - feature_split: split into feature extraction and rest
      
      layer_projections: dict[str, float] (for custom mode, maps layer names to proj_ratios)
      proj_ratio: float (0..1) (base/default projection ratio)
      
      # For feature_split mode:
      feature_extractor_layers: int (number of first layers belonging to feature extractor)
      feature_proj_ratio: float (projection ratio for feature extractor layers)
      rest_proj_ratio: float (projection ratio for remaining layers)
      
      transform_type: str | None ("fwht" | "srht" | "dct" | "dht" | "none" | None)
                      - None/"none": Skip projection, merge in original space
      merge_where:    "subspace" | "postlift"
      merge_func:     "sum" | "mean" | "max" | "signmax" | "ema" | "ties_sum" | "ties_mean" | "ties_max"
      block_rows:     int
      weights:        list[float] (donor weights; normalized internally)
      scale:          float (post-merge scale on Δ*)
      
      # EMA-specific parameters (when merge_func="ema"):
      ema_task_order: "given" | "random" | "cosine_similarity" | "custom"
      ema_gamma:      float (sigmoid scaling factor, default 1.2)
      ema_w_c:        float (cosine alignment weight, default 0.6)
      ema_w_s:        float (scale ratio weight, default 0.4)
      ema_custom_order: list[str] (task names in desired order, when ema_task_order="custom")
    """

    def __init__(
        self,
        proj_ratio: float = 0.10,
        use_G: bool = False,  # Deprecated, kept for config compatibility
        device: str = "cuda",
        transform_type: str | None = "srht",    # "fwht" | "srht" | "dct" | "dht" | "none" | None
        layer_proj_mode: str = "inverse",  # "custom" | "inverse" | "forward" | "feature_split"
        layer_projections: Dict[str, float] | None = None,  # for custom mode
        # For feature_split mode:
        feature_extractor_layers: int = 0,
        feature_proj_ratio: float = 0.05,
        rest_proj_ratio: float = 0.15,
        merge_where: str = "subspace",   # "subspace" | "postlift"
        merge_func: str = "sum",         # "sum" | "mean" | "max" | "signmax" | "ema"
        block_rows: int = 8192,
        weights: List[float] | None = None,
        scale: float = 1.0,
        # EMA-specific parameters
        ema_task_order: str = "given",   # "given" | "random" | "cosine_similarity" | "custom"
        ema_gamma: float = 1.2,          # sigmoid scaling factor
        ema_w_c: float = 0.6,            # cosine alignment weight
        ema_w_s: float = 0.4,            # scale ratio weight
        ema_custom_order: List[str] | None = None,  # task names in desired order (when ema_task_order="custom")
        # Analysis integration parameters
        run_analysis: bool = False,      # Whether to run integrated analysis after merging
        analysis_methods: List[str] = None,  # List of analysis methods to run
        analysis_output_path: str = None,    # Path for analysis outputs
        # Kept in signature for compatibility (not used since align logic removed)
        ties_trim_pct: float = 0.0,
        tadrop_tau: float = 0.0,
        use_pareto: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.transform_type = str(transform_type)
        self.device = torch.device(device)
        
        # Layer-wise projection configuration
        self.layer_proj_mode = str(layer_proj_mode)
        assert self.layer_proj_mode in {"custom", "inverse", "forward", "feature_split"}
        self.layer_projections = layer_projections or {}
        self.feature_extractor_layers = int(feature_extractor_layers)
        self.feature_proj_ratio = float(feature_proj_ratio)
        self.rest_proj_ratio = float(rest_proj_ratio)
        
        assert merge_where in {"subspace", "postlift"}
        self.merge_where = merge_where
        self.merge_func = str(merge_func).lower()
        self.block_rows = int(block_rows)
        self.weights = list(weights) if weights is not None else None
        self.scale = float(scale)
        
        # EMA parameters
        self.ema_task_order = str(ema_task_order)
        self.ema_gamma = float(ema_gamma)
        self.ema_w_c = float(ema_w_c)
        self.ema_w_s = float(ema_w_s)
        self.ema_custom_order = list(ema_custom_order) if ema_custom_order is not None else None
        
        # Analysis parameters
        self.run_analysis = run_analysis
        self.analysis_methods = analysis_methods or []
        self.analysis_output_path = analysis_output_path

    def _extract_layer_number(self, param_name: str) -> int:
        """
        Extract layer number from parameter name.
        
        Tries to find patterns like:
        - model.layer.0.attention... -> 0
        - encoder.layers.12.mlp... -> 12
        - blocks.5.norm... -> 5
        
        Returns -1 if no layer number found.
        """
        import re
        # Common patterns for layer numbers
        patterns = [
            r'\.layer\.(\d+)\.',
            r'\.layers\.(\d+)\.',
            r'\.blocks\.(\d+)\.',
            r'\.encoder\.(\d+)\.',
            r'\.decoder\.(\d+)\.',
            r'\.h\.(\d+)\.',  # GPT-style
        ]
        
        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        
        return -1
    
    def _get_proj_ratio_for_layer(self, param_name: str, layer_name: str, layer_num: int, total_layers: int) -> float:
        """
        Get projection ratio for a specific layer based on the configured mode.
        
        Args:
            param_name: Full parameter name
            layer_name: Layer key (from layer_key function)
            layer_num: Extracted layer number (-1 if not found)
            total_layers: Total number of layers in the model
            
        Returns:
            Projection ratio for this layer
        """
        if self.layer_proj_mode == "custom":
            # Use explicit mapping
            if layer_name in self.layer_projections:
                return float(self.layer_projections[layer_name])
            # Fallback to base ratio
            return self.proj_ratio
        
        elif self.layer_proj_mode == "inverse":
            # Earlier layers get smaller projections: proj_ratio * (1 / (layer_num + 1))
            if layer_num >= 0 and total_layers > 0:
                # Normalize to [0, 1] range and invert
                normalized_pos = (layer_num + 1) / (total_layers + 1)
                return self.proj_ratio * (1.0 / normalized_pos)
            return self.proj_ratio
        
        elif self.layer_proj_mode == "forward":
            # Later layers get smaller projections: proj_ratio * layer_num / total_layers
            if layer_num >= 0 and total_layers > 0:
                normalized_pos = (layer_num + 1) / (total_layers + 1)
                return self.proj_ratio * normalized_pos
            return self.proj_ratio
        
        elif self.layer_proj_mode == "feature_split":
            # Split into feature extraction and rest
            if layer_num >= 0:
                if layer_num < self.feature_extractor_layers:
                    return self.feature_proj_ratio
                else:
                    return self.rest_proj_ratio
            # If layer number not found, use feature extractor ratio as default
            return self.feature_proj_ratio
        
        # Default fallback
        return self.proj_ratio

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

        # ---------- Layer analysis for projection sizing ----------
        # Extract all layer names and numbers
        layer_info = {}  # layer_name -> (layer_num, param_names)
        for k in keys_float:
            lname = _layer_key(k)
            lnum = self._extract_layer_number(k)
            if lname not in layer_info:
                layer_info[lname] = (lnum, [])
            layer_info[lname][1].append(k)
        
        # Determine total number of layers (max layer number found)
        layer_numbers = [lnum for lnum, _ in layer_info.values() if lnum >= 0]
        total_layers = max(layer_numbers) + 1 if layer_numbers else 0
        
        print(f"[Layer Analysis] Found {len(layer_info)} layer groups | Total layers: {total_layers}")
        print(f"[Projection Mode] {self.layer_proj_mode}")
        
        # Compute projection ratios for each layer
        layer_proj_ratios = {}
        for lname, (lnum, params) in layer_info.items():
            proj_r = self._get_proj_ratio_for_layer(params[0], lname, lnum, total_layers)
            layer_proj_ratios[lname] = proj_r
        
        # Print summary of projection ratios
        sorted_layers = sorted(layer_info.items(), key=lambda x: x[1][0] if x[1][0] >= 0 else 999)
        print("\n[Projection Ratios by Layer]")
        for lname, (lnum, params) in sorted_layers[:10]:  # Show first 10
            proj_r = layer_proj_ratios[lname]
            print(f"  Layer {lnum:3d} ({lname}): proj_ratio={proj_r:.4f} | {len(params)} params")
        if len(sorted_layers) > 10:
            print(f"  ... and {len(sorted_layers) - 10} more layers")

        # ---------- Seed scoping (always per-layer now) ----------
        def proj_seed_key(param_name: str) -> str:
            return _layer_key(param_name)
        
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
            for name in keys_float:
                tb = base_cpu[name]
                d_last = int(tb.shape[-1])
                rows = tb.numel() // d_last
                if rows <= 0:
                    continue

                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                # Get layer-specific projection ratio
                seed_key = proj_seed_key(name)
                layer_num = self._extract_layer_number(name)
                layer_proj_ratio = self._get_proj_ratio_for_layer(name, seed_key, layer_num, total_layers)
                
                # Use actual dimension (no global scope)
                cur_D = d_last
                proj_dim = max(1, int(cur_D * layer_proj_ratio))
                
                cache_key = (seed_key, cur_D, proj_dim)
                if cache_key not in op_cache:
                    fwd, lift = create_projection_ops(
                        cur_D, proj_dim, 
                        transform_type=self.transform_type,
                        seed_key=seed_key, 
                        device=dev
                    )
                    op_cache[cache_key] = (fwd, lift)
                else:
                    fwd, lift = op_cache[cache_key]

                cursor = 0
                tensor_changed = False

                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor:cursor + take, :]

                    # donor deltas (no global alignment needed - per-layer scope)
                    Xs: List[Tensor] = []
                    for dsd in donors_cpu:
                        sl_donor = dsd[name].view(rows, d_last).float()[cursor:cursor + take, :]
                        delta = sl_donor - sl_base
                        Xs.append(delta)

                    # project all donors
                    Ys = [fwd(X.to(dev, non_blocking=True)) for X in Xs]

                    # mix in chosen space
                    if self.merge_where == "postlift":
                        # reconstruct donors first, then aggregate in original space
                        Xhats = [lift(Y).to("cpu", non_blocking=True)[:, :d_last] for Y in Ys]
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
                        ).to("cpu", non_blocking=True)
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
                        Xmerge_full = lift(Ymerge).to("cpu", non_blocking=True)  # [take, cur_D]
                        Xmerge = Xmerge_full[:, :d_last]

                        # accumulate lift reconstruction error stats on a tiny slice
                        if take > 0 and (lift_err_den < 1e8):  # guard cost
                            # pick the first donor to estimate lift error
                            X0 = Xs[0].to(dev, non_blocking=True)
                            Y0 = Ys[0]
                            X0_rec = lift(Y0).to("cpu", non_blocking=True)[:, :d_last]
                            diff = (X0_rec.to(torch.float32) - Xs[0][:, :d_last].to(torch.float32))
                            lift_err_num += float(diff.pow(2).sum().item())
                            lift_err_den += float(Xs[0][:, :d_last].pow(2).sum().item())

                    # write back (scale)
                    upd = (self.scale * Xmerge).to(sl_base.dtype)
                    sl_base.add_(upd)

                    # did anything change?
                    tensor_changed = tensor_changed or bool(upd.abs().max().item() > 0)

                    cursor += take
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                merged_tensors += 1
                if tensor_changed:
                    changed_params += 1

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
        
        # Add layer projection mode
        parts.append(f"layerwise_{self.layer_proj_mode}")
        
        if self.proj_ratio != 0.95:
            parts.append(f"proj{self.proj_ratio}")
        if self.merge_func != 'signmax':
            parts.append(f"{self.merge_func}")
        if self.merge_where != 'subspace':
            parts.append(f"{self.merge_where}")
        
        # Add mode-specific parameters
        if self.layer_proj_mode == "feature_split":
            parts.append(f"fe{self.feature_extractor_layers}")
            parts.append(f"feproj{self.feature_proj_ratio}")
            parts.append(f"restproj{self.rest_proj_ratio}")
        
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
            return "fastfood_layerwise_default"
    
    def _run_merged_task_vector_analysis(self, modelpool: BaseModelPool, method_id: str):
        """Run merged task vector analysis with reused Fastfood projections."""
        try:
            from fusion_bench.method.analysis.merged_task_vector_analysis import MergedTaskVectorAnalysis
            
            # Create analyzer with matching parameters (using layer scope)
            analyzer = MergedTaskVectorAnalysis(
                merging_methods=["fastfood_merging"],
                proj_ratio=self.proj_ratio,
                transform_type=self.transform_type,
                merge_func=self.merge_func,
                subspace_scope="layer",  # Always layer scope for layerwise implementation
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
                transform_type=self.transform_type,
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
