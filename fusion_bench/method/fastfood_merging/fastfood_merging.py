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
    create_projection_ops,
    create_multi_sketch_ops,
    zero_aware_aggregate,
    layer_key,
    normalize_weights,
    compute_global_dim,
    subspace_boosting,
    generate_tall_masks,
    apply_consensus_mask,
)

# Import projection size estimator for adaptive sizing
from .projection_size_estimator import (
    ProjSizeCfg,
    proj_size_for,
    Mode,
    Strategy,
)

# Keep a local alias for clarity
_fastfood_ops = create_projection_ops
_multi_sketch_ops = create_multi_sketch_ops
_zero_aware_aggregate = zero_aware_aggregate
_layer_key = layer_key


# ------------------ Algorithm ------------------
@auto_register_config
class FastfoodSubspaceMergeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Task-vector merging via structured subspace projections (SRHT/FWHT/DCT/DHT).

    Controls:
      subspace_scope: "per_tensor" | "per_flat_tensor" | "layer" | "global"
                      - "per_tensor": row-wise projection on 2D tensors (rows x in_dim)
                      - "per_flat_tensor": flatten 2D tensors to (out·in) and apply one projection
                      - "layer": layer-wise projection
                      - "global": global projection across all parameters
      merge_where:    "subspace" | "postlift"
      merge_func:     "sum" | "mean" | "max" | "signmax" | "ema" | "ties_sum" | "ties_mean" | "ties_max"
      proj_ratio:     float (0..1) - used for fixed sizing or as ratio parameter in adaptive config
      use_G:          bool (kept for analyzer compatibility; not used by projections)
      block_rows:     int
      weights:        list[float] (donor weights; normalized internally)
      scale:          float (post-merge scale on Δ*)
      
      # Task Arithmetic Reconstruction:
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
      only_project_linear: bool
      
      # Adaptive projection sizing:
      use_adaptive_proj_size: bool
      adaptive_proj_mode: "tensor" | "layer"
      adaptive_proj_strategy: "fixed" | "random" | "rank" | "layer_progressive" | "layer_group"
      adaptive_proj_m_min: int
      adaptive_proj_f_max: float
      adaptive_proj_pow2: bool
      adaptive_proj_pow2_mode: "ceil" | "floor" | "nearest"
      adaptive_proj_beta: float
      adaptive_proj_seed: int | None
      adaptive_proj_start_ratio: float
      adaptive_proj_end_ratio: float
      adaptive_proj_growth_mode: "linear" | "exponential"
      adaptive_proj_group_boundary: int
      adaptive_proj_feature_ratio: float
      adaptive_proj_head_ratio: float
      
      # Subspace Boosting parameters:
      use_subspace_boosting: bool
      subspace_boosting_beta: float
    """

    def __init__(
        self,
        proj_ratio: float = 0.10,
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",   # "per_tensor" | "per_flat_tensor" | "layer" | "global"
        merge_where: str = "subspace",    # "subspace" | "postlift"
        merge_func: str = "sum",          # "sum" | "mean" | "max" | "signmax" | "ema" | "ties_*"
        block_rows: int = 8192,
        weights: List[float] | None = None,
        scale: float = 1.0,
        # ============================================================================
        # Transform Type (normalized to one of: "srht", "fwht", "dct", "dht")
        transform_type: str = "hadamard",  # maps to "srht"
        # ============================================================================
        # Task Arithmetic Reconstruction mode parameters
        use_task_arithmetic_reconstruction: bool = False,
        task_arithmetic_scaling: float = 1.0,
        report_reconstruction_error: bool = False,
        # ============================================================================
        # EMA-specific parameters
        ema_task_order: str = "given",
        ema_gamma: float = 1.2,
        ema_w_c: float = 0.6,
        ema_w_s: float = 0.4,
        ema_custom_order: List[str] | None = None,
        # Weight matching parameters (optional preprocessing)
        use_weight_matching: bool = False,
        weight_matching_max_iter: int = 100,
        weight_matching_seed: int = 0,
        weight_matching_verbose: bool = True,
        weight_matching_input_shapes: Tuple[Tuple[int, ...], ...] | None = None,
        # Analysis integration parameters
        run_analysis: bool = False,
        analysis_methods: List[str] = None,
        analysis_output_path: str = None,
        # TSV-style linear/non-linear separation
        only_project_linear: bool = False,
        # Embedding layer projection control
        project_embeddings: bool = True,
        # Adaptive projection size estimation
        use_adaptive_proj_size: bool = False,
        adaptive_proj_mode: str = "tensor",
        adaptive_proj_strategy: str = "rank",
        adaptive_proj_m_min: int = 16,
        adaptive_proj_f_max: float = 1.0,
        adaptive_proj_pow2: bool = False,
        adaptive_proj_pow2_mode: str = "ceil",
        adaptive_proj_beta: float = 2.5,
        adaptive_proj_seed: int | None = None,
        adaptive_proj_start_ratio: float = 0.1,
        adaptive_proj_end_ratio: float = 1.0,
        adaptive_proj_growth_mode: str = "linear",
        adaptive_proj_group_boundary: int = 5,
        adaptive_proj_feature_ratio: float = 0.3,
        adaptive_proj_head_ratio: float = 0.8,
        adaptive_proj_global_ratio: float = 0.25,
        adaptive_proj_power_law_alpha: float = 0.85,
        # LiNeS (Layer Scaling) Parameters
        use_lines: bool = False,
        lines_num_blocks: int | None = None,
        lines_alpha: float | None = None,
        lines_beta: float = 1.0,
        lines_auto_alpha: bool = True,
        # Multi-Sketch Parameters
        num_sketches: int = 1,
        sketch_ensemble_mode: str = "mean",
        # Subspace Boosting Parameters
        use_subspace_boosting: bool = False,
        subspace_boosting_beta: float = 0.01,
        # TALL Masks / Consensus Masking Parameters
        use_consensus_mask: bool = False,
        tall_mask_lambda: float = 0.6,
        consensus_threshold: int = 2,
        # Kept in signature for compatibility (not used)
        ties_trim_pct: float = 0.0,
        tadrop_tau: float = 0.0,
        use_pareto: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)  # kept for analyzer interface; projection ops ignore it
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        assert merge_where in {"subspace", "postlift"}
        self.merge_where = merge_where
        self.merge_func = str(merge_func).lower()
        self.block_rows = int(block_rows)
        self.weights = list(weights) if weights is not None else None
        self.scale = float(scale)
        
        # LiNeS parameters
        self.use_lines = bool(use_lines)
        self.lines_num_blocks = int(lines_num_blocks) if lines_num_blocks is not None else None
        self.lines_alpha = float(lines_alpha) if lines_alpha is not None else None
        self.lines_beta = float(lines_beta)
        self.lines_auto_alpha = bool(lines_auto_alpha)

        # Normalize transform type to the set expected by create_projection_ops
        self.transform_type = self._normalize_transform_type(transform_type)

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

        # Embedding layer projection control
        self.project_embeddings = bool(project_embeddings)

        # Multi-Sketch parameters
        self.num_sketches = int(num_sketches)
        self.sketch_ensemble_mode = str(sketch_ensemble_mode).lower()
        assert self.num_sketches >= 1, f"num_sketches must be >= 1, got {self.num_sketches}"
        assert self.sketch_ensemble_mode in {"mean", "sum", "max", "median"}, \
            f"sketch_ensemble_mode must be one of: mean, sum, max, median (got {self.sketch_ensemble_mode})"

        # Subspace Boosting parameters
        self.use_subspace_boosting = bool(use_subspace_boosting)
        self.subspace_boosting_beta = float(subspace_boosting_beta)
        assert 0.0 <= self.subspace_boosting_beta <= 1.0, \
            f"subspace_boosting_beta must be in [0, 1], got {self.subspace_boosting_beta}"

        # TALL Masks / Consensus Masking parameters
        self.use_consensus_mask = bool(use_consensus_mask)
        self.tall_mask_lambda = float(tall_mask_lambda)
        self.consensus_threshold = int(consensus_threshold)
        assert 0.0 <= self.tall_mask_lambda <= 1.0, \
            f"tall_mask_lambda must be in [0, 1], got {self.tall_mask_lambda}"
        assert self.consensus_threshold >= 0, \
            f"consensus_threshold must be >= 0, got {self.consensus_threshold}"

        # Adaptive projection size estimation
        self.use_adaptive_proj_size = bool(use_adaptive_proj_size)
        self.adaptive_proj_mode = str(adaptive_proj_mode)
        self.adaptive_proj_strategy = str(adaptive_proj_strategy)

        # Initialize projection size config if adaptive sizing is enabled
        if self.use_adaptive_proj_size:
            import random as _random
            self.proj_size_cfg = ProjSizeCfg(
                m_min=int(adaptive_proj_m_min),
                f_max=float(adaptive_proj_f_max),
                pow2_round=bool(adaptive_proj_pow2),
                pow2_mode=str(adaptive_proj_pow2_mode),
                ratio=self.proj_ratio,  # Use main proj_ratio for fixed strategy
                beta=float(adaptive_proj_beta),
                start_proj_ratio=float(adaptive_proj_start_ratio),
                end_proj_ratio=float(adaptive_proj_end_ratio),
                growth_mode=str(adaptive_proj_growth_mode),
                group_boundary_layer=int(adaptive_proj_group_boundary),
                feature_proj_ratio=float(adaptive_proj_feature_ratio),
                head_proj_ratio=float(adaptive_proj_head_ratio),
                global_ratio=float(adaptive_proj_global_ratio),
                power_law_alpha=float(adaptive_proj_power_law_alpha),
                rng=_random.Random(adaptive_proj_seed) if adaptive_proj_seed is not None else None,
            )
        else:
            self.proj_size_cfg = None

        # Analysis parameters
        self.run_analysis = run_analysis
        self.analysis_methods = analysis_methods or []
        self.analysis_output_path = analysis_output_path

    # ---------- small helper ----------
    @staticmethod
    def _normalize_transform_type(s: str) -> str:
        """
        Map user-friendly names to the canonical set {'srht','fwht','dct','dht','none'}.
        - 'hadamard' -> 'srht' (subsampled Hadamard sketch)
        - 'hadamard_full' -> 'fwht' (full FWHT, m must equal 2^k)
        - 'none' -> 'none' (identity transform, no projection)
        Also accept already-canonical names.
        """
        ss = str(s).lower()
        if ss in {"srht", "fwht", "dct", "dht", "fastfood", "none", "identity"}:
            # Map 'identity' to 'none' for consistency
            if ss == "identity":
                return "none"
            return ss
        if ss in {"hadamard", "wht", "had"}:
            return "srht"
        if ss in {"hadamard_full", "fwht_full"}:
            return "fwht"
        raise ValueError(f"Unknown transform_type='{s}'. Use one of: 'srht','fwht','dct','dht','fastfood','none' (or 'hadamard','hadamard_full','identity').")

    # ------------------- helpers -------------------
    @staticmethod
    def _is_linear_layer(name: str) -> bool:
        """
        Check if a parameter belongs to a linear layer (MLP or attention projection).
        Used for selective application of Subspace Boosting.
        """
        linear_patterns = [
            "attn", "attention",  # Attention projections
            "q_proj", "k_proj", "v_proj", "out_proj",  # Explicit attention names
            "query", "key", "value",
            "mlp", "fc", "dense",  # MLP layers
            "linear",  # Generic linear
        ]
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in linear_patterns) and ".weight" in name

    @staticmethod
    def _is_embedding_layer(name: str) -> bool:
        """
        Check if a parameter name corresponds to an embedding layer.
        
        Embedding layers typically include:
        - position_embedding, pos_embed, positional_embedding
        - patch_embed, patch_embedding
        - token_embedding, word_embedding
        - cls_token, class_token
        
        Args:
            name: Parameter name
            
        Returns:
            True if parameter is an embedding layer
        """
        name_lower = name.lower()
        embedding_patterns = [
            'position_embedding',
            'pos_embed',
            'positional_embedding',
            'patch_embed',
            'patch_embedding',
            'token_embedding',
            'word_embedding',
            'cls_token',
            'class_token',
            'mask_token',
            'decoder_embed',
            'text_projection',
            'embeddings.weight',  # Generic embedding weight
        ]
        return any(pattern in name_lower for pattern in embedding_patterns)

    def _compute_proj_size(
        self,
        param_name: str,
        tensor: torch.Tensor,
        layer_params: Dict[str, torch.Tensor] | None = None,
        layer_idx: int = 0,
        num_layers: int = 1,
        all_dims: List[int] | None = None,  # For layer_power_law strategy (DEPRECATED - now uses per-layer dims)
    ) -> int:
        """
        Compute projection size for a parameter tensor.
        
        Args:
            param_name: Name of the parameter
            tensor: Parameter tensor
            layer_params: Dictionary of layer parameters (for adaptive_proj_mode="layer")
            layer_idx: Current layer index (for layer_progressive and layer_group strategies)
            num_layers: Total number of layers (for layer_progressive and layer_group strategies)
            all_dims: List of all dimensions (DEPRECATED - for backward compatibility only)
        """
        if not self.use_adaptive_proj_size:
            d_last = int(tensor.shape[-1])
            if self.subspace_scope == "global" and hasattr(self, "_global_D"):
                cur_D = self._global_D
            elif self.subspace_scope == "per_flat_tensor" and tensor.ndim == 2:
                cur_D = tensor.numel()
            else:
                cur_D = d_last
            return max(1, int(cur_D * self.proj_ratio))

        # Adaptive sizing mode
        try:
            mode: Mode = self.adaptive_proj_mode  # type: ignore
            strategy: Strategy = self.adaptive_proj_strategy  # type: ignore

            # For layer_power_law strategy, get per-layer dimensions
            if strategy == "layer_power_law" and hasattr(self, '_layer_dims_map'):
                lkey = layer_key(param_name)
                layer_all_dims = self._layer_dims_map.get(lkey, None)
                if layer_all_dims is None:
                    # Fallback if parameter not found in map
                    print(f"[Warning] Parameter {param_name} not found in layer dims map, using fallback")
                    layer_all_dims = [int(tensor.shape[-1])]
            else:
                layer_all_dims = all_dims  # For other strategies or backward compatibility

            if mode == "layer":
                if layer_params is None:
                    raise ValueError("layer_params required for adaptive layer mode")
                m = proj_size_for(
                    layer_params, mode="layer", strategy=strategy, cfg=self.proj_size_cfg,
                    layer_idx=layer_idx, num_layers=num_layers, all_dims=layer_all_dims
                )
            else:  # tensor mode
                m = proj_size_for(
                    tensor, mode="tensor", strategy=strategy, cfg=self.proj_size_cfg,
                    layer_idx=layer_idx, num_layers=num_layers, all_dims=layer_all_dims
                )

            return m
        except Exception as e:
            print(f"[Adaptive Proj Size] Error for {param_name}: {e}, falling back to fixed ratio")
            d_last = int(tensor.shape[-1])
            return max(1, int(d_last * self.proj_ratio))

    def _build_layer_index_map(self, keys: List[str]) -> Tuple[Dict[str, int], int]:
        """
        Build a mapping from parameter name to layer index.
        
        Used by layer_progressive and layer_group strategies to determine which 
        layer each parameter belongs to.
        
        Returns:
            (param_to_layer_idx, num_layers) - mapping and total layer count
        """
        # Group parameters by layer key
        layer_keys_ordered = []
        seen = set()
        for k in keys:
            lk = _layer_key(k)
            if lk not in seen:
                layer_keys_ordered.append(lk)
                seen.add(lk)
        
        num_layers = len(layer_keys_ordered)
        layer_to_idx = {lk: i for i, lk in enumerate(layer_keys_ordered)}
        
        # Map each parameter to its layer index
        param_to_idx = {k: layer_to_idx[_layer_key(k)] for k in keys}
        
        return param_to_idx, num_layers

    def _generate_layer_projection_report(
        self,
        param_to_layer_idx: Dict[str, int],
        num_layers: int,
        keys_linear: List[str],
        base_cpu: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Generate a detailed report of projection sizes per layer for layer_progressive
        and layer_group strategies.
        
        Args:
            param_to_layer_idx: Mapping from parameter name to layer index
            num_layers: Total number of layers
            keys_linear: List of linear parameter names
            base_cpu: Base model state dict for getting tensor shapes
            
        Returns:
            Dictionary containing layer-wise projection information
        """
        import json
        from pathlib import Path
        from fusion_bench.method.fastfood_merging.projection_size_estimator import (
            compute_layer_progressive_ratio,
            compute_layer_group_ratio,
        )
        
        report = {
            "strategy": self.adaptive_proj_strategy,
            "adaptive_proj_mode": self.adaptive_proj_mode,
            "num_layers": num_layers,
            "layers": []
        }
        
        # Add strategy-specific configuration
        if self.adaptive_proj_strategy == "layer_progressive":
            report["config"] = {
                "start_proj_ratio": self.proj_size_cfg.start_proj_ratio,
                "end_proj_ratio": self.proj_size_cfg.end_proj_ratio,
                "growth_mode": self.proj_size_cfg.growth_mode,
            }
        elif self.adaptive_proj_strategy == "layer_group":
            report["config"] = {
                "group_boundary_layer": self.proj_size_cfg.group_boundary_layer,
                "feature_proj_ratio": self.proj_size_cfg.feature_proj_ratio,
                "head_proj_ratio": self.proj_size_cfg.head_proj_ratio,
            }
        
        # Collect layer information
        layer_info = {}
        for layer_idx in range(num_layers):
            layer_info[layer_idx] = {
                "layer_index": layer_idx,
                "parameters": [],
                "projection_ratios": [],
                "projection_sizes": [],
            }
        
        # Process each parameter
        for name in keys_linear:
            if name not in param_to_layer_idx:
                continue
                
            layer_idx = param_to_layer_idx[name]
            tb = base_cpu[name]
            d_last = int(tb.shape[-1])
            
            # Compute projection size
            all_dims = getattr(self, '_all_dims', None)
            if self.adaptive_proj_mode == "layer":
                lkey = layer_key(name)
                layer_params = self._layer_groups.get(lkey, {name: tb})
                proj_size = self._compute_proj_size(name, tb, layer_params, layer_idx, num_layers, all_dims)
            else:
                proj_size = self._compute_proj_size(name, tb, None, layer_idx, num_layers, all_dims)
            
            # Compute the actual ratio used
            proj_ratio = proj_size / d_last if d_last > 0 else 0.0
            
            layer_info[layer_idx]["parameters"].append({
                "name": name,
                "shape": list(tb.shape),
                "d_last": d_last,
                "proj_size": proj_size,
                "proj_ratio": round(proj_ratio, 4),
            })
            layer_info[layer_idx]["projection_ratios"].append(proj_ratio)
            layer_info[layer_idx]["projection_sizes"].append(proj_size)
        
        # Aggregate layer statistics
        for layer_idx in range(num_layers):
            info = layer_info[layer_idx]
            
            # Compute theoretical ratio for this layer
            if self.adaptive_proj_strategy == "layer_progressive":
                theoretical_ratio = compute_layer_progressive_ratio(
                    layer_idx,
                    num_layers,
                    self.proj_size_cfg.start_proj_ratio,
                    self.proj_size_cfg.end_proj_ratio,
                    self.proj_size_cfg.growth_mode
                )
            elif self.adaptive_proj_strategy == "layer_group":
                theoretical_ratio = compute_layer_group_ratio(
                    layer_idx,
                    self.proj_size_cfg.group_boundary_layer,
                    self.proj_size_cfg.feature_proj_ratio,
                    self.proj_size_cfg.head_proj_ratio
                )
            else:
                theoretical_ratio = None
            
            if theoretical_ratio is not None:
                info["theoretical_proj_ratio"] = round(theoretical_ratio, 4)
            
            # Average actual projection ratio
            if info["projection_ratios"]:
                info["avg_proj_ratio"] = round(
                    sum(info["projection_ratios"]) / len(info["projection_ratios"]), 4
                )
                info["min_proj_size"] = min(info["projection_sizes"])
                info["max_proj_size"] = max(info["projection_sizes"])
            
            report["layers"].append(info)
        
        return report

    def _print_layer_power_law_report(self):
        """
        Print a detailed report of layer power-law projection sizes showing:
        - Per-layer budget and actual projections
        - Per-parameter dimensions, projection sizes, and effective ratios
        - Verification that per-layer budget is preserved
        """
        if not hasattr(self, '_layer_power_law_projections') or not self._layer_power_law_projections:
            print("[Layer Power-Law] No projection data available")
            return
        
        print(f"\n{'='*80}")
        print(f"Layer Power-Law Projection Report")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Target per-layer ratio: {self.proj_size_cfg.global_ratio:.3f}")
        print(f"  Power-law alpha: {self.proj_size_cfg.power_law_alpha:.3f}")
        print(f"{'='*80}\n")
        
        # Sort layers for consistent output
        sorted_layers = sorted(self._layer_power_law_projections.keys())
        
        total_original_dims = 0
        total_projected_dims = 0
        
        for layer_idx, lkey in enumerate(sorted_layers, 1):
            params = self._layer_power_law_projections[lkey]
            
            # Calculate layer statistics
            layer_original_sum = sum(d_last for d_last, _, _ in params.values())
            layer_projected_sum = sum(proj_dim for _, proj_dim, _ in params.values())
            layer_budget = self.proj_size_cfg.global_ratio * layer_original_sum
            layer_actual_ratio = layer_projected_sum / layer_original_sum if layer_original_sum > 0 else 0.0
            
            total_original_dims += layer_original_sum
            total_projected_dims += layer_projected_sum
            
            # Print layer header
            print(f"Layer {layer_idx}: {lkey}")
            print(f"  Layer Budget: {self.proj_size_cfg.global_ratio:.3f} × {layer_original_sum} = {layer_budget:.0f}")
            print(f"  Actual Total: {layer_projected_sum} (ratio: {layer_actual_ratio:.4f})")
            print(f"  Budget Match: {'✓' if abs(layer_actual_ratio - self.proj_size_cfg.global_ratio) < 0.01 else '✗'}")
            print(f"\n  Parameters ({len(params)}):")
            
            # Sort parameters by name for consistent output
            sorted_params = sorted(params.items())
            
            # Find max name length for alignment
            max_name_len = max(len(name.split('.')[-1]) for name, _ in sorted_params) if sorted_params else 10
            
            for param_name, (d_last, proj_dim, effective_ratio) in sorted_params:
                param_short = param_name.split('.')[-1]  # Just the last part (e.g., "weight", "bias")
                ratio_indicator = ""
                if effective_ratio > self.proj_size_cfg.global_ratio + 0.02:
                    ratio_indicator = " ↑"  # Higher compression ratio (less compressed)
                elif effective_ratio < self.proj_size_cfg.global_ratio - 0.02:
                    ratio_indicator = " ↓"  # Lower compression ratio (more compressed)
                else:
                    ratio_indicator = " ≈"  # About equal
                
                print(f"    {param_short:<{max_name_len}} : d={d_last:>5} → m={proj_dim:>5}  "
                      f"(ratio={effective_ratio:.4f}{ratio_indicator})")
            
            print()  # Empty line between layers
        
        # Print overall summary
        print(f"{'='*80}")
        print(f"Overall Summary:")
        print(f"  Total layers: {len(sorted_layers)}")
        print(f"  Total original dimensions: {total_original_dims}")
        print(f"  Total projected dimensions: {total_projected_dims}")
        overall_ratio = total_projected_dims / total_original_dims if total_original_dims > 0 else 0.0
        print(f"  Overall average ratio: {overall_ratio:.4f}")
        print(f"  Target ratio: {self.proj_size_cfg.global_ratio:.3f}")
        print(f"  Ratio difference: {abs(overall_ratio - self.proj_size_cfg.global_ratio):.4f}")
        print(f"{'='*80}\n")
        
        # Legend
        print("Legend:")
        print("  ↑ : Parameter compressed LESS than layer average (higher ratio)")
        print("  ↓ : Parameter compressed MORE than layer average (lower ratio)")
        print("  ≈ : Parameter compressed about equal to layer average")
        print()

    def _generate_layer_power_law_report_data(self) -> Dict[str, Any]:
        """
        Generate layer power-law report data as a dictionary for JSON export.
        
        Returns:
            Dictionary containing the complete report data
        """
        if not hasattr(self, '_layer_power_law_projections') or not self._layer_power_law_projections:
            return {"error": "No projection data available"}
        
        report = {
            "configuration": {
                "target_per_layer_ratio": float(self.proj_size_cfg.global_ratio),
                "power_law_alpha": float(self.proj_size_cfg.power_law_alpha),
            },
            "layers": [],
            "overall_summary": {}
        }
        
        # Sort layers for consistent output
        sorted_layers = sorted(self._layer_power_law_projections.keys())
        
        total_original_dims = 0
        total_projected_dims = 0
        
        for lkey in sorted_layers:
            params = self._layer_power_law_projections[lkey]
            
            # Calculate layer statistics
            layer_original_sum = sum(d_last for d_last, _, _ in params.values())
            layer_projected_sum = sum(proj_dim for _, proj_dim, _ in params.values())
            layer_budget = self.proj_size_cfg.global_ratio * layer_original_sum
            layer_actual_ratio = layer_projected_sum / layer_original_sum if layer_original_sum > 0 else 0.0
            
            total_original_dims += layer_original_sum
            total_projected_dims += layer_projected_sum
            
            # Build layer data
            layer_data = {
                "layer_key": lkey,
                "budget": {
                    "target_ratio": float(self.proj_size_cfg.global_ratio),
                    "original_dims_sum": int(layer_original_sum),
                    "budget_projection_size": float(layer_budget),
                    "actual_projection_size": int(layer_projected_sum),
                    "actual_ratio": float(layer_actual_ratio),
                    "budget_match": abs(layer_actual_ratio - self.proj_size_cfg.global_ratio) < 0.01
                },
                "parameters": []
            }
            
            # Sort parameters by name for consistent output
            sorted_params = sorted(params.items())
            
            for param_name, (d_last, proj_dim, effective_ratio) in sorted_params:
                ratio_indicator = ""
                if effective_ratio > self.proj_size_cfg.global_ratio + 0.02:
                    ratio_indicator = "higher"  # Higher compression ratio (less compressed)
                elif effective_ratio < self.proj_size_cfg.global_ratio - 0.02:
                    ratio_indicator = "lower"  # Lower compression ratio (more compressed)
                else:
                    ratio_indicator = "equal"  # About equal
                
                param_data = {
                    "name": param_name,
                    "original_dim": int(d_last),
                    "projection_dim": int(proj_dim),
                    "effective_ratio": float(effective_ratio),
                    "compression_indicator": ratio_indicator
                }
                layer_data["parameters"].append(param_data)
            
            report["layers"].append(layer_data)
        
        # Add overall summary
        overall_ratio = total_projected_dims / total_original_dims if total_original_dims > 0 else 0.0
        report["overall_summary"] = {
            "total_layers": len(sorted_layers),
            "total_original_dimensions": int(total_original_dims),
            "total_projected_dimensions": int(total_projected_dims),
            "overall_average_ratio": float(overall_ratio),
            "target_ratio": float(self.proj_size_cfg.global_ratio),
            "ratio_difference": float(abs(overall_ratio - self.proj_size_cfg.global_ratio))
        }
        
        return report

    def _get_output_directory(self) -> Path:
        """
        Determine the output directory for saving reports.
        Tries fabric loggers, hydra config, then falls back to ./outputs
        
        Returns:
            Path to output directory
        """
        from pathlib import Path
        
        # Try to get output directory from fabric loggers
        if hasattr(self, 'fabric') and hasattr(self.fabric, 'loggers'):
            for logger in self.fabric.loggers:
                if hasattr(logger, 'log_dir'):
                    return Path(logger.log_dir)
        
        # Try hydra output directory
        try:
            from omegaconf import DictConfig
            import hydra
            if hasattr(hydra, 'core') and hasattr(hydra.core, 'hydra_config') and hydra.core.hydra_config.HydraConfig.initialized():
                hconf = hydra.core.hydra_config.HydraConfig.get()
                return Path(hconf.runtime.output_dir)
        except:
            pass
        
        # Fallback to current directory
        return Path.cwd() / "outputs"

    def _apply_lines_scaling(
        self,
        merged_delta: Dict[str, torch.Tensor],
        num_tasks: int,
        norm_summed_tvs: float | None = None,
        norm_merged_tv: float | None = None,
    ) -> tuple[Dict[str, torch.Tensor], float]:
        """
        Apply LiNeS (Layer Scaling) to the merged task vector.
        
        LiNeS progressively scales the task vector based on layer depth:
        - Early layers (closer to input): Lower scaling (preserve general features)
        - Later layers (closer to output): Higher scaling (preserve task-specific features)
        
        Scaling formula: scale(layer) = alpha + beta * (layer_idx / (num_blocks - 1))
        
        Args:
            merged_delta: Merged task vector (state dict of deltas)
            num_tasks: Number of tasks being merged
            norm_summed_tvs: L1 norm of summed task vectors (for auto alpha computation)
            norm_merged_tv: L1 norm of merged task vector (for auto alpha computation)
            
        Returns:
            Tuple of (scaled task vector (state dict), computed alpha value)
        """
        if not self.use_lines:
            return merged_delta, None
        
        import copy
        scaled_delta = copy.deepcopy(merged_delta)
        
        # Determine num_blocks if not specified
        if self.lines_num_blocks is None:
            # Auto-detect from model architecture (default: 12 for ViT-B, 24 for ViT-L)
            # Check if any key contains "layers" to infer model type
            has_24_layers = any(f".23." in k for k in merged_delta.keys())
            num_blocks = 24 if has_24_layers else 12
            print(f"[LiNeS] Auto-detected num_blocks={num_blocks}")
        else:
            num_blocks = self.lines_num_blocks
        
        # Determine alpha
        if self.lines_auto_alpha and norm_summed_tvs is not None and norm_merged_tv is not None:
            # Auto-compute alpha as in LiNeS paper (multi-task setting)
            alpha = (norm_summed_tvs / (norm_merged_tv + 1e-12)) * (1.0 / num_tasks)
            print(f"[LiNeS] Auto-computed alpha={alpha:.6f} (norm_ratio={norm_summed_tvs/norm_merged_tv:.4f}, num_tasks={num_tasks})")
        elif self.lines_alpha is not None:
            alpha = self.lines_alpha
            print(f"[LiNeS] Using manual alpha={alpha:.6f}")
        else:
            # Default: single-task mode uses beta as alpha in LiNeS paper
            alpha = self.lines_beta
            print(f"[LiNeS] Using default alpha={alpha:.6f} (same as beta)")
        
        beta = self.lines_beta
        
        # Build layer key patterns (e.g., ".0.", ".1.", ..., ".11." for 12 blocks)
        key_blocks = [f".{i}." for i in range(num_blocks)]
        
        # Compute scaling factors for each parameter
        layer_scalings_dict = {}
        for k in scaled_delta.keys():
            # Check if parameter belongs to a residual block
            found_layer = False
            for layer_idx, block_pattern in enumerate(key_blocks):
                if block_pattern in k:
                    # Linear scaling from alpha to alpha+beta
                    scaling = alpha + beta * (layer_idx / (num_blocks - 1))
                    layer_scalings_dict[k] = scaling
                    found_layer = True
                    break
            
            # If not in a residual block, use minimum scaling (alpha)
            if not found_layer:
                layer_scalings_dict[k] = alpha
        
        # Apply scaling
        for k in scaled_delta.keys():
            scaling = layer_scalings_dict.get(k, alpha)
            scaled_delta[k] = scaled_delta[k] * scaling
        
        # Print summary
        unique_scalings = sorted(set(layer_scalings_dict.values()))
        print(f"[LiNeS] Applied layer scaling: alpha={alpha:.4f}, beta={beta:.4f}")
        print(f"[LiNeS] Scaling range: [{min(unique_scalings):.4f}, {max(unique_scalings):.4f}]")
        print(f"[LiNeS] Number of unique scaling factors: {len(unique_scalings)}")
        
        return scaled_delta, alpha

    # ------------------- Multi-Sketch Helper -------------------
    def _create_projection_operators(
        self,
        cur_D: int,
        proj_dim: int,
        seed_key: str,
        dev: torch.device,
        op_cache: Dict,
    ) -> Tuple[List[Callable], List[Callable]]:
        """
        Create projection operators, potentially multiple sketches.
        
        Returns:
            (fwd_ops, lift_ops) - Lists of forward and lift operators
            For num_sketches=1, returns single-element lists for consistency
        """
        if self.num_sketches == 1:
            # Single sketch - use cached ops
            cache_key = (self.transform_type, seed_key, cur_D, proj_dim)
            
            if cache_key not in op_cache:
                fwd, lift = _fastfood_ops(
                    cur_D,
                    proj_dim,
                    seed_key=seed_key,
                    device=dev,
                    transform_type=self.transform_type,
                )
                op_cache[cache_key] = (fwd, lift)
            else:
                fwd, lift = op_cache[cache_key]
            
            return [fwd], [lift]
        else:
            # Multi-sketch - create independent operators
            cache_key = (self.transform_type, seed_key, cur_D, proj_dim, self.num_sketches)
            
            if cache_key not in op_cache:
                fwd_ops, lift_ops = _multi_sketch_ops(
                    cur_D,
                    proj_dim,
                    num_sketches=self.num_sketches,
                    seed_key=seed_key,
                    device=dev,
                    transform_type=self.transform_type,
                )
                op_cache[cache_key] = (fwd_ops, lift_ops)
            else:
                fwd_ops, lift_ops = op_cache[cache_key]
            
            return fwd_ops, lift_ops

    def _multi_sketch_project(
        self,
        X_batch: Tensor,
        fwd_ops: List[Callable],
    ) -> List[Tensor]:
        """
        Project data using multiple sketch operators.
        
        Args:
            X_batch: Input tensor [..., D]
            fwd_ops: List of forward projection operators
            
        Returns:
            List of projected tensors, one per sketch
        """
        Y_sketches = []
        for fwd_j in fwd_ops:
            Y_j = fwd_j(X_batch)
            Y_sketches.append(Y_j)
        return Y_sketches

    def _multi_sketch_lift_and_ensemble(
        self,
        Y_merged_sketches: List[Tensor],
        lift_ops: List[Callable],
    ) -> Tensor:
        """
        Lift merged sketches back to original space and ensemble.
        
        Args:
            Y_merged_sketches: List of merged projections (one per sketch)
            lift_ops: List of lift operators
            
        Returns:
            Ensembled reconstruction in original space
        """
        # Lift each sketch
        X_lifted = []
        for Y_j, lift_j in zip(Y_merged_sketches, lift_ops):
            X_j = lift_j(Y_j)
            X_lifted.append(X_j)
        
        # Ensemble across sketches
        X_stacked = torch.stack(X_lifted, dim=0)  # [J, ...]
        
        if self.sketch_ensemble_mode == "mean":
            return X_stacked.mean(dim=0)
        elif self.sketch_ensemble_mode == "sum":
            return X_stacked.sum(dim=0)
        elif self.sketch_ensemble_mode == "max":
            return X_stacked.abs().max(dim=0)[0] * torch.sign(X_stacked.sum(dim=0))
        elif self.sketch_ensemble_mode == "median":
            return X_stacked.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown sketch_ensemble_mode: {self.sketch_ensemble_mode}")

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
                modelpool.load_model(n).state_dict(keep_vars=True) for n in donor_names
            ]
            base_sd: Dict[str, Tensor] = base_model.state_dict(keep_vars=True)

        # ---------- Weight Matching Preprocessing (Optional) ----------
        if self.use_weight_matching:
            print("\n=== Weight Matching Preprocessing ===")
            print(f"[Weight Matching] Aligning {len(donor_names)} donors to base model")
            print(f"[Weight Matching] max_iter={self.weight_matching_max_iter}, seed={self.weight_matching_seed}")

            with self.profile("weight_matching"):
                try:
                    from .weight_matching.core import get_permutation_spec
                    from .weight_matching.weight_matching import weight_matching
                    from .weight_matching.core import apply_perm
                except ImportError as e:
                    raise ImportError(
                        f"Weight matching modules not found. Error: {e}\n"
                        "Make sure weight_matching package is properly installed."
                    )

                try:
                    if self.weight_matching_input_shapes is not None:
                        input_shapes = self.weight_matching_input_shapes
                    else:
                        input_shapes = ((1, 3, 224, 224),)
                        print(f"[Weight Matching] Using default input shapes: {input_shapes}")

                    spec = get_permutation_spec(
                        base_model,
                        input_shapes,
                        verbose=False,
                    )
                    print(f"[Weight Matching] Found {len(spec)} permutation groups")
                except Exception as e:
                    print(f"[Weight Matching] Warning: Could not generate permutation spec: {e}")
                    print("Skipping weight matching preprocessing")
                    spec = None

                if spec is not None:
                    for i, donor_name in enumerate(donor_names):
                        print(f"[Weight Matching] Aligning donor {i+1}/{len(donor_names)}: {donor_name}")
                        weight_matching(
                            spec=spec,
                            state_as=base_sd,
                            state_bs=donors_sd[i],
                            max_iter=self.weight_matching_max_iter,
                            init_perm=None,
                            inplace=True,
                            skip_suffixes=("running_mean", "running_var"),
                            skip_missing=True,
                            verbose=self.weight_matching_verbose,
                            seed=self.weight_matching_seed,
                            return_costs=False,
                        )
                    print("[Weight Matching] All donors aligned to base model")
                    print("=" * 50 + "\n")

        # ---------- Task Arithmetic Reconstruction Mode (Optional) ----------
        if self.use_task_arithmetic_reconstruction:
            print("\n=== Task Arithmetic Reconstruction Mode ===")
            print(f"[Task Arithmetic] Merging {len(donor_names)} task vectors with scaling factor {self.task_arithmetic_scaling}")
            print(f"[Task Arithmetic] Will project/lift the MERGED TASK VECTOR (not the whole model)")

            with self.profile("task_arithmetic"):
                base_dict = {k: (v.data if hasattr(v, "data") else v) for k, v in base_sd.items()}
                donors_dict = [{k: (v.data if hasattr(v, "data") else v) for k, v in d.items()} for d in donors_sd]

                task_vector = None
                for i, donor_name in enumerate(donor_names):
                    print(f"[Task Arithmetic] Processing task {i+1}/{len(donor_names)}: {donor_name}")
                    donor_tv = state_dict_sub(donors_dict[i], base_dict)
                    task_vector = donor_tv if task_vector is None else state_dict_add(task_vector, donor_tv)

                task_vector = state_dict_mul(task_vector, self.task_arithmetic_scaling)
                print(f"[Task Arithmetic] Merged task vector computed: θ_tv = λ·Σᵢ(θᵢ - θ₀)")

                if self.report_reconstruction_error or self.proj_ratio < 1.0:
                    print(f"\n[Projection Test] Projecting task vector with proj_ratio={self.proj_ratio}")

                    # 🚀 OPTIMIZATION: Remove redundant .clone()
                    tv_cpu = {k: v.detach().cpu() for k, v in task_vector.items()}
                    dev = self.device

                    total_reconstruction_error = 0.0
                    total_norm = 0.0
                    reconstruction_errors: List[Tuple[str, float]] = []

                    # Separate keys into embedding and non-embedding layers
                    keys_float_ta = [
                        k
                        for k in tv_cpu.keys()
                        if torch.is_floating_point(tv_cpu[k]) and tv_cpu[k].ndim >= 1
                    ]
                    
                    # Filter out embedding layers if project_embeddings=False
                    if not self.project_embeddings:
                        keys_embedding_ta = [k for k in keys_float_ta if self._is_embedding_layer(k)]
                        keys_float_ta = [k for k in keys_float_ta if not self._is_embedding_layer(k)]
                        if keys_embedding_ta:
                            print(f"[Task Arithmetic] Excluding {len(keys_embedding_ta)} embedding layers from projection")
                            print(f"[Task Arithmetic] Embedding layers (merged in original space): {keys_embedding_ta[:3]}{'...' if len(keys_embedding_ta) > 3 else ''}")
                    else:
                        keys_embedding_ta = []

                    # Build layer index map for layer_progressive and layer_group strategies
                    if self.use_adaptive_proj_size and self.adaptive_proj_strategy in ("layer_progressive", "layer_group"):
                        param_to_layer_idx_ta, num_layers_ta = self._build_layer_index_map(keys_float_ta)
                    else:
                        param_to_layer_idx_ta, num_layers_ta = {}, 1

                    # operator cache (include transform type!)
                    op_cache_ta: Dict[Tuple[str, str, int, int], Tuple[Any, Any]] = {}

                    def proj_seed_key_ta(param_name: str) -> str:
                        if self.subspace_scope == "global":
                            return "global"
                        elif self.subspace_scope == "layer":
                            return layer_key(param_name)
                        else:
                            return param_name

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
                            cur_D = d_last
                        else:  # global
                            if "global_D_ta" not in locals():
                                global_D_ta = max(tv_cpu[k].shape[-1] for k in keys_float_ta)
                                print(f"[Projection Test] Global dimension: {global_D_ta}")
                            cur_D = global_D_ta

                        # Compute projection size (adaptive or fixed)
                        if self.use_adaptive_proj_size:
                            layer_idx_ta = param_to_layer_idx_ta.get(name, 0)
                            all_dims = getattr(self, '_all_dims', None)
                            if self.adaptive_proj_mode == "layer":
                                lkey = layer_key(name)
                                layer_params_ta = {k: tv_cpu[k] for k in keys_float_ta if layer_key(k) == lkey}
                                proj_dim = self._compute_proj_size(
                                    name, tb, layer_params_ta, layer_idx_ta, num_layers_ta, all_dims
                                )
                            else:
                                proj_dim = self._compute_proj_size(
                                    name, tb, None, layer_idx_ta, num_layers_ta, all_dims
                                )
                        else:
                            proj_dim = max(1, int(cur_D * self.proj_ratio))

                        cache_key = (self.transform_type, seed_key, cur_D, proj_dim)

                        if cache_key not in op_cache_ta:
                            fwd, lift = create_projection_ops(
                                cur_D,
                                proj_dim,
                                seed_key=seed_key,
                                device=dev,
                                transform_type=self.transform_type,
                            )
                            op_cache_ta[cache_key] = (fwd, lift)
                        else:
                            fwd, lift = op_cache_ta[cache_key]

                        reconstructed_tv = torch.zeros_like(original_tv)
                        br = min(self.block_rows, rows)
                        cursor = 0

                        while cursor < rows:
                            take = min(rows - cursor, br)
                            sl = original_tv[cursor : cursor + take, :]

                            if cur_D > d_last:
                                sl_input = torch.nn.functional.pad(sl, (0, cur_D - d_last))
                            else:
                                sl_input = sl

                            Y = fwd(sl_input.to(dev, non_blocking=True))
                            X_rec = lift(Y).to("cpu", non_blocking=True)

                            if cur_D > d_last:
                                X_rec = X_rec[:, :d_last]

                            reconstructed_tv[cursor : cursor + take, :] = X_rec
                            cursor += take

                        diff = reconstructed_tv - original_tv
                        tensor_error = float(diff.pow(2).sum().item())
                        tensor_norm = float(original_tv.pow(2).sum().item())

                        total_reconstruction_error += tensor_error
                        total_norm += tensor_norm

                        if tensor_norm > 0:
                            relative_error = tensor_error / tensor_norm
                            reconstruction_errors.append((name, relative_error))

                        tv_cpu[name] = reconstructed_tv.view(tb.shape).to(tb.dtype)

                    if total_norm > 0:
                        global_relative_error = total_reconstruction_error / total_norm
                        print(f"\n[Reconstruction Error] Global relative error: {global_relative_error:.6e}")
                        print(f"[Reconstruction Error] Total squared error: {total_reconstruction_error:.6e}")
                        print(f"[Reconstruction Error] Total norm: {total_norm:.6e}")

                        reconstruction_errors.sort(key=lambda x: x[1], reverse=True)
                        print(f"\n[Reconstruction Error] Top 10 tensors by relative error:")
                        for i, (name, rel_err) in enumerate(reconstruction_errors[:10], 1):
                            print(f"  {i}. {name}: {rel_err:.6e}")

                    merged_sd = state_dict_add(base_dict, tv_cpu)
                    # 🚀 OPTIMIZATION: Remove redundant .clone()
                    base_cpu = {k: v.detach().cpu() for k, v in merged_sd.items()}
                else:
                    merged_sd = state_dict_add(base_dict, task_vector)
                    # 🚀 OPTIMIZATION: Remove redundant .clone()
                    base_cpu = {k: v.detach().cpu() for k, v in merged_sd.items()}

                print("=" * 50 + "\n")

                print("[Task Arithmetic] Reconstruction test complete, skipping normal Fastfood merging")
                merged_tensors = len(keys_float_ta) if self.report_reconstruction_error or self.proj_ratio < 1.0 else 0
                changed_params = merged_tensors

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
            k
            for k in keys_all
            if (k in donors_sd[0])
            and torch.is_floating_point(base_sd[k])
            and base_sd[k].ndim >= 1
            and all((k in d) and torch.is_floating_point(d[k]) and d[k].shape == base_sd[k].shape for d in donors_sd)
        ]

        if self.only_project_linear:
            keys_linear = [k for k in keys_float if base_sd[k].ndim == 2]
            keys_nonlinear = [k for k in keys_float if base_sd[k].ndim != 2]
            print(
                f"[only_project_linear] linear (2D) tensors={len(keys_linear)} | non-linear tensors={len(keys_nonlinear)}"
            )
        else:
            keys_linear = keys_float
            keys_nonlinear = []
        
        # Filter out embedding layers if project_embeddings=False
        if not self.project_embeddings:
            keys_embedding = [k for k in keys_linear if self._is_embedding_layer(k)]
            keys_linear = [k for k in keys_linear if not self._is_embedding_layer(k)]
            keys_nonlinear.extend(keys_embedding)
            if keys_embedding:
                print(f"[project_embeddings=False] Excluding {len(keys_embedding)} embedding layers from projection")
                print(f"[project_embeddings=False] Embedding layers (merged in original space): {keys_embedding[:3]}{'...' if len(keys_embedding) > 3 else ''}")

        K = len(donor_names)
        print(
            f"[Setup] donors={K} | total tensors={len(keys_all)} | eligible float tensors={len(keys_float)}"
        )

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
            return param_name

        # Determine global D (max last-dim) if needed
        global_D = None
        if self.subspace_scope == "global":
            maxd = 1
            for k in keys_float:
                t = base_sd[k]
                maxd = max(maxd, int(t.shape[-1]))
            global_D = maxd
            self._global_D = global_D  # Store for adaptive sizing helper

        # Report subspace sizing (first few examples)
        def _dim_for(k: str) -> Tuple[int, int]:
            t = base_sd[k]
            if self.subspace_scope == "per_flat_tensor" and t.ndim == 2:
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
                print(
                    f"   - {k}: original_last_dim={int(base_sd[k].shape[-1])} | scoped_dim={D} → proj_dim={m} (compression={m/max(1,D):.3f})"
                )

        if self.merge_func == "ema":
            print(
                f"[EMA] task_order={self.ema_task_order} | gamma={self.ema_gamma:.3f} | w_c={self.ema_w_c:.3f} | w_s={self.ema_w_s:.3f}"
            )

        # ---------- Work on CPU copies with pinned memory ----------
        # 🚀 OPTIMIZATION: Use pinned memory for faster H2D/D2H transfers
        use_pinned = torch.cuda.is_available() and self.device.type == "cuda"
        
        if use_pinned:
            # Create pinned tensors for faster async GPU transfers
            base_cpu = {}
            for k, v in base_sd.items():
                v_cpu = v.detach().cpu()
                if v_cpu.is_floating_point():
                    v_pinned = torch.empty_like(v_cpu).pin_memory()
                    v_pinned.copy_(v_cpu)
                    base_cpu[k] = v_pinned
                else:
                    base_cpu[k] = v_cpu
            
            donors_cpu = []
            for d in donors_sd:
                d_cpu = {}
                for k, v in d.items():
                    v_cpu = v.detach().cpu()
                    if v_cpu.is_floating_point():
                        v_pinned = torch.empty_like(v_cpu).pin_memory()
                        v_pinned.copy_(v_cpu)
                        d_cpu[k] = v_pinned
                    else:
                        d_cpu[k] = v_cpu
                donors_cpu.append(d_cpu)
        else:
            # Fallback for CPU-only mode
            base_cpu = {k: v.detach().cpu() for k, v in base_sd.items()}
            donors_cpu = [{k: v.detach().cpu() for k, v in d.items()} for d in donors_sd]
        
        # Save original base for LiNeS and Consensus Mask if needed
        if self.use_lines or self.use_consensus_mask:
            if self.use_lines:
                print("\n[LiNeS] Saving original base model for post-processing")
            if self.use_consensus_mask:
                print("\n[Consensus Mask] Saving original base model and individual task vectors for post-processing")
            original_base_cpu = {k: v.clone() for k, v in base_cpu.items()}
            
            # Save individual task vectors for TALL mask generation
            if self.use_consensus_mask:
                individual_task_vectors = {}
                for i, (donor_name, donor_sd_cpu) in enumerate(zip(donor_names, donors_cpu)):
                    task_vector = {}
                    for k in keys_float:
                        if k in donor_sd_cpu and k in base_cpu:
                            # Task vector: δ_t = θ_t - θ_0
                            task_vector[k] = donor_sd_cpu[k].float() - base_cpu[k].float()
                    individual_task_vectors[donor_name] = task_vector
                print(f"[Consensus Mask] Saved {len(individual_task_vectors)} individual task vectors for {len(keys_float)} parameters")
        
        dev = self.device

        # Adaptive projection size initialization (after base_cpu is created)
        if self.use_adaptive_proj_size:
            print(
                f"[Adaptive Proj Size] mode={self.adaptive_proj_mode} | strategy={self.adaptive_proj_strategy} | beta={self.proj_size_cfg.beta:.2f}"
            )

            if self.adaptive_proj_mode == "layer":
                self._layer_groups: Dict[str, Dict[str, torch.Tensor]] = {}
                for k in keys_linear:
                    lkey = layer_key(k)
                    if lkey not in self._layer_groups:
                        self._layer_groups[lkey] = {}
                    self._layer_groups[lkey][k] = base_cpu[k]
                print(
                    f"[Adaptive Proj Size] Grouped {len(keys_linear)} tensors into {len(self._layer_groups)} layers"
                )
            else:
                self._layer_groups = None

        # ---------- Merge ----------
        merged_tensors = 0
        changed_params = 0

        # Build layer index map for layer_progressive, layer_group, and layer_power_law strategies
        # Use keys_linear when only_project_linear=True to only count projected parameters
        if self.use_adaptive_proj_size and self.adaptive_proj_strategy in ("layer_progressive", "layer_group", "layer_power_law"):
            layer_map_keys = keys_linear if self.only_project_linear else keys_float
            
            # For layer_power_law, group parameters by layer and collect dimensions per layer
            if self.adaptive_proj_strategy == "layer_power_law":
                from .projection_size_estimator import last_dim_from_tensor
                
                # Group parameters by layer
                self._layer_dims_map: Dict[str, List[int]] = {}
                for k in layer_map_keys:
                    if k in base_cpu:
                        lkey = layer_key(k)
                        if lkey not in self._layer_dims_map:
                            self._layer_dims_map[lkey] = []
                        self._layer_dims_map[lkey].append(last_dim_from_tensor(base_cpu[k]))
                
                # Print summary
                total_params = sum(len(dims) for dims in self._layer_dims_map.values())
                total_dim_sum = sum(sum(dims) for dims in self._layer_dims_map.values())
                print(f"[Layer Power-Law] Grouped {total_params} parameters into {len(self._layer_dims_map)} layers")
                print(f"[Layer Power-Law] Total dimension sum: {total_dim_sum}, per-layer budget ratio: {self.proj_size_cfg.global_ratio:.2f}")
                print(f"[Layer Power-Law] Power-law alpha: {self.proj_size_cfg.power_law_alpha:.2f}")
                
                # Initialize projection tracking for detailed report
                self._layer_power_law_projections: Dict[str, Dict[str, Tuple[int, int, float]]] = {}
                
                param_to_layer_idx, num_layers = {}, 1  # Not used for layer_power_law
                self._all_dims = None  # Not used anymore
            elif self.adaptive_proj_strategy in ("layer_progressive", "layer_group"):
                param_to_layer_idx, num_layers = self._build_layer_index_map(layer_map_keys)
                self._layer_dims_map = None
            else:
                param_to_layer_idx, num_layers = {}, 1
                self._layer_dims_map = None
            if self.adaptive_proj_strategy == "layer_progressive":
                print(f"[Layer Progressive] Detected {num_layers} layers for progressive sizing")
            elif self.adaptive_proj_strategy == "layer_group":
                print(f"[Layer Group] Detected {num_layers} layers, boundary at layer {self.proj_size_cfg.group_boundary_layer}")
        else:
            param_to_layer_idx, num_layers = {}, 1

        # operator cache keyed by (transform_type, seed_key, cur_D, proj_dim)
        op_cache: Dict[Tuple[str, str, int, int], Tuple[Any, Any]] = {}

        # Small sample for lift error (subspace only)
        lift_err_num = 0.0
        lift_err_den = 0.0

        with self.profile("merging models"):
            # ---------- Process non-linear weights (1D) with mean in original space ----------
            if self.only_project_linear and keys_nonlinear:
                print(
                    f"[Merging non-linear weights] Processing {len(keys_nonlinear)} non-linear tensors with mean in original space"
                )
                for name in keys_nonlinear:
                    result = base_cpu[name].clone().float()
                    for i, dsd in enumerate(donors_cpu):
                        donor_val = dsd[name].float()
                        result += (donor_val - result) / (i + 2)
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

                # Special case: per_flat_tensor
                if self.subspace_scope == "per_flat_tensor":
                    flat_dim = tb.numel()
                    vb_flat = tb.view(-1).float()

                    layer_idx = param_to_layer_idx.get(name, 0)
                    all_dims = getattr(self, '_all_dims', None)
                    if self.use_adaptive_proj_size and self.adaptive_proj_mode == "layer":
                        lkey = layer_key(name)
                        layer_params = self._layer_groups.get(lkey, {name: tb})
                        proj_dim = self._compute_proj_size(name, tb, layer_params, layer_idx, num_layers, all_dims)
                    else:
                        proj_dim = self._compute_proj_size(name, tb, None, layer_idx, num_layers, all_dims)

                    seed_key = proj_seed_key(name)
                    cur_D = flat_dim
                    
                    # Compute the maximum allowed projection size based on transform type
                    # For DCT/DHT: L = D (no padding), For FWHT/SRHT/Fastfood: L = next_pow2(D)
                    if self.transform_type in ("dct", "dht"):
                        max_proj_dim = cur_D
                    else:  # fwht, srht, fastfood
                        # Compute next power of 2
                        max_proj_dim = 1 << (cur_D - 1).bit_length()
                    
                    # Clip projection dimension to not exceed the transform's capacity
                    proj_dim = min(proj_dim, max_proj_dim)
                    
                    # Track projection for layer_power_law reporting
                    if self.adaptive_proj_strategy == "layer_power_law" and hasattr(self, '_layer_power_law_projections'):
                        lkey = layer_key(name)
                        if lkey not in self._layer_power_law_projections:
                            self._layer_power_law_projections[lkey] = {}
                        effective_ratio = proj_dim / d_last if d_last > 0 else 0.0
                        self._layer_power_law_projections[lkey][name] = (d_last, proj_dim, effective_ratio)
                    
                    cache_key = (self.transform_type, seed_key, cur_D, proj_dim)

                    # Get projection operators (single or multi-sketch)
                    fwd_ops, lift_ops = self._create_projection_operators(cur_D, proj_dim, seed_key, dev, op_cache)

                    # Collect all donor deltas
                    Xs: List[Tensor] = []
                    for dsd in donors_cpu:
                        donor_flat = dsd[name].view(-1).float()
                        delta = donor_flat - vb_flat
                        Xs.append(delta)

                    # 🚀 OPTIMIZATION: Batch all donors into single GPU call
                    X_batch = torch.stack(Xs, dim=0).to(dev, non_blocking=True)  # [K, flat_dim]
                    
                    # Multi-sketch: project with each sketch operator
                    Y_sketches = self._multi_sketch_project(X_batch, fwd_ops)

                    if self.merge_where == "postlift":
                        # Lift each sketch, then merge in original space, then ensemble
                        Xhats_sketches = []
                        for Y_j, lift_j in zip(Y_sketches, lift_ops):
                            Y_j_flat = Y_j.view(-1, Y_j.shape[-1])
                            Xhats_j = lift_j(Y_j_flat).view(len(Xs), -1)  # [K, flat_dim]
                            Xhats_sketches.append(Xhats_j)
                        
                        # Merge each sketch separately in original space
                        Xmerge_sketches = []
                        for Xhats_j in Xhats_sketches:
                            Xmerge_j = _zero_aware_aggregate(
                                Xhats_j,  # Already on GPU
                                merge_func=self.merge_func,
                                weights=w
                                if self.merge_func
                                in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"}
                                else None,
                                ema_task_order=self.ema_task_order,
                                ema_gamma=self.ema_gamma,
                                ema_w_c=self.ema_w_c,
                                ema_w_s=self.ema_w_s,
                                ema_custom_order=ema_custom_indices,
                            )
                            Xmerge_sketches.append(Xmerge_j)
                        
                        # Ensemble across sketches
                        if self.num_sketches > 1:
                            Xmerge_stacked = torch.stack(Xmerge_sketches, dim=0)
                            if self.sketch_ensemble_mode == "mean":
                                Xmerge = Xmerge_stacked.mean(dim=0)
                            elif self.sketch_ensemble_mode == "sum":
                                Xmerge = Xmerge_stacked.sum(dim=0)
                            elif self.sketch_ensemble_mode == "max":
                                Xmerge = Xmerge_stacked.abs().max(dim=0)[0] * torch.sign(Xmerge_stacked.sum(dim=0))
                            elif self.sketch_ensemble_mode == "median":
                                Xmerge = Xmerge_stacked.median(dim=0)[0]
                        else:
                            Xmerge = Xmerge_sketches[0]
                        
                        Xmerge = Xmerge.to("cpu", non_blocking=True)
                    else:
                        # Merge in subspace for each sketch, then lift and ensemble
                        Ymerge_sketches = []
                        for Y_j in Y_sketches:
                            Ymerge_j = _zero_aware_aggregate(
                                Y_j,  # Already on GPU
                                merge_func=self.merge_func,
                                weights=w
                                if self.merge_func
                                in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"}
                                else None,
                                ema_task_order=self.ema_task_order,
                                ema_gamma=self.ema_gamma,
                                ema_w_c=self.ema_w_c,
                                ema_w_s=self.ema_w_s,
                                ema_custom_order=ema_custom_indices,
                            )
                            Ymerge_sketches.append(Ymerge_j)
                        
                        # Lift and ensemble
                        Xmerge = self._multi_sketch_lift_and_ensemble(Ymerge_sketches, lift_ops).to("cpu", non_blocking=True)

                        if lift_err_den < 1e8:
                            # Use first sketch for error estimation
                            X0_rec = lift_ops[0](Y_sketches[0][0:1]).view(-1)
                            diff = (X0_rec.to(torch.float32) - X_batch[0].to(torch.float32))
                            lift_err_num += float(diff.pow(2).sum().item())
                            lift_err_den += float(X_batch[0].pow(2).sum().item())

                    upd = (self.scale * Xmerge).view_as(tb).to(tb.dtype)

                    max_ratio = 2.0
                    upd_norm = upd.norm().item()
                    base_norm = tb.norm().item() + 1e-12
                    if upd_norm > max_ratio * base_norm:
                        scale_factor = (max_ratio * base_norm) / (upd_norm + 1e-12)
                        upd = upd * scale_factor

                    tb.add_(upd)

                    merged_tensors += 1
                    if upd.abs().max().item() > 0:
                        changed_params += 1

                    continue  # done with per_flat_tensor

                # Standard row-wise processing
                vb = tb.view(rows, d_last).float()
                br = min(self.block_rows, rows)

                layer_idx = param_to_layer_idx.get(name, 0)
                all_dims = getattr(self, '_all_dims', None)
                if self.use_adaptive_proj_size and self.adaptive_proj_mode == "layer":
                    lkey = layer_key(name)
                    layer_params = self._layer_groups.get(lkey, {name: tb})
                    proj_dim = self._compute_proj_size(name, tb, layer_params, layer_idx, num_layers, all_dims)
                else:
                    proj_dim = self._compute_proj_size(name, tb, None, layer_idx, num_layers, all_dims)

                seed_key = proj_seed_key(name)
                cur_D = global_D if (global_D is not None) else d_last
                
                # Compute the maximum allowed projection size based on transform type
                # For DCT/DHT: L = D (no padding), For FWHT/SRHT/Fastfood: L = next_pow2(D)
                if self.transform_type in ("dct", "dht"):
                    max_proj_dim = cur_D
                else:  # fwht, srht, fastfood
                    # Compute next power of 2
                    max_proj_dim = 1 << (cur_D - 1).bit_length()
                
                # Clip projection dimension to not exceed the transform's capacity
                proj_dim = min(proj_dim, max_proj_dim)
                
                # Track projection for layer_power_law reporting
                if self.adaptive_proj_strategy == "layer_power_law" and hasattr(self, '_layer_power_law_projections'):
                    lkey = layer_key(name)
                    if lkey not in self._layer_power_law_projections:
                        self._layer_power_law_projections[lkey] = {}
                    effective_ratio = proj_dim / d_last if d_last > 0 else 0.0
                    self._layer_power_law_projections[lkey][name] = (d_last, proj_dim, effective_ratio)
                
                cache_key = (self.transform_type, seed_key, cur_D, proj_dim)

                # Get projection operators (single or multi-sketch)
                fwd_ops, lift_ops = self._create_projection_operators(cur_D, proj_dim, seed_key, dev, op_cache)

                # 🚀 OPTIMIZATION: Pre-allocate GPU buffers for this tensor
                max_block_size = min(br, rows)
                if use_pinned:
                    # Pre-allocate pinned CPU buffers
                    X_batch_cpu_buffer = torch.empty((len(donors_cpu), max_block_size, cur_D), 
                                                      dtype=torch.float32).pin_memory()
                else:
                    X_batch_cpu_buffer = torch.empty((len(donors_cpu), max_block_size, cur_D), 
                                                      dtype=torch.float32)
                
                # Pre-allocate GPU buffers
                X_batch_gpu_buffer = torch.empty((len(donors_cpu), max_block_size, cur_D), 
                                                  dtype=torch.float32, device=dev)
                # Note: For multi-sketch, we'll allocate Y buffers per-sketch on-demand

                cursor = 0
                tensor_changed = False

                while cursor < rows:
                    take = min(rows - cursor, br)
                    sl_base = vb[cursor : cursor + take, :]

                    # 🚀 OPTIMIZATION: Use pre-allocated buffer slices
                    X_batch_view = X_batch_cpu_buffer[:, :take, :]
                    
                    # Collect all donor deltas directly into buffer
                    for i, dsd in enumerate(donors_cpu):
                        sl_donor = dsd[name].view(rows, d_last).float()[cursor : cursor + take, :]
                        delta = sl_donor - sl_base
                        if global_D is not None and d_last < cur_D:
                            X_batch_view[i, :, :d_last].copy_(delta)
                            X_batch_view[i, :, d_last:].zero_()
                        else:
                            X_batch_view[i].copy_(delta)

                    # 🚀 OPTIMIZATION: Use pre-allocated GPU buffers, async transfer
                    X_batch_gpu_view = X_batch_gpu_buffer[:, :take, :]
                    X_batch_gpu_view.copy_(X_batch_view, non_blocking=True)
                    
                    X_batch_flat = X_batch_gpu_view.view(-1, X_batch_gpu_view.shape[-1])  # [K*take, d]
                    
                    # Multi-sketch: project with each sketch operator
                    Y_sketches = []
                    for fwd_j in fwd_ops:
                        Y_j_flat = fwd_j(X_batch_flat)  # [K*take, m]
                        Y_j = Y_j_flat.view(len(donors_cpu), take, -1)  # [K, take, m]
                        Y_sketches.append(Y_j)

                    if self.merge_where == "postlift":
                        # Lift each sketch, merge in original space, then ensemble
                        Xmerge_sketches = []
                        for Y_j, lift_j in zip(Y_sketches, lift_ops):
                            Y_j_flat = Y_j.view(-1, Y_j.shape[-1])
                            Xhats_j_flat = lift_j(Y_j_flat)  # [K*take, cur_D]
                            Xhats_j = Xhats_j_flat.view(len(donors_cpu), take, -1)  # [K, take, cur_D]
                            Xhats_j = Xhats_j[:, :, :d_last] if cur_D > d_last else Xhats_j
                            
                            Xmerge_j = _zero_aware_aggregate(
                                Xhats_j,  # Already on GPU
                                merge_func=self.merge_func,
                                weights=w
                                if self.merge_func
                                in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"}
                                else None,
                                ema_task_order=self.ema_task_order,
                                ema_gamma=self.ema_gamma,
                                ema_w_c=self.ema_w_c,
                                ema_w_s=self.ema_w_s,
                                ema_custom_order=ema_custom_indices,
                            )
                            Xmerge_sketches.append(Xmerge_j)
                        
                        # Ensemble across sketches
                        if self.num_sketches > 1:
                            Xmerge_stacked = torch.stack(Xmerge_sketches, dim=0)
                            if self.sketch_ensemble_mode == "mean":
                                Xmerge = Xmerge_stacked.mean(dim=0)
                            elif self.sketch_ensemble_mode == "sum":
                                Xmerge = Xmerge_stacked.sum(dim=0)
                            elif self.sketch_ensemble_mode == "max":
                                Xmerge = Xmerge_stacked.abs().max(dim=0)[0] * torch.sign(Xmerge_stacked.sum(dim=0))
                            elif self.sketch_ensemble_mode == "median":
                                Xmerge = Xmerge_stacked.median(dim=0)[0]
                        else:
                            Xmerge = Xmerge_sketches[0]
                        
                        Xmerge = Xmerge.to("cpu", non_blocking=True)
                    else:
                        # Merge in subspace for each sketch, then lift and ensemble
                        Ymerge_sketches = []
                        for Y_j in Y_sketches:
                            Ymerge_j = _zero_aware_aggregate(
                                Y_j,  # Already on GPU, [K, take, m]
                                merge_func=self.merge_func,
                                weights=w
                                if self.merge_func
                                in {"sum", "mean", "ema", "ties_sum", "ties_mean", "ties_max"}
                                else None,
                                ema_task_order=self.ema_task_order,
                                ema_gamma=self.ema_gamma,
                                ema_w_c=self.ema_w_c,
                                ema_w_s=self.ema_w_s,
                                ema_custom_order=ema_custom_indices,
                            )
                            Ymerge_sketches.append(Ymerge_j)
                        
                        # Lift each sketch and ensemble
                        Xmerge_full_sketches = []
                        for Ymerge_j, lift_j in zip(Ymerge_sketches, lift_ops):
                            Xmerge_full_j = lift_j(Ymerge_j)  # [take, cur_D]
                            Xmerge_full_sketches.append(Xmerge_full_j)
                        
                        # Ensemble
                        if self.num_sketches > 1:
                            Xmerge_full_stacked = torch.stack(Xmerge_full_sketches, dim=0)
                            if self.sketch_ensemble_mode == "mean":
                                Xmerge_full = Xmerge_full_stacked.mean(dim=0)
                            elif self.sketch_ensemble_mode == "sum":
                                Xmerge_full = Xmerge_full_stacked.sum(dim=0)
                            elif self.sketch_ensemble_mode == "max":
                                Xmerge_full = Xmerge_full_stacked.abs().max(dim=0)[0] * torch.sign(Xmerge_full_stacked.sum(dim=0))
                            elif self.sketch_ensemble_mode == "median":
                                Xmerge_full = Xmerge_full_stacked.median(dim=0)[0]
                        else:
                            Xmerge_full = Xmerge_full_sketches[0]
                        
                        Xmerge = (Xmerge_full[:, :d_last] if cur_D > d_last else Xmerge_full).to("cpu", non_blocking=True)

                        if take > 0 and (lift_err_den < 1e8):
                            # Use first sketch for error estimation
                            X0_rec = Xmerge_full_sketches[0][0:1, :d_last] if cur_D > d_last else Xmerge_full_sketches[0][0:1]
                            X0_orig = X_batch_gpu_view[0, :1, :d_last] if cur_D > d_last else X_batch_gpu_view[0, :1]
                            diff = (X0_rec.to(torch.float32) - X0_orig.to(torch.float32))
                            lift_err_num += float(diff.pow(2).sum().item())
                            lift_err_den += float(X0_orig.pow(2).sum().item())

                    upd = (self.scale * Xmerge).to(sl_base.dtype)

                    max_ratio = 2.0
                    upd_norm = upd.norm().item()
                    base_norm = sl_base.norm().item() + 1e-12
                    if upd_norm > max_ratio * base_norm:
                        scale_factor = (max_ratio * base_norm) / (upd_norm + 1e-12)
                        upd = upd * scale_factor

                    vb[cursor : cursor + take, :].add_(upd)
                    tensor_changed = tensor_changed or bool(upd.abs().max().item() > 0)
                    cursor += take

                merged_tensors += 1
                if tensor_changed:
                    changed_params += 1

            # 🚀 OPTIMIZATION: Removed torch.cuda.empty_cache() from hot loop
            # (expensive operation, rarely beneficial, adds ~5-10% overhead)

        # ---------- Stats / sanity ----------
        bad, total_float = [], 0
        for n, p in base_cpu.items():
            if not torch.is_floating_point(p):
                continue
            total_float += 1
            if torch.isnan(p).any() or torch.isinf(p).any():
                bad.append(n)

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

        print("\n=== Merge Summary ===")
        print(
            f"[Summary] donors={K} | eligible_tensors={len(keys_float)} | processed={merged_tensors} | changed_tensors={changed_params}"
        )
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

        # ---------- Generate Layer Power-Law Projection Report ----------
        if self.use_adaptive_proj_size and self.adaptive_proj_strategy == "layer_power_law" and hasattr(self, '_layer_power_law_projections'):
            print("\n=== Layer Power-Law Projection Report ===")
            self._print_layer_power_law_report()
            
            # Save report to JSON file
            try:
                report_data = self._generate_layer_power_law_report_data()
                
                import json
                from pathlib import Path
                
                # Determine output directory
                output_dir = self._get_output_directory()
                output_dir.mkdir(parents=True, exist_ok=True)
                
                report_path = output_dir / "layer_power_law_report.json"
                with open(report_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                print(f"[Layer Power-Law] Report saved to: {report_path}")
            except Exception as e:
                print(f"[Layer Power-Law] Warning: Could not save report to file: {e}")

        # ---------- Generate Layer Projection Report (for layer_progressive and layer_group) ----------
        if self.use_adaptive_proj_size and self.adaptive_proj_strategy in ("layer_progressive", "layer_group"):
            print("\n=== Generating Layer Projection Report ===")
            try:
                report = self._generate_layer_projection_report(
                    param_to_layer_idx, num_layers, keys_linear, base_cpu
                )
                
                # Save report to JSON file
                import json
                from pathlib import Path
                import os
                
                # Determine output directory
                if hasattr(self, 'fabric') and hasattr(self.fabric, 'loggers'):
                    # Try to get output directory from fabric loggers
                    output_dir = None
                    for logger in self.fabric.loggers:
                        if hasattr(logger, 'log_dir'):
                            output_dir = Path(logger.log_dir)
                            break
                    if output_dir is None:
                        output_dir = Path.cwd() / "outputs"
                else:
                    # Use current working directory or hydra output dir
                    from omegaconf import DictConfig
                    import hydra
                    if hasattr(hydra, 'core') and hasattr(hydra.core, 'hydra_config') and hydra.core.hydra_config.HydraConfig.initialized():
                        hconf = hydra.core.hydra_config.HydraConfig.get()
                        output_dir = Path(hconf.runtime.output_dir)
                    else:
                        output_dir = Path.cwd() / "outputs"
                
                output_dir.mkdir(parents=True, exist_ok=True)
                report_path = output_dir / "layer_projection_report.json"
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                print(f"[Layer Projection Report] Saved to: {report_path}")
                print(f"[Layer Projection Report] Strategy: {self.adaptive_proj_strategy}")
                print(f"[Layer Projection Report] Number of layers: {num_layers}")
                
                # Print summary
                if self.adaptive_proj_strategy == "layer_group":
                    boundary = self.proj_size_cfg.group_boundary_layer
                    feature_ratio = self.proj_size_cfg.feature_proj_ratio
                    head_ratio = self.proj_size_cfg.head_proj_ratio
                    print(f"[Layer Projection Report] Boundary: layer {boundary}")
                    print(f"[Layer Projection Report] Feature layers (0-{boundary-1}): ratio={feature_ratio:.3f}")
                    print(f"[Layer Projection Report] Head layers ({boundary}-{num_layers-1}): ratio={head_ratio:.3f}")
                elif self.adaptive_proj_strategy == "layer_progressive":
                    start_ratio = self.proj_size_cfg.start_proj_ratio
                    end_ratio = self.proj_size_cfg.end_proj_ratio
                    growth_mode = self.proj_size_cfg.growth_mode
                    print(f"[Layer Projection Report] Start ratio: {start_ratio:.3f}")
                    print(f"[Layer Projection Report] End ratio: {end_ratio:.3f}")
                    print(f"[Layer Projection Report] Growth mode: {growth_mode}")
                
            except Exception as e:
                print(f"[Layer Projection Report] Warning: Failed to generate report: {e}")
                import traceback
                traceback.print_exc()

        # ---------- Apply LiNeS Scaling (Layer Scaling) ----------
        if self.use_lines:
            print("\n=== Applying LiNeS (Layer Scaling) ===")
            with self.profile("lines_scaling"):
                # Compute merged delta: delta = merged_model - base_model
                merged_delta = {}
                norm_summed_tvs = 0.0
                norm_merged_tv = 0.0
                
                for k in keys_float:
                    delta = base_cpu[k].float() - original_base_cpu[k].float()
                    merged_delta[k] = delta
                    norm_merged_tv += delta.abs().sum().item()
                
                # Compute norm of summed task vectors (for auto alpha)
                # summed_tv = sum of all individual task vectors
                for k in keys_float:
                    base_val = original_base_cpu[k].float()
                    for dsd in donors_cpu:
                        donor_val = dsd[k].float()
                        task_vec = donor_val - base_val
                        norm_summed_tvs += task_vec.abs().sum().item()
                
                print(f"[LiNeS] Merged delta L1 norm: {norm_merged_tv:.6e}")
                print(f"[LiNeS] Summed task vectors L1 norm: {norm_summed_tvs:.6e}")
                
                # Apply LiNeS scaling to the merged delta
                scaled_delta, computed_alpha = self._apply_lines_scaling(
                    merged_delta,
                    num_tasks=len(donors_cpu),
                    norm_summed_tvs=norm_summed_tvs,
                    norm_merged_tv=norm_merged_tv,
                )
                
                # Store LiNeS information for reporting
                self._lines_alpha_value = computed_alpha
                self._lines_beta_value = self.lines_beta
                self._lines_num_blocks_used = self.lines_num_blocks if self.lines_num_blocks else (24 if any(f".23." in k for k in merged_delta.keys()) else 12)
                
                # Add LiNeS metadata to runtime info for persistence in results files
                if not hasattr(self, '_runtime_info'):
                    self._runtime_info = {}
                self._runtime_info['lines_enabled'] = True
                self._runtime_info['lines_alpha'] = float(computed_alpha) if computed_alpha is not None else None
                self._runtime_info['lines_beta'] = float(self.lines_beta)
                self._runtime_info['lines_num_blocks'] = int(self._lines_num_blocks_used)
                self._runtime_info['lines_auto_alpha'] = bool(self.lines_auto_alpha)
                
                # Reconstruct final model: base + scaled_delta
                for k in keys_float:
                    base_cpu[k] = (original_base_cpu[k].float() + scaled_delta[k]).to(original_base_cpu[k].dtype)
                
                print(f"[LiNeS] Applied layer-wise scaling to {len(scaled_delta)} parameters")
                print("=" * 50)
        else:
            # LiNeS not enabled - add runtime info to indicate this
            if not hasattr(self, '_runtime_info'):
                self._runtime_info = {}
            self._runtime_info['lines_enabled'] = False

        # ---------- Apply Subspace Boosting (if enabled) ----------
        if self.use_subspace_boosting:
            print("\n=== Applying Subspace Boosting ===")
            with self.profile("subspace_boosting"):
                # Compute task vectors for boosting
                boosted_count = 0
                skipped_count = 0
                
                for k in keys_float:
                    # Only apply to linear layers (attention + MLP weights)
                    if not self._is_linear_layer(k):
                        skipped_count += 1
                        continue
                    
                    # Only apply to 2D tensors (weight matrices)
                    if base_cpu[k].ndim != 2:
                        skipped_count += 1
                        continue
                    
                    # Compute task vector: merged - base
                    task_vector = base_cpu[k].float() - original_base_cpu[k].float()
                    
                    # Apply subspace boosting
                    boosted_tv = subspace_boosting(task_vector, beta=self.subspace_boosting_beta)
                    
                    # Reconstruct parameter: base + boosted_task_vector
                    base_cpu[k] = (original_base_cpu[k].float() + boosted_tv).to(original_base_cpu[k].dtype)
                    boosted_count += 1
                
                print(f"[Subspace Boosting] Beta: {self.subspace_boosting_beta:.4f}")
                print(f"[Subspace Boosting] Boosted {boosted_count} linear layer weights")
                print(f"[Subspace Boosting] Skipped {skipped_count} non-linear/non-2D parameters")
                
                # Store SB metadata for reporting
                if not hasattr(self, '_runtime_info'):
                    self._runtime_info = {}
                self._runtime_info['subspace_boosting_enabled'] = True
                self._runtime_info['subspace_boosting_beta'] = float(self.subspace_boosting_beta)
                self._runtime_info['subspace_boosting_count'] = int(boosted_count)
                print("=" * 50)
        else:
            # SB not enabled - add runtime info to indicate this
            if not hasattr(self, '_runtime_info'):
                self._runtime_info = {}
            self._runtime_info['subspace_boosting_enabled'] = False

        # ---------- Apply Consensus Masking with TALL Masks (if enabled) ----------
        if self.use_consensus_mask:
            print("\n=== Applying Consensus Masking (TALL Masks) ===")
            with self.profile("consensus_masking"):
                # Compute merged task vector: merged - base
                merged_task_vector = {}
                for k in keys_float:
                    merged_task_vector[k] = base_cpu[k].float() - original_base_cpu[k].float()
                
                # Generate per-task TALL masks
                print(f"[Consensus Mask] Generating task-specific masks with λ={self.tall_mask_lambda:.2f}")
                tall_masks = generate_tall_masks(
                    merged_delta=merged_task_vector,
                    individual_deltas=individual_task_vectors,
                    base_state=original_base_cpu,
                    tall_mask_lambda=self.tall_mask_lambda,
                )
                
                # Count mask statistics
                total_params = sum(v.numel() for v in merged_task_vector.values())
                task_activation_counts = {
                    task_name: sum(mask.sum().item() for mask in task_mask.values())
                    for task_name, task_mask in tall_masks.items()
                }
                
                print(f"[Consensus Mask] Generated masks for {len(tall_masks)} tasks")
                print(f"[Consensus Mask] Total parameters: {total_params:,}")
                for task_name, count in task_activation_counts.items():
                    activation_pct = 100.0 * count / total_params if total_params > 0 else 0.0
                    print(f"[Consensus Mask]   - {task_name}: {int(count):,} active ({activation_pct:.1f}%)")
                
                # Apply consensus filtering
                print(f"\n[Consensus Mask] Applying consensus threshold: {self.consensus_threshold}")
                filtered_delta = apply_consensus_mask(
                    merged_delta=merged_task_vector,
                    tall_masks=tall_masks,
                    consensus_threshold=self.consensus_threshold,
                )
                
                # Reconstruct final model: base + filtered_delta
                for k in keys_float:
                    base_cpu[k] = (original_base_cpu[k].float() + filtered_delta[k]).to(original_base_cpu[k].dtype)
                
                # Count consensus mask statistics
                consensus_count = sum(
                    ((filtered_delta[k] != 0).sum().item() if k in filtered_delta else 0)
                    for k in keys_float
                )
                consensus_pct = 100.0 * consensus_count / total_params if total_params > 0 else 0.0
                
                print(f"[Consensus Mask] Retained {consensus_count:,} parameters ({consensus_pct:.1f}%)")
                print(f"[Consensus Mask] Pruned {total_params - consensus_count:,} parameters ({100.0 - consensus_pct:.1f}%)")
                
                # Store metadata for reporting
                if not hasattr(self, '_runtime_info'):
                    self._runtime_info = {}
                self._runtime_info['consensus_mask_enabled'] = True
                self._runtime_info['tall_mask_lambda'] = float(self.tall_mask_lambda)
                self._runtime_info['consensus_threshold'] = int(self.consensus_threshold)
                self._runtime_info['task_activation_counts'] = {k: int(v) for k, v in task_activation_counts.items()}
                self._runtime_info['consensus_retained_count'] = int(consensus_count)
                self._runtime_info['consensus_retained_pct'] = float(consensus_pct)
                print("=" * 50)
        else:
            # Consensus masking not enabled
            if not hasattr(self, '_runtime_info'):
                self._runtime_info = {}
            self._runtime_info['consensus_mask_enabled'] = False

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

        if self.run_analysis and self.analysis_methods:
            print("\n=== Running Integrated Analysis ===")
            self._run_integrated_analysis(modelpool, model)

        return model

    def _run_integrated_analysis(self, modelpool: BaseModelPool, merged_model: nn.Module):
        print(f"[Analysis] Running {len(self.analysis_methods)} analysis methods")
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
        parts = []

        if self.proj_ratio != 0.95:
            parts.append(f"proj{self.proj_ratio}")
        if self.merge_func != "signmax":
            parts.append(f"{self.merge_func}")
        if self.subspace_scope != "global":
            parts.append(f"{self.subspace_scope}")
        if self.merge_where != "subspace":
            parts.append(f"{self.merge_where}")

        if self.merge_func == "ema":
            parts.append(f"g{self.ema_gamma}")
            parts.append(f"wc{self.ema_w_c}")
            parts.append(f"ws{self.ema_w_s}")
            if self.ema_task_order != "given":
                order_abbrev = {"random": "rand", "cosine_similarity": "cos", "custom": "cust"}
                parts.append(f"{order_abbrev.get(self.ema_task_order, self.ema_task_order)}")

        if self.merge_func.startswith("ties_"):
            parts.append("ties")

        if parts:
            return f"fastfood_{'_'.join(parts)}"
        else:
            return "fastfood_default"

    def _run_merged_task_vector_analysis(self, modelpool: BaseModelPool, method_id: str):
        try:
            from fusion_bench.method.analysis.merged_task_vector_analysis import MergedTaskVectorAnalysis

            analyzer = MergedTaskVectorAnalysis(
                merging_methods=["fastfood_merging"],
                proj_ratio=self.proj_ratio,
                use_G=self.use_G,
                merge_func=self.merge_func,
                subspace_scope=self.subspace_scope,
                merge_where=self.merge_where,
                trainable_only=True,
                output_path=self.analysis_output_path,
                device=str(self.device),
            )

            print(f"[Analysis] Running merged task vector analysis for {method_id}")
            analyzer.run(modelpool)

        except ImportError as e:
            print(f"[Analysis] Could not import MergedTaskVectorAnalysis: {e}")

    def _run_task_vector_similarity_analysis(self, modelpool: BaseModelPool, method_id: str):
        try:
            from fusion_bench.method.analysis.task_vector_cos_similarity import TaskVectorCosSimilarity

            analyzer = TaskVectorCosSimilarity(
                plot_heatmap=True,
                trainable_only=True,
                method_name=method_id,
                proj_ratio=self.proj_ratio,
                use_G=self.use_G,
                analyze_subspace=True,
                device=str(self.device),
                output_path=self.analysis_output_path,
            )

            print(f"[Analysis] Running task vector similarity analysis for {method_id}")
            analyzer.run(modelpool)

        except ImportError as e:
            print(f"[Analysis] Could not import TaskVectorCosSimilarity: {e}")

    def _run_task_vector_layer_analysis(self, modelpool: BaseModelPool, method_id: str):
        try:
            from fusion_bench.method.analysis.task_vector_layer_analysis import TaskVectorLayerAnalysis

            analyzer = TaskVectorLayerAnalysis(
                trainable_only=True,
                method_name=method_id,
                device=str(self.device),
                output_path=self.analysis_output_path,
            )

            print(f"[Analysis] Running layer-wise task vector analysis for {method_id}")
            analyzer.run(modelpool)

        except ImportError as e:
            print(f"[Analysis] Could not import TaskVectorLayerAnalysis: {e}")
