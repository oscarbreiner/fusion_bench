"""
FastFood Regularization Experiment

This method tests the effect of Fastfood projection as a regularization technique
by projecting individual task models (not merging them) and evaluating performance
on their respective test datasets.

The goal is to understand how lossy compression affects individual model performance
before considering it for model merging scenarios.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

import torch
from torch import nn

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.utils import LazyStateDict

# Import Fastfood utilities
from fusion_bench.method.fastfood_merging.fastfood_utils import create_fastfood_ops

log = logging.getLogger(__name__)


@auto_register_config
class FastfoodRegularizeAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    FastFood Regularization: Project individual task models without merging.
    
    This method evaluates the effect of Fastfood projection as a form of
    regularization by:
    1. Loading each fine-tuned task model
    2. Projecting parameters down to m dimensions and lifting back to D
    3. Evaluating on the task's test dataset
    4. Comparing with baseline (no projection)
    
    Parameters:
        proj_ratio: Compression ratio (0.0-1.0), e.g., 0.5 = 50% compression
        k_min: Minimum projection dimension (default 64, protects tiny tensors)
        use_G: Use Gaussian scaling in Fastfood transform
        device: Computation device (cuda/cpu)
        subspace_scope: "per_tensor" | "layer" | "global"
        block_rows: Memory management for large tensors
        exclude_parameters: Patterns to exclude from projection (e.g., normalization layers)
        
    Returns:
        A dictionary with results for each task model (original + projected)
    """
    
    def __init__(
        self,
        proj_ratio: float = 0.5,
        use_G: bool = False,
        device: str = "cuda",
        subspace_scope: str = "global",
        block_rows: int = 8192,
        k_min: int = 64,
        exclude_parameters: List[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.device = torch.device(device)
        self.subspace_scope = str(subspace_scope)
        self.block_rows = int(block_rows)
        self.k_min = int(k_min)
        self.exclude_parameters = list(exclude_parameters) if exclude_parameters is not None else [
            "*norm*", "*bias*", "*pos_embed*", "*cls_token*", "patch_embed.*", 
            "*running_*", "*num_batches_tracked*"
        ]
        
        log.info(f"FastFood Regularize initialized: proj_ratio={proj_ratio}, scope={subspace_scope}, k_min={k_min}")
    
    def _should_exclude(self, param_name: str) -> bool:
        """Check if parameter should be excluded from projection."""
        import fnmatch
        for pattern in self.exclude_parameters:
            if fnmatch.fnmatch(param_name, pattern):
                return True
        return False
    
    def _layer_key(self, name: str) -> str:
        """Extract layer key for layer-scoped projection."""
        parts = name.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
        if len(parts) >= 2:
            return ".".join(parts[:2])
        return name
    
    @torch.no_grad()
    def project_model(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Project model parameters down and up using Fastfood.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Projected state dictionary
        """
        log.info(f"Projecting model with proj_ratio={self.proj_ratio}")
        
        # Skip projection if proj_ratio >= 1.0 (baseline/identity)
        if self.proj_ratio >= 1.0:
            log.info("proj_ratio >= 1.0: Skipping projection (identity/baseline)")
            return {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        
        # Work on CPU copies
        projected_sd = {}
        dev = self.device
        
        # Identify eligible parameters
        keys_float = [
            k for k in state_dict.keys()
            if torch.is_floating_point(state_dict[k])
            and state_dict[k].ndim >= 1
            and not self._should_exclude(k)
        ]
        
        # Operator cache
        op_cache = {}
        
        # Projection statistics
        total_params = 0
        projected_params = 0
        reconstruction_error = 0.0
        reconstruction_norm = 0.0
        
        for k in state_dict.keys():
            if k not in keys_float:
                # Copy non-projected parameters as-is
                projected_sd[k] = state_dict[k].detach().cpu().clone()
                continue
            
            # Project this parameter
            tb = state_dict[k].detach().cpu().clone()
            d_last = int(tb.shape[-1])
            rows = tb.numel() // d_last
            
            if rows <= 0:
                projected_sd[k] = tb
                continue
            
            original = tb.view(rows, d_last).float()
            
            # Determine projection dimension based on ACTUAL dimension (not padded)
            seed_key = self._get_seed_key(k)
            
            # For per_tensor: use actual dimension d_last
            # For layer/global: use max dimension in scope
            if self.subspace_scope == "per_tensor":
                cur_D = d_last
            elif self.subspace_scope == "layer":
                # For layer scope, we'd need to scan all params in the layer
                # For now, use actual dimension (conservative approach)
                cur_D = d_last
            else:  # global
                # For global scope, compute max dimension across all eligible params
                if 'global_D' not in locals():
                    global_D = max(state_dict[key].shape[-1] for key in keys_float)
                    log.info(f"Global dimension: {global_D}")
                cur_D = global_D
            
            # Projection size based on actual dimension (not padded to power of 2)
            proj_dim = max(1, int(cur_D * self.proj_ratio))
            
            cache_key = (seed_key, cur_D, proj_dim)
            
            if cache_key not in op_cache:
                fwd, lift = create_fastfood_ops(
                    cur_D, proj_dim, seed_key=seed_key, device=dev, use_G=self.use_G, k_min=self.k_min
                )
                op_cache[cache_key] = (fwd, lift)
            else:
                fwd, lift = op_cache[cache_key]
            
            # Process in blocks
            reconstructed = torch.zeros_like(original)
            br = min(self.block_rows, rows)
            cursor = 0
            
            while cursor < rows:
                take = min(rows - cursor, br)
                sl = original[cursor:cursor + take, :]
                
                # NO EXTERNAL PADDING - operator handles power-of-2 padding internally
                # If using per_tensor scope, operator is sized for d_last
                # If using global scope and d_last < cur_D, we need to pad to cur_D
                if cur_D > d_last:
                    # Only pad if we're using a shared operator sized for larger dimension
                    sl_input = torch.nn.functional.pad(sl, (0, cur_D - d_last))
                else:
                    sl_input = sl
                
                # Project down and up
                Y = fwd(sl_input.to(dev, non_blocking=True))
                X_rec = lift(Y).to("cpu", non_blocking=True)
                
                # Extract only the original dimension (no need to slice if no external padding)
                if cur_D > d_last:
                    X_rec = X_rec[:, :d_last]
                
                reconstructed[cursor:cursor + take, :] = X_rec
                cursor += take
            
            # Compute reconstruction error
            diff = reconstructed - original
            reconstruction_error += float(diff.pow(2).sum().item())
            reconstruction_norm += float(original.pow(2).sum().item())
            
            # Store projected parameter
            projected_sd[k] = reconstructed.view(tb.shape).to(tb.dtype)
            
            total_params += original.numel()
            projected_params += 1
        
        # Report reconstruction error
        if reconstruction_norm > 0:
            rel_error = reconstruction_error / reconstruction_norm
            log.info(f"Projection stats: {projected_params}/{len(state_dict)} parameters projected")
            log.info(f"Reconstruction error: {rel_error:.6e} (relative)")
            log.info(f"Total squared error: {reconstruction_error:.6e}")
            log.info(f"Total norm: {reconstruction_norm:.6e}")
        
        return projected_sd
    
    def _get_seed_key(self, param_name: str) -> str:
        """Get seed key based on subspace scope."""
        if self.subspace_scope == "global":
            return "__GLOBAL__"
        elif self.subspace_scope == "layer":
            return self._layer_key(param_name)
        else:
            return param_name
    
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool, **kwargs: Any) -> Dict[str, Any]:
        """
        Run FastFood regularization experiment on each task model.
        
        Args:
            modelpool: Model pool containing task models
            
        Returns:
            Dictionary with original and projected state dicts for each task
        """
        modelpool = to_modelpool(modelpool)
        
        log.info("=" * 80)
        log.info("FastFood Regularization Experiment")
        log.info("=" * 80)
        log.info(f"Configuration:")
        log.info(f"  proj_ratio: {self.proj_ratio}")
        log.info(f"  subspace_scope: {self.subspace_scope}")
        log.info(f"  use_G: {self.use_G}")
        log.info(f"  exclude_patterns: {self.exclude_parameters}")
        log.info("=" * 80)
        
        # Load base model as template
        with self.profile("loading base model"):
            base_model = modelpool.load_model("_pretrained_")
            log.info(f"Base model loaded: {type(base_model)}")
        
        # Get task names
        task_names = list(modelpool.model_names)
        log.info(f"Found {len(task_names)} task models: {task_names}")
        
        results = {}
        
        # Process each task model
        for task_name in task_names:
            log.info(f"\n{'='*60}")
            log.info(f"Processing task: {task_name}")
            log.info(f"{'='*60}")
            
            with self.profile(f"loading {task_name}"):
                task_model = modelpool.load_model(task_name)
                task_sd = task_model.state_dict()
            
            # Project the model
            with self.profile(f"projecting {task_name}"):
                projected_sd = self.project_model(task_sd)
            
            # Create projected model by loading state dict
            with self.profile(f"creating projected model {task_name}"):
                if isinstance(task_model, nn.Module):
                    # Clone the task model structure and load projected weights
                    from copy import deepcopy
                    projected_model = deepcopy(task_model)
                    projected_model.load_state_dict(projected_sd, strict=False)
                elif isinstance(base_model, LazyStateDict):
                    projected_model = deepcopy(base_model.meta_module)
                    projected_model = projected_model.to_empty(device=base_model._device)
                    projected_model.load_state_dict(projected_sd, strict=False)
                else:
                    log.warning(f"Unknown model type: {type(base_model)}, using state dict directly")
                    projected_model = projected_sd
            
            results[task_name] = {
                "original_model": task_model,
                "projected_model": projected_model,
                "task_name": task_name
            }
            
            log.info(f"✓ Task {task_name} processed successfully")
        
        log.info(f"\n{'='*80}")
        log.info(f"All {len(task_names)} task models processed")
        log.info(f"{'='*80}")
        
        self.print_profile_summary()
        
        # Return the first projected model (for evaluation compatibility)
        # In practice, the taskpool will evaluate each task separately
        if results:
            first_task = list(results.keys())[0]
            return results[first_task]["projected_model"]
        
        return base_model
