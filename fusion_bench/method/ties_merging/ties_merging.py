R"""
Overview of Ties-Merging:

1. Trim: For each task t, we trim the redundant parameters from the task vector $\tau_t$ to create $\hat{\tau}_t$ by keeping the top-k% values according to their magnitude and trimming the bottom $(100 - k)\%$ of the redundant parameters by resetting them to 0. This can be decomposed further as $\hat{\tau}_t = \hat{\gamma}_t \odot \hat{\mu}_t$.

2. Elect: Next, we create an aggregate elected sign vector $\gamma_m$ for the merged model that resolves the disagreements in the sign for each parameter p across different models. To create the elected sign vector, we choose the sign with the highest total magnitude across all relevant models. For each parameter $p \in \{1, 2, \ldots, d\}$, we separate the values $\{\hat{\tau}_t^p\}_{t=1}^n$ based on their sign $(+1$ or $-1)$ and take their sum to calculate the total mass (i.e., total magnitude) in the positive and the negative direction. We then assign $\gamma_m^p$ as the sign with greater total movement. This can be efficiently computed using $\gamma_m^p = \text{sgn}(\sum_{t=1}^n \hat{\tau}_t^p)$.

3. Disjoint Merge: Then, for each parameter p, we compute a disjoint mean by only keeping the parameter values from the models whose signs are the same as the aggregated elected sign and calculate their mean. Formally, let $A_p = \{t \in [n] \mid \hat{\gamma}_t^p = \gamma_m^p\}$, then $\tau_m^p = \frac{1}{|A_p|}\sum_{t\in A_p} \hat{\tau}_t^p$. Note that the disjoint mean always ignores the zero values.
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Literal, Mapping, Union  # noqa: F401

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from fusion_bench import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType

# Import post-processing utilities
try:
    from fusion_bench.method.fastfood_merging.fastfood_utils import (
        subspace_boosting,
    )
except ImportError:
    # Fallback if fastfood_utils is not available
    def subspace_boosting(param, beta, eps=1e-12):
        raise NotImplementedError("Subspace boosting requires fastfood_utils module")

from .ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict

log = logging.getLogger(__name__)


@auto_register_config
class TiesMergingAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        scaling_factor: float,
        threshold: float,
        remove_keys: List[str],
        merge_func: Literal["sum", "mean", "max"],
        use_subspace_boosting: bool = False,
        subspace_boosting_beta: float = 0.01,
        use_lines: bool = False,
        lines_num_blocks: int | None = None,
        lines_alpha: float | None = None,
        lines_beta: float = 1.0,
        lines_auto_alpha: bool = True,
        **kwargs: Any,
    ):
        """
        TiesMergingAlgorithm is a class for fusing multiple models using the TIES merging technique.

        Initialize the TiesMergingAlgorithm with the given parameters.

        Args:
            scaling_factor (float): The scaling factor to apply to the merged task vector.
            threshold (float): The threshold for resetting values in the task vector.
            remove_keys (List[str]): List of keys to remove from the state dictionary.
            merge_func (Literal["sum", "mean", "max"]): The merge function to use for disjoint merging.
            use_subspace_boosting (bool): Whether to apply Subspace Boosting.
            subspace_boosting_beta (float): Beta parameter for subspace boosting (0.0-1.0).
            use_lines (bool): Whether to apply LiNeS layer-wise scaling.
            lines_num_blocks (int | None): Number of transformer blocks (auto-detected if None).
            lines_alpha (float | None): Base scaling factor (auto-computed if None).
            lines_beta (float): Maximum scaling factor for deepest layers.
            lines_auto_alpha (bool): Whether to auto-compute alpha from norm ratios.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        self.use_subspace_boosting = use_subspace_boosting
        self.subspace_boosting_beta = subspace_boosting_beta
        self.use_lines = use_lines
        self.lines_num_blocks = lines_num_blocks
        self.lines_alpha = lines_alpha
        self.lines_beta = lines_beta
        self.lines_auto_alpha = lines_auto_alpha

    @torch.no_grad()
    def run(
        self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs: Any
    ) -> nn.Module:
        """
        Run the TIES merging algorithm to fuse models in the model pool.

        Args:
            modelpool (BaseModelPool | Dict[str, nn.Module]): The model pool containing the models to fuse.

        Returns:
            nn.Module: The fused model.
        """
        log.info("Fusing models using ties merging.")
        modelpool = to_modelpool(modelpool)
        remove_keys = self.config.get("remove_keys", [])
        merge_func = self.config.get("merge_func", "sum")
        scaling_factor = self.scaling_factor
        threshold = self.threshold

        with self.profile("loading models"):
            # Load the pretrained model
            pretrained_model = modelpool.load_model("_pretrained_")

            # Load the state dicts of the models
            ft_checks: List[StateDictType] = [
                modelpool.load_model(model_name).state_dict(keep_vars=True)
                for model_name in modelpool.model_names
            ]
            ptm_check: StateDictType = pretrained_model.state_dict(keep_vars=True)

        with self.profile("merging models"):
            # Compute the task vectors
            flat_ft: Tensor = torch.vstack(
                [state_dict_to_vector(check, remove_keys) for check in ft_checks]
            )
            flat_ptm: Tensor = state_dict_to_vector(ptm_check, remove_keys)
            tv_flat_checks = flat_ft - flat_ptm

            # Perform TIES Merging
            merged_tv = ties_merging(
                tv_flat_checks,
                reset_thresh=threshold,
                merge_func=merge_func,
            )
            
            # Convert back to state dict for post-processing
            merged_tv_dict = vector_to_state_dict(
                merged_tv, ptm_check, remove_keys=remove_keys
            )
            
            # Post-processing pipeline (optional) - BEFORE scaling
            if self.use_subspace_boosting or self.use_lines:
                with self.profile("post-processing"):
                    merged_tv_dict = self._apply_post_processing(
                        task_vector=merged_tv_dict,
                        num_tasks=len(ft_checks),
                        pretrained_state=ptm_check,
                    )
            
            # Convert back to vector and apply scaling (after post-processing)
            merged_tv = state_dict_to_vector(merged_tv_dict, remove_keys)
            merged_check = flat_ptm + scaling_factor * merged_tv
            state_dict = vector_to_state_dict(
                merged_check, ptm_check, remove_keys=remove_keys
            )
        self.print_profile_summary()

        # apply state dict to model
        if isinstance(pretrained_model, nn.Module):
            model = pretrained_model
            model.load_state_dict(state_dict)
        elif isinstance(pretrained_model, LazyStateDict):
            model = deepcopy(pretrained_model.meta_module)
            model = model.to_empty(device=pretrained_model._device)
            result = model.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys in state dict: {result.unexpected_keys}"
                )
            if result.missing_keys:
                log.warning(f"Missing keys in state dict: {result.missing_keys}")
        else:
            raise TypeError(f"Unsupported model type: {type(pretrained_model)}")
        return model

    @staticmethod
    def _is_linear_layer(key: str) -> bool:
        """
        Check if a parameter belongs to a linear layer (attention or MLP).
        
        Args:
            key: Parameter name from state dict
            
        Returns:
            True if the parameter is from a linear layer
        """
        linear_patterns = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention projections
            "qkv", "out_proj",  # alternative attention naming
            "fc1", "fc2", "mlp",  # MLP layers
            "c_fc", "c_proj",  # GPT-style naming
        ]
        return any(pattern in key for pattern in linear_patterns)

    def _apply_lines_scaling(
        self,
        task_vector: StateDictType,
        num_tasks: int,
        norm_summed_tvs: float | None = None,
        norm_merged_tv: float | None = None,
    ) -> tuple[StateDictType, float]:
        """
        Apply LiNeS (Layer Scaling) to the task vector.
        
        Args:
            task_vector: Task vector to scale
            num_tasks: Number of tasks being merged
            norm_summed_tvs: L1 norm of summed task vectors (for auto alpha)
            norm_merged_tv: L1 norm of merged task vector (for auto alpha)
            
        Returns:
            Tuple of (scaled task vector, computed alpha value)
        """
        if not self.use_lines:
            return task_vector, None
        
        import copy
        scaled_vector = copy.deepcopy(task_vector)
        
        # Determine num_blocks if not specified
        if self.lines_num_blocks is None:
            has_24_layers = any(f".23." in k for k in task_vector.keys())
            num_blocks = 24 if has_24_layers else 12
            log.info(f"[LiNeS] Auto-detected num_blocks={num_blocks}")
        else:
            num_blocks = self.lines_num_blocks
        
        # Determine alpha
        if self.lines_auto_alpha and norm_summed_tvs is not None and norm_merged_tv is not None:
            alpha = (norm_summed_tvs / (norm_merged_tv + 1e-12)) * (1.0 / num_tasks)
            log.info(f"[LiNeS] Auto-computed alpha={alpha:.6f}")
        elif self.lines_alpha is not None:
            alpha = self.lines_alpha
            log.info(f"[LiNeS] Using manual alpha={alpha:.6f}")
        else:
            alpha = self.lines_beta
            log.info(f"[LiNeS] Using default alpha={alpha:.6f}")
        
        beta = self.lines_beta
        
        # Build layer key patterns
        key_blocks = [f".{i}." for i in range(num_blocks)]
        
        # Apply scaling
        for k in scaled_vector.keys():
            found_layer = False
            for layer_idx, block_pattern in enumerate(key_blocks):
                if block_pattern in k:
                    scaling = alpha + beta * (layer_idx / (num_blocks - 1))
                    scaled_vector[k] = scaled_vector[k] * scaling
                    found_layer = True
                    break
            
            if not found_layer:
                scaled_vector[k] = scaled_vector[k] * alpha
        
        log.info(f"[LiNeS] Applied layer scaling: alpha={alpha:.4f}, beta={beta:.4f}")
        return scaled_vector, alpha

    def _apply_post_processing(
        self,
        task_vector: StateDictType,
        num_tasks: int,
        pretrained_state: StateDictType,
    ) -> StateDictType:
        """
        Apply post-processing steps to the merged task vector.
        
        Pipeline:
            1. Subspace Boosting (if enabled)
            2. LiNeS Scaling (if enabled)
        
        Args:
            task_vector: Merged task vector to post-process
            num_tasks: Number of tasks being merged
            pretrained_state: Pretrained model state dict (for norm computation)
            
        Returns:
            Post-processed task vector
        """
        import copy
        processed_vector = copy.deepcopy(task_vector)
        
        # Step 1: Apply Subspace Boosting (if enabled)
        if self.use_subspace_boosting:
            log.info("\n=== Applying Subspace Boosting ===")
            boosted_count = 0
            skipped_count = 0
            
            for k in processed_vector.keys():
                # Only apply to linear layers
                if not self._is_linear_layer(k):
                    skipped_count += 1
                    continue
                
                # Only apply to 2D tensors
                if processed_vector[k].ndim != 2:
                    skipped_count += 1
                    continue
                
                processed_vector[k] = subspace_boosting(
                    processed_vector[k], beta=self.subspace_boosting_beta
                )
                boosted_count += 1
            
            log.info(f"[Subspace Boosting] Beta: {self.subspace_boosting_beta:.4f}")
            log.info(f"[Subspace Boosting] Boosted {boosted_count} parameters")
            log.info(f"[Subspace Boosting] Skipped {skipped_count} parameters")
        
        # Step 2: Apply LiNeS Scaling (if enabled)
        if self.use_lines:
            log.info("\n=== Applying LiNeS (Layer Scaling) ===")
            
            # Compute norm of merged task vector
            norm_merged_tv = sum(
                delta.abs().sum().item() for delta in processed_vector.values()
            )
            
            # Compute norm of summed task vectors (approximate from merged TV)
            # For TIES, this is approximate since we've trimmed and signed
            norm_summed_tvs = norm_merged_tv * num_tasks
            
            log.info(f"[LiNeS] Merged TV L1 norm: {norm_merged_tv:.6e}")
            log.info(f"[LiNeS] Summed TVs L1 norm (approx): {norm_summed_tvs:.6e}")
            
            processed_vector, computed_alpha = self._apply_lines_scaling(
                processed_vector,
                num_tasks=num_tasks,
                norm_summed_tvs=norm_summed_tvs,
                norm_merged_tv=norm_merged_tv,
            )
        
        return processed_vector
