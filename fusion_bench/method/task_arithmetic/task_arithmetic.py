"""
This script contains the general implementation of the Task Arithmetic method.

http://arxiv.org/abs/2212.04089
"""

import logging
from copy import deepcopy
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Dict,
    List,
    Mapping,
    Optional,
    TypeVar,
    Union,
)

import torch
from torch import nn

from fusion_bench import LazyStateDict
from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType, TorchModelType

# Import post-processing utilities
try:
    from fusion_bench.method.fastfood_merging.fastfood_utils import (
        subspace_boosting,
    )
except ImportError:
    # Fallback if fastfood_utils is not available
    def subspace_boosting(param, beta, eps=1e-12):
        raise NotImplementedError("Subspace boosting requires fastfood_utils module")

if TYPE_CHECKING:
    from transformers import PreTrainedModel
log = logging.getLogger(__name__)


@torch.no_grad()
def task_arithmetic_merge(
    pretrained_model: TorchModelType,
    finetuned_models: List[TorchModelType],
    scaling_factor: float,
    inplace: bool = True,
) -> TorchModelType:
    """
    Merges the task vectors from multiple fine-tuned models into a single pre-trained model.

    Args:
        pretrained_model (nn.Module): The pre-trained model to which the task vectors will be added.
        finetuned_models (List[nn.Module]): A list of fine-tuned models from which task vectors will be calculated.
        scaling_factor (float): A factor by which the task vectors will be scaled before merging.
        inplace (bool, optional): If True, the pre-trained model will be modified in place.
                                  If False, a copy of the pre-trained model will be modified. Defaults to True.

    Returns:
        nn.Module: The pre-trained model with the merged task vectors.
    """
    if not inplace:
        pretrained_model = deepcopy(pretrained_model)
    task_vector: Optional[StateDictType] = None
    # Calculate the total task vector
    for model in finetuned_models:
        if task_vector is None:
            # calculate the task vector for the first model
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
            )
        else:
            # calculate the task vector for the remaining models
            task_vector = state_dict_add(
                task_vector,
                state_dict_sub(
                    model.state_dict(keep_vars=True),
                    pretrained_model.state_dict(keep_vars=True),
                ),
            )
    # scale the task vector
    task_vector = state_dict_mul(task_vector, scaling_factor)
    # add the task vector to the pretrained model
    state_dict = state_dict_add(
        pretrained_model.state_dict(keep_vars=True), task_vector
    )
    pretrained_model.load_state_dict(state_dict)
    return pretrained_model


@auto_register_config
class TaskArithmeticAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    """
    Task Arithmetic Algorithm for model fusion.

    This class implements the Task Arithmetic method for fusing models. It inherits from
    BaseModelFusionAlgorithm and SimpleProfilerMixin to provide the necessary functionality
    for model fusion and profiling.

    Attributes:
        scaling_factor (int): The factor by which the task vectors will be scaled before merging.
        use_subspace_boosting (bool): Whether to apply Subspace Boosting to restore rank.
        subspace_boosting_beta (float): Cumulative energy threshold for boosting (0.0-1.0).
        use_lines (bool): Whether to apply LiNeS layer-wise scaling.
        lines_num_blocks (int | None): Number of transformer blocks (auto-detected if None).
        lines_alpha (float | None): Base scaling factor (auto-computed if None and auto_alpha=True).
        lines_beta (float): Maximum scaling factor for deepest layers.
        lines_auto_alpha (bool): Whether to auto-compute alpha from norm ratios.
    """

    def __init__(
        self,
        scaling_factor: int,
        use_subspace_boosting: bool = False,
        subspace_boosting_beta: float = 0.01,
        use_lines: bool = False,
        lines_num_blocks: int | None = None,
        lines_alpha: float | None = None,
        lines_beta: float = 1.0,
        lines_auto_alpha: bool = True,
        **kwargs
    ):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor and optional post-processing.

        Args:
            scaling_factor (int): The factor by which the task vectors will be scaled before merging.
            use_subspace_boosting (bool): Whether to apply Subspace Boosting.
            subspace_boosting_beta (float): Beta parameter for subspace boosting (0.0-1.0).
            use_lines (bool): Whether to apply LiNeS layer-wise scaling.
            lines_num_blocks (int | None): Number of transformer blocks (auto-detected if None).
            lines_alpha (float | None): Base scaling factor (auto-computed if None).
            lines_beta (float): Maximum scaling factor for deepest layers.
            lines_auto_alpha (bool): Whether to auto-compute alpha from norm ratios.
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
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> nn.Module:
        """
        Runs the Task Arithmetic Algorithm to fuse models in the given model pool.

        Args:
            modelpool (Union[BaseModelPool, Dict[str, nn.Module]]): The pool of models to fuse.

        Returns:
            nn.Module: The pre-trained model with the merged task vectors.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info("Fusing models using task arithmetic.")
        task_vector = None
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        # Calculate the total task vector
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            with self.profile("merge weights"):
                if task_vector is None:
                    task_vector = state_dict_sub(
                        model.state_dict(),
                        pretrained_model.state_dict(),
                    )
                else:
                    task_vector = state_dict_add(
                        task_vector,
                        state_dict_sub(
                            model.state_dict(),
                            pretrained_model.state_dict(),
                        ),
                    )
        with self.profile("merge weights"):
            # Post-processing pipeline (optional) - BEFORE scaling
            if self.use_subspace_boosting or self.use_lines:
                with self.profile("post-processing"):
                    task_vector = self._apply_post_processing(
                        task_vector=task_vector,
                        num_tasks=len(modelpool.model_names),
                        pretrained_state=pretrained_model.state_dict(),
                    )
            
            # scale the task vector (after post-processing)
            task_vector = state_dict_mul(task_vector, self.config.scaling_factor)
            # add the task vector to the pretrained model
            state_dict = state_dict_add(pretrained_model.state_dict(), task_vector)
        
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
            
            # For task arithmetic, norm_summed_tvs = num_tasks * norm_merged_tv
            # (since we're just summing task vectors)
            norm_summed_tvs = norm_merged_tv * num_tasks
            
            log.info(f"[LiNeS] Merged TV L1 norm: {norm_merged_tv:.6e}")
            log.info(f"[LiNeS] Summed TVs L1 norm: {norm_summed_tvs:.6e}")
            
            processed_vector, computed_alpha = self._apply_lines_scaling(
                processed_vector,
                num_tasks=num_tasks,
                norm_summed_tvs=norm_summed_tvs,
                norm_merged_tv=norm_merged_tv,
            )
        
        return processed_vector
