"""
EMR-Merging (Elect-Mask-Rescale): Tuning-Free High-Performance Model Merging

Based on the NeurIPS 2024 Spotlight Paper:
"EMR-Merging: Tuning-Free High-Performance Model Merging"
https://arxiv.org/abs/2405.17461

This method merges multiple fine-tuned models by:
1. Elect: Determining unified direction from mean signs and maximum magnitude
2. Mask: Creating task-specific masks based on direction alignment 
3. Rescale: Adjusting magnitudes to preserve task-specific characteristics
"""

import logging
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from fusion_bench.method.base_algorithm import BaseModelFusionAlgorithm
try:
    from fusion_bench.mixins import SimpleProgressBarMixin
except ImportError:
    # Create a dummy mixin if not available
    class SimpleProgressBarMixin:
        pass

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


class EMRMergingAlgorithm(BaseModelFusionAlgorithm, SimpleProgressBarMixin):
    """
    EMR-Merging: Elect-Mask-Rescale Model Merging Algorithm
    
    This algorithm implements the EMR merging method which consists of three steps:
    1. Elect: Determine unified task vector with direction from element-wise mean
              and magnitude from maximum absolute values
    2. Mask: Create task-specific binary masks based on direction alignment
    3. Rescale: Apply task-specific rescaling factors to preserve magnitudes
    
    The method can operate in two modes:
    - 'unified': Returns a single merged model (average over all task-specific vectors)
    - 'separate': Returns individual task-specific models
    """

    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
        "normalize": "normalize",
        "mode": "mode",
        "weights": "weights",
    }

    def __init__(
        self,
        normalize: bool = True,
        mode: str = "unified", 
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize EMR-Merging algorithm.
        
        Args:
            normalize: Whether to normalize task vectors before merging
            mode: Merging mode - 'unified' for single merged model, 
                  'separate' for task-specific models
            weights: Optional weights for combining task-specific vectors in unified mode
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.normalize = normalize
        self.mode = mode
        self.weights = weights

    def run(self, modelpool: BaseModelPool) -> Union[nn.Module, List[nn.Module]]:
        """
        Run EMR merging on the model pool.
        
        Args:
            modelpool: Pool of models to merge
            
        Returns:
            Merged model(s) - single model if mode='unified', 
            list of models if mode='separate'
        """
        log.info("Starting EMR-Merging algorithm")
        log.info(f"Mode: {self.mode}, Normalize: {self.normalize}")
        
        if len(modelpool.model_names) < 2:
            raise ValueError("EMR merging requires at least 2 models")
            
        # Get pretrained model and fine-tuned models
        pretrained_model = modelpool.load_model('_pretrained_')
        finetuned_models = {
            name: modelpool.load_model(name) 
            for name in modelpool.model_names 
            if name != '_pretrained_'
        }
        
        if len(finetuned_models) == 0:
            raise ValueError("No fine-tuned models found in model pool")
            
        log.info(f"Merging {len(finetuned_models)} models: {list(finetuned_models.keys())}")
        
        # Compute task vectors (fine-tuned - pretrained)
        task_vectors = self._compute_task_vectors(pretrained_model, finetuned_models)
        
        # Apply EMR merging algorithm
        unified_vector, task_masks, rescaling_factors = self._emr_merge(task_vectors)
        
        if self.mode == "unified":
            # Return single merged model
            merged_vector = self._create_unified_vector(
                unified_vector, task_masks, rescaling_factors
            )
            merged_model = self._apply_vector_to_model(merged_vector, pretrained_model)
            log.info("EMR merging completed - unified model created")
            return merged_model
            
        elif self.mode == "separate":
            # Return task-specific models
            merged_models = []
            model_names = list(finetuned_models.keys())
            
            for i, model_name in enumerate(model_names):
                task_vector = self._create_task_specific_vector(
                    unified_vector, task_masks, rescaling_factors, i
                )
                task_model = self._apply_vector_to_model(task_vector, pretrained_model)
                merged_models.append(task_model)
                
            log.info(f"EMR merging completed - {len(merged_models)} task-specific models created")
            return merged_models
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'unified' or 'separate'")

    def _compute_task_vectors(
        self, 
        pretrained_model: nn.Module, 
        finetuned_models: Dict[str, nn.Module]
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute task vectors by subtracting pretrained from fine-tuned models."""
        log.info("Computing task vectors...")
        
        pretrained_state_dict = pretrained_model.state_dict()
        task_vectors = []
        
        for model_name, finetuned_model in finetuned_models.items():
            finetuned_state_dict = finetuned_model.state_dict()
            
            # Compute task vector: finetuned - pretrained  
            task_vector = state_dict_sub(finetuned_state_dict, pretrained_state_dict)
            
            # Optional normalization
            if self.normalize:
                task_vector = self._normalize_vector(task_vector)
                
            task_vectors.append(task_vector)
            log.info(f"Computed task vector for {model_name}")
            
        return task_vectors

    def _normalize_vector(self, vector: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize task vector to unit norm."""
        total_norm = 0.0
        for param in vector.values():
            total_norm += param.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 0:
            return {k: v / total_norm for k, v in vector.items()}
        return vector

    def _emr_merge(
        self, 
        task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], torch.Tensor]:
        """
        Apply EMR merging algorithm: Elect → Mask → Rescale
        
        Returns:
            unified_vector: Unified task vector 
            task_masks: Per-task binary masks for each parameter
            rescaling_factors: Per-task rescaling factors
        """
        log.info("Applying EMR merging algorithm...")
        
        num_tasks = len(task_vectors)
        param_names = list(task_vectors[0].keys())
        
        # Initialize outputs
        unified_vector = {}
        task_masks = {name: [] for name in param_names}
        rescaling_factors = torch.zeros(num_tasks)
        
        for param_name in param_names:
            # Stack parameters from all tasks: [num_tasks, ...]
            stacked_params = torch.stack([tv[param_name] for tv in task_vectors], dim=0)
            
            # Step 1: Elect
            # Direction: sign of element-wise mean
            mean_param = stacked_params.mean(dim=0)
            direction_sign = torch.sign(mean_param)  # {-1, 0, +1}
            
            # Magnitude: maximum absolute value among tasks that agree with direction
            param_magnitudes = torch.abs(stacked_params)
            unified_magnitudes = torch.zeros_like(mean_param)
            
            # For each spatial location, find max magnitude among agreeing tasks
            for task_idx in range(num_tasks):
                task_param = stacked_params[task_idx]
                # Mask for parameters that agree in sign with elected direction
                agreement_mask = (task_param * direction_sign) > 0
                task_magnitudes = param_magnitudes[task_idx]
                
                # Update unified magnitude where this task has larger magnitude
                unified_magnitudes = torch.where(
                    agreement_mask & (task_magnitudes > unified_magnitudes),
                    task_magnitudes,
                    unified_magnitudes
                )
            
            # Unified parameter: direction × magnitude
            unified_param = direction_sign * unified_magnitudes
            unified_vector[param_name] = unified_param
            
            # Step 2: Mask  
            # Create binary masks for each task based on direction alignment
            for task_idx in range(num_tasks):
                task_param = stacked_params[task_idx]
                # Mask: 1 where task parameter agrees in sign with unified parameter
                task_mask = (task_param * unified_param) > 0
                task_masks[param_name].append(task_mask)
                
            # Step 3: Rescale (accumulate magnitudes for rescaling factor computation)
            for task_idx in range(num_tasks):
                task_param = stacked_params[task_idx]
                rescaling_factors[task_idx] += torch.mean(torch.abs(task_param)).item()
        
        # Compute final rescaling factors
        new_rescaling_factors = torch.zeros(num_tasks)
        for task_idx in range(num_tasks):
            for param_name in param_names:
                unified_param = unified_vector[param_name]
                task_mask = task_masks[param_name][task_idx]
                masked_unified = unified_param * task_mask.to(unified_param.dtype)
                new_rescaling_factors[task_idx] += torch.mean(torch.abs(masked_unified)).item()
        
        # Final rescaling factors: original_magnitude / new_magnitude 
        final_rescaling_factors = rescaling_factors / (new_rescaling_factors + 1e-12)
        
        log.info(f"EMR merging completed. Rescaling factors: {final_rescaling_factors.tolist()}")
        return unified_vector, task_masks, final_rescaling_factors

    def _create_unified_vector(
        self,
        unified_vector: Dict[str, torch.Tensor],
        task_masks: Dict[str, List[torch.Tensor]], 
        rescaling_factors: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Create unified merged vector by averaging task-specific modulated vectors."""
        merged_vector = {}
        num_tasks = len(rescaling_factors)
        
        weights = self.weights
        if weights is None:
            weights = [1.0 / num_tasks] * num_tasks
        else:
            # Normalize weights
            weight_sum = sum(weights)
            weights = [w / weight_sum for w in weights]
            
        weights = torch.tensor(weights, dtype=unified_vector[next(iter(unified_vector))].dtype)
        
        for param_name in unified_vector.keys():
            unified_param = unified_vector[param_name]
            
            # Create task-specific modulated vectors
            task_specific_vectors = []
            for task_idx in range(num_tasks):
                task_mask = task_masks[param_name][task_idx]
                rescaling_factor = rescaling_factors[task_idx]
                
                # Apply mask and rescaling
                modulated_vector = unified_param * task_mask.to(unified_param.dtype) * rescaling_factor
                task_specific_vectors.append(modulated_vector)
            
            # Weighted average of task-specific vectors
            stacked_vectors = torch.stack(task_specific_vectors, dim=0)
            merged_vector[param_name] = (weights.view(-1, *([1] * (stacked_vectors.ndim - 1))) * stacked_vectors).sum(dim=0)
            
        return merged_vector

    def _create_task_specific_vector(
        self,
        unified_vector: Dict[str, torch.Tensor],
        task_masks: Dict[str, List[torch.Tensor]],
        rescaling_factors: torch.Tensor,
        task_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Create task-specific vector for a given task index."""
        task_vector = {}
        rescaling_factor = rescaling_factors[task_idx]
        
        for param_name in unified_vector.keys():
            unified_param = unified_vector[param_name]
            task_mask = task_masks[param_name][task_idx]
            
            # Apply task-specific mask and rescaling
            task_vector[param_name] = unified_param * task_mask.to(unified_param.dtype) * rescaling_factor
            
        return task_vector

    def _apply_vector_to_model(
        self, 
        vector: Dict[str, torch.Tensor], 
        base_model: nn.Module
    ) -> nn.Module:
        """Apply task vector to base model to create merged model."""
        import copy
        
        # Create a deep copy of the base model to avoid modifying the original
        merged_model = copy.deepcopy(base_model)
        merged_state_dict = merged_model.state_dict()
        
        # Add task vector
        for param_name, param_delta in vector.items():
            if param_name in merged_state_dict:
                merged_state_dict[param_name] = merged_state_dict[param_name] + param_delta
            else:
                log.warning(f"Parameter {param_name} not found in model state dict")
                
        merged_model.load_state_dict(merged_state_dict)
        
        # Clean up any problematic config attributes
        if hasattr(merged_model, 'config') and hasattr(merged_model.config, 'return_dict'):
            # Remove problematic config attribute if it exists
            if hasattr(merged_model.config, '__dict__') and 'return_dict' in merged_model.config.__dict__:
                delattr(merged_model.config, 'return_dict')
                
        return merged_model


# Configuration mapping for Hydra
EMRMergingAlgorithm._config_mapping = EMRMergingAlgorithm._config_mapping
