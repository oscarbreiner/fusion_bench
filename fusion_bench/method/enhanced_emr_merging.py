"""
EMR-Merging: Enhanced implementation with proper task-specific evaluation

This implementation provides an enhanced version of the EMR algorithm that ensures
proper task-specific evaluation, matching the evaluation protocol from the original paper.

Based on:
- "EMR-Merging: Tuning-Free High-Performance Model Merging" (NeurIPS 2024 Spotlight)
- https://arxiv.org/abs/2405.17461
"""

import logging
from typing import Dict, Any, Union, List, Optional
import torch
import torch.nn as nn

from fusion_bench.method.emr_merging import EMRMergingAlgorithm
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.method.base_algorithm import BaseModelFusionAlgorithm

log = logging.getLogger(__name__)


class TaskSpecificEvaluationWrapper(nn.Module):
    """
    A wrapper that holds multiple models and automatically switches between them
    based on the current task being evaluated. This matches the evaluation protocol
    from the original EMR paper where each task gets its own optimized model.
    """
    
    def __init__(self, task_models: Dict[str, nn.Module]):
        """
        Initialize the wrapper with task-specific models.
        
        Args:
            task_models: Dictionary mapping task names to their corresponding models
        """
        super().__init__()
        self.task_models = nn.ModuleDict(task_models)
        self._current_task = None
        
    def set_task(self, task_name: str):
        """Set the current task for evaluation."""
        if task_name not in self.task_models:
            available_tasks = list(self.task_models.keys())
            log.warning(f"Task '{task_name}' not found. Available tasks: {available_tasks}")
            # Fallback to first available task
            task_name = available_tasks[0] if available_tasks else None
        
        self._current_task = task_name
        
    def forward(self, *args, **kwargs):
        """Forward pass using the current task's model."""
        if self._current_task is None:
            # If no task is set, use the first available model
            task_name = list(self.task_models.keys())[0]
            log.warning(f"No task set, defaulting to '{task_name}'")
            self._current_task = task_name
            
        return self.task_models[self._current_task](*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the current task's model."""
        if name in ['task_models', '_current_task'] or name.startswith('_'):
            return super().__getattr__(name)
        
        if self._current_task is None and hasattr(self, 'task_models') and self.task_models:
            self._current_task = list(self.task_models.keys())[0]
            
        if self._current_task and self._current_task in self.task_models:
            return getattr(self.task_models[self._current_task], name)
        
        return super().__getattr__(name)


class EMRMergingAlgorithmWithTaskEval(EMRMergingAlgorithm):
    """
    Enhanced EMR algorithm that supports task-specific evaluation.
    
    This version extends the base EMR algorithm to provide proper task-specific
    evaluation capabilities, matching the original paper's evaluation protocol.
    
    For the analysis phase, it uses unified mode for compatibility.
    The task-specific behavior is preserved for the benchmarking phase.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        mode: str = "task_specific",
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize the enhanced EMR algorithm.
        
        Args:
            normalize: Whether to normalize task vectors before merging
            mode: Merging mode - 'unified', 'separate', or 'task_specific'
                - 'unified': Returns single merged model (standard fusion)
                - 'separate': Returns list of task-specific models  
                - 'task_specific': Returns wrapper that automatically switches models per task
            weights: Optional weights for combining task vectors
        """
        super().__init__(normalize=normalize, mode=mode, weights=weights, **kwargs)
        
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool) -> Union[nn.Module, List[nn.Module], TaskSpecificEvaluationWrapper]:
        """
        Run the enhanced EMR merging algorithm.
        
        Args:
            modelpool: Pool of models to merge
            
        Returns:
            Merged model(s) based on the specified mode:
            - 'unified': Single merged model
            - 'separate': List of task-specific models
            - 'task_specific': TaskSpecificEvaluationWrapper for task-specific analysis and evaluation
        """
        log.info(f"Running Enhanced EMR merging with mode: {self.mode}")
        
        if self.mode == "task_specific":
            # For task_specific mode, get separate models and wrap them
            original_mode = self.mode
            self.mode = "separate"
            try:
                base_result = super().run(modelpool)
                self.mode = original_mode  # Restore original mode
                
                # Create task-specific evaluation wrapper
                if isinstance(base_result, list):
                    # Get task names by excluding the foundation model key
                    task_names = [name for name in modelpool.model_names if name != "_pretrained_"]
                    
                    if len(base_result) == len(task_names):
                        task_models = {task: model for task, model in zip(task_names, base_result)}
                        log.info(f"Created task-specific wrapper for tasks: {list(task_models.keys())}")
                        return TaskSpecificEvaluationWrapper(task_models)
                    else:
                        log.warning(f"Mismatch between models ({len(base_result)}) and tasks ({len(task_names)})")
                        return base_result[0] if base_result else None
                else:
                    # If single model, wrap it for all tasks
                    task_names = [name for name in modelpool.model_names if name != "_pretrained_"]
                    task_models = {task: base_result for task in task_names}
                    log.info(f"Wrapped single model for tasks: {list(task_models.keys())}")
                    return TaskSpecificEvaluationWrapper(task_models)
            except Exception as e:
                # Restore original mode in case of error
                self.mode = original_mode
                raise e
        else:
            # For other modes, delegate to base class
            return super().run(modelpool)

    @property 
    def method_name(self) -> str:
        """Return the method name for identification."""
        return "emr_merging_enhanced"
    
    def __repr__(self) -> str:
        return f"EMRMergingAlgorithmWithTaskEval(normalize={self.normalize}, mode='{self.mode}')"
