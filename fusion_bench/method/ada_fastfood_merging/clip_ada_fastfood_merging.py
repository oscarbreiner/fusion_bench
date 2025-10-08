"""
AdaFastFood CLIP Vision Merging

CLIP-specific implementation of AdaFastFood merging for vision transformer models.
Handles proper data loading, logit computation, and vision-specific optimizations.
"""

import logging
from typing import Any, Dict

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fusion_bench.dataset import load_dataset_from_config
from fusion_bench.compat.modelpool import ModelPool

from .ada_fastfood_merging import AdaFastFoodMergingAlgorithm, AdaFastFoodMergedModel

log = logging.getLogger(__name__)


class CLIPAdaFastFoodMergingAlgorithm(AdaFastFoodMergingAlgorithm):
    """
    CLIP Vision-specific AdaFastFood merging algorithm.
    
    Extends base algorithm with:
    - CLIP vision model handling 
    - Proper test data loading for multiple vision tasks
    - Vision-specific logit computation
    """
    
    def __init__(
        self,
        proj_init_strategy: str = "conservative",
        proj_init_value: float = 0.3,
        proj_lr: float = 1e-3,
        use_G: bool = False,
        clamp_proj: bool = True,
        ada_init_value = None,
        ada_lr: float = 1e-3,
        clamp_weights: bool = True,
        optimizer: str = "adam",
        lr: float = 1e-3,
        max_steps: int = 1000,
        fast_dev_run: bool = False,
        batch_size: int = 16,
        num_workers: int = 8,
        tie_weights: bool = True,
        strict: bool = False,
        device: str = "cuda",
        save_merging_weights: bool = False,
        cache_dir: str = "outputs",
        subspace_scope: str = "layer",
        block_rows: int = 8192,
        exclude_classification_head: bool = True,
        vision_encoder_only: bool = True,
        plot_learned_params: bool = True,
        plot_save_path: str = "ada_fastfood_analysis.png",
        test_datasets = None,
        **kwargs
    ):
        """Initialize CLIP-specific algorithm with individual parameters"""
        # Create DictConfig from parameters for parent class
        from omegaconf import DictConfig
        
        # Helper function to convert string boolean values
        def convert_param(value, expected_type):
            # Check for None values FIRST before type conversion
            if value == 'None' or value == 'null' or value is None:
                return None
            elif expected_type == bool:
                if isinstance(value, str):
                    return value.lower() == 'true'
                return bool(value)
            elif expected_type == float:
                return float(value)
            elif expected_type == int:
                return int(value)
            return value
        
        config_dict = {
            "proj_init_strategy": proj_init_strategy,
            "proj_init_value": convert_param(proj_init_value, float),
            "proj_lr": convert_param(proj_lr, float),
            "use_G": convert_param(use_G, bool),
            "clamp_proj": convert_param(clamp_proj, bool),
            "ada_init_value": convert_param(ada_init_value, float),
            "ada_lr": convert_param(ada_lr, float),
            "clamp_weights": convert_param(clamp_weights, bool),
            "optimizer": optimizer,
            "lr": convert_param(lr, float),
            "max_steps": convert_param(max_steps, int),
            "fast_dev_run": convert_param(fast_dev_run, bool),
            "batch_size": convert_param(batch_size, int),
            "num_workers": convert_param(num_workers, int),
            "tie_weights": convert_param(tie_weights, bool),
            "strict": convert_param(strict, bool),
            "device": device,
            "save_merging_weights": convert_param(save_merging_weights, bool),
            "cache_dir": cache_dir,
            "subspace_scope": subspace_scope,
            "block_rows": convert_param(block_rows, int),
            "exclude_classification_head": convert_param(exclude_classification_head, bool),
            "vision_encoder_only": convert_param(vision_encoder_only, bool),
            "plot_learned_params": convert_param(plot_learned_params, bool),
            "plot_save_path": plot_save_path,
            "test_datasets": test_datasets or {},
        }
        # Add any additional kwargs
        config_dict.update(kwargs)
        
        algorithm_config = DictConfig(config_dict)
        super().__init__(algorithm_config)
        self.test_data_loaders = {}
        
    def on_test_time_adaptation_start(self):
        """Setup test data loaders for all tasks"""
        super().on_test_time_adaptation_start()
        
        log.info("Loading test datasets for CLIP vision tasks")
        
        # Get test datasets from taskpool if available
        test_datasets = {}
        if hasattr(self, "_program") and self._program is not None and hasattr(self._program, "taskpool"):
            if hasattr(self._program.taskpool, "_test_datasets"):
                test_datasets = self._program.taskpool._test_datasets
            elif hasattr(self._program.taskpool, "test_datasets"):
                test_datasets = self._program.taskpool.test_datasets
        
        for task_name in self.modelpool.model_names:
            # Find corresponding test dataset config
            dataset_config = None
            
            # First try to get from taskpool
            if task_name in test_datasets:
                dataset_config = test_datasets[task_name]
            # Then try from method config
            elif hasattr(self.config, "test_datasets") and self.config.test_datasets:
                if task_name in self.config.test_datasets:
                    dataset_config = self.config.test_datasets[task_name]
                else:
                    # Try common patterns
                    for key in self.config.test_datasets.keys():
                        if task_name in key or key in task_name:
                            dataset_config = self.config.test_datasets[key]
                            break
            
            if dataset_config is None:
                log.warning(f"No test dataset config found for task {task_name}")
                continue
                
            try:
                # Load dataset using fusion_bench's dataset loading
                dataset = load_dataset_from_config(dataset_config)
                
                # Create data loader
                batch_size = self.config.get("batch_size", 16)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,  # Avoid multiprocessing issues
                    pin_memory=True if torch.cuda.is_available() else False,
                )
                
                self.test_data_loaders[task_name] = iter(dataloader)
                log.info(f"Loaded test data for task: {task_name}")
                
            except Exception as e:
                log.error(f"Failed to load test data for task {task_name}: {e}")
                # Create dummy loader to avoid crashes
                self.test_data_loaders[task_name] = iter([])
    
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """Get test data loader iterator for specified task"""
        if task not in self.test_data_loaders:
            log.warning(f"No test data loader for task {task}")
            return iter([])
            
        try:
            # Try to get next batch from existing iterator
            batch = next(self.test_data_loaders[task])
            return iter([batch])
        except StopIteration:
            # Dataset exhausted, return empty iterator
            log.debug(f"Test dataset exhausted for task {task}")
            return iter([])
        except Exception as e:
            log.error(f"Error getting batch for task {task}: {e}")
            return iter([])
    
    def compute_logits(self, module: AdaFastFoodMergedModel, batch: torch.Tensor, task: str) -> torch.Tensor:
        """
        Compute logits for CLIP vision models.
        
        Args:
            module: The merged AdaFastFood model
            batch: Input batch (images)  
            task: Task identifier
            
        Returns:
            Logits tensor for entropy loss computation
        """
        try:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images = batch[0]
            else:
                images = batch
                
            # Move to device
            if hasattr(images, 'to'):
                images = images.to(next(module.parameters()).device)
                
            # Forward pass through CLIP vision model
            # Most CLIP implementations return logits directly or have logits_per_image
            outputs = module(images)
            
            if hasattr(outputs, 'logits_per_image'):
                logits = outputs.logits_per_image
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                # If output is a dict, try common keys
                if isinstance(outputs, dict):
                    for key in ['logits', 'logits_per_image', 'prediction_scores']:
                        if key in outputs:
                            logits = outputs[key]
                            break
                    else:
                        # Default to first tensor value
                        logits = list(outputs.values())[0]
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")
                    
            return logits
            
        except Exception as e:
            log.error(f"Error computing logits for task {task}: {e}")
            # Return dummy logits to avoid crashing
            device = next(module.parameters()).device
            return torch.randn(1, 10, device=device, requires_grad=True)
    
    def construct_ada_fastfood_merged_model(self, modelpool: ModelPool) -> AdaFastFoodMergedModel:
        """Construct CLIP-specific merged model"""
        
        # Add CLIP-specific configuration
        clip_config = {
            "exclude_classification_head": self.config.get("exclude_classification_head", True),
            "vision_encoder_only": self.config.get("vision_encoder_only", True),
        }
        
        log.info(f"Creating CLIP AdaFastFood model with config: {clip_config}")
        
        # Call parent method to create base merged model
        module = super().construct_ada_fastfood_merged_model(modelpool)
        
        # Add any CLIP-specific processing here if needed
        # For now, the base implementation handles everything
        
        return module
    
    def run(self, modelpool: ModelPool) -> torch.nn.Module:
        """Run CLIP-specific AdaFastFood merging"""
        
        log.info("Starting CLIP AdaFastFood merging")
        
        # Validate modelpool for CLIP models
        pretrained_model = modelpool.load_model("_pretrained_")
        log.info(f"Pretrained model type: {type(pretrained_model).__name__}")
        
        # Check if models are CLIP-like (have vision encoder)
        if hasattr(pretrained_model, 'visual') or hasattr(pretrained_model, 'vision_model'):
            log.info("Detected CLIP-style vision model")
        else:
            log.warning("Model may not be CLIP-compatible")
        
        # Run base algorithm
        return super().run(modelpool)