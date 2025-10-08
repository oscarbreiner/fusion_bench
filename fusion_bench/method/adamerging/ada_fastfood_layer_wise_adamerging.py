"""
AdaFastFood Layer-wise Merging Algorithm

Hybrid approach combining AdaMerging with FastFood subspace projections:
1. Each layer learns optimal subspace projection ratio
2. AdaMerging coefficients learned within compressed subspaces
3. Efficient test-time adaptation with entropy loss
"""

import functools
import logging
from abc import abstractmethod
from typing import List, Mapping, TypeVar, Union, cast

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.utils.type import TorchModelType

from fusion_bench.models.wrappers.ada_fastfood_fusion import AdaFastFoodMergedModel
from .entropy_loss import entropy_loss
from .utils import get_memory_usage

log = logging.getLogger(__name__)


class AdaFastFoodLayerWiseMergingAlgorithm(
    LightningFabricMixin,
    SimpleProfilerMixin, 
    ModelFusionAlgorithm,
):
    """
    Hybrid AdaMerging + FastFood Algorithm
    
    Combines the benefits of:
    - AdaMerging: Adaptive per-layer task coefficients learned via entropy loss
    - FastFood: Efficient subspace projections for computational efficiency
    
    Key innovation: Each layer learns its own optimal subspace projection ratio,
    while AdaMerging coefficients are learned within the compressed subspaces.
    """
    
    def __init__(self, algorithm_config: DictConfig):
        """Initialize the hybrid algorithm with configuration"""
        super().__init__(algorithm_config)
        
    @torch.no_grad()
    def construct_ada_fastfood_merged_model(
        self, modelpool: "ModelPool"
    ) -> AdaFastFoodMergedModel:
        """
        Construct hybrid AdaFastFood merged model from model pool.
        
        Args:
            modelpool: Contains pretrained and fine-tuned models
            
        Returns:
            AdaFastFoodMergedModel: Hybrid model with learnable projection and AdaMerging weights
        """
        pretrained_model = modelpool.load_model("_pretrained_")
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]
        
        # Create hybrid model with learnable parameters
        module = AdaFastFoodMergedModel(
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            proj_init_strategy=self.config.get("proj_init_strategy", "conservative"),
            proj_init_value=self.config.get("proj_init_value", 0.3),
            ada_init_value=self.config.get("ada_init_value", None),
            use_G=self.config.get("use_G", False),
            clamp_weights=self.config.get("clamp_weights", True),
            clamp_proj=self.config.get("clamp_proj", True),
            tie_weights=self.config.get("tie_weights", True),
            strict=self.config.get("strict", False),
            device=self.config.get("device", "cuda"),
        )
        
        print(f"AdaFastFood model created:")
        print(f"  - Projection params shape: {module.proj_params.shape}")
        print(f"  - AdaMerging weights shape: {module.ada_weights.shape}")
        print(f"  - Total learnable params: {module.proj_params.numel() + module.ada_weights.numel()}")
        
        return module
    
    @rank_zero_only 
    def save_merging_weights(self, file_path: str, module: AdaFastFoodMergedModel):
        """Save both projection ratios and AdaMerging weights"""
        import os
        
        if self.fabric.is_global_zero and self.config.get("save_merging_weights", False):
            if isinstance(file_path, str) and not file_path.startswith(("/", ".")):
                save_path = os.path.join(self.log_dir, file_path)
            else:
                save_path = file_path
                
            log.info(f"Saving AdaFastFood weights to {save_path}")
            
            # Save both projection parameters and AdaMerging weights
            weights_dict = {
                "proj_params": module.proj_params.detach().cpu(),
                "ada_weights": module.ada_weights.detach().cpu(),
                "compression_stats": module.get_compression_stats(),
                "config": {
                    "num_tasks": module.num_tasks,
                    "num_layers": module.num_layers,
                    "param_shapes": module.param_shapes,
                    "param_names": module.param_names,
                }
            }
            
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True) 
            torch.save(weights_dict, save_path)
    
    def run(self, modelpool: ModelPool, **kwargs) -> nn.Module:
        """
        Run the AdaFastFood merging algorithm.
        
        Args:
            modelpool: Model pool with pretrained and fine-tuned models
            
        Returns:
            Merged model after test-time adaptation
        """
        log.info("Fusing models using AdaFastFood (AdaMerging + FastFood subspaces)")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)
        
        with self.profile("construct AdaFastFood model"):
            module = self.construct_ada_fastfood_merged_model(modelpool)
            
        # Skip test-time adaptation if weights are provided
        if self.config.get("weights", None) is not None:
            log.info("Loading pre-trained weights, skipping test-time adaptation")
            # TODO: Implement weight loading
            return module.merge_and_unload()
        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
            
            # Save learned weights
            if self.config.get("save_merging_weights", False):
                self.save_merging_weights(
                    self.config.save_merging_weights, module
                )
                
            # Log final compression statistics
            stats = module.get_compression_stats()
            log.info(f"Final compression stats: {stats}")
            
            return module.merge_and_unload()
    
    def on_test_time_adaptation_start(self):
        """Setup before test-time adaptation (e.g., CLIP heads)"""
        pass
    
    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """Get shuffled test data loader for a task (to be implemented by subclass)"""
        pass
    
    @abstractmethod 
    def compute_logits(
        self, module: AdaFastFoodMergedModel, batch: Tensor, task: str
    ) -> Tensor:
        """Compute logits for given batch and task (to be implemented by subclass)"""
        pass
    
    def test_time_adaptation(
        self, module: AdaFastFoodMergedModel
    ) -> AdaFastFoodMergedModel:
        """
        Perform test-time adaptation on the hybrid model.
        
        Jointly optimizes:
        1. Projection ratios per layer (how much to compress each layer)
        2. AdaMerging coefficients per task per layer (within subspaces)
        """
        self.on_test_time_adaptation_start()
        
        # Configure optimizer for both parameter types
        if self.config.get("optimizer", "adam") == "adam":
            optimizer = torch.optim.Adam([
                {
                    "params": [module.proj_params],
                    "lr": self.config.get("proj_lr", self.config.get("lr", 1e-3)),
                    "name": "projection_ratios"
                },
                {
                    "params": [module.ada_weights], 
                    "lr": self.config.get("ada_lr", self.config.get("lr", 1e-3)),
                    "name": "adamerging_weights"
                }
            ])
            print(f"Optimizer configured with projection_lr={self.config.get('proj_lr', self.config.get('lr', 1e-3))}, "
                  f"ada_lr={self.config.get('ada_lr', self.config.get('lr', 1e-3))}")
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.get('optimizer')}")
        
        module, optimizer = self.fabric.setup(module, optimizer)
        
        # Training loop
        module.train()
        module.merge_weights()  # Initial merge
        
        num_steps = self.config.get("max_steps", 1000)
        if self.is_debug_mode:
            num_steps = 1
            
        for step_idx in (
            pbar := tqdm(
                range(num_steps),
                ("[DEBUG MODE] " if self.is_debug_mode else "") + "AdaFastFood Test-time Adaptation",
                dynamic_ncols=True,
            )
        ):
            total_loss = 0.0
            
            # Iterate through all tasks
            for task_idx, task in enumerate(self.modelpool.model_names):
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    
                with self.profile("forward pass"):
                    if batch is None or len(batch) == 0:
                        log.warning(f"Empty batch for task {task}, skipping")
                        continue
                        
                    logits = self.compute_logits(module, batch[0], task)
                    loss = entropy_loss(logits)
                    total_loss += loss.item()
                    
                with self.profile("backward pass"):
                    self.fabric.backward(loss, retain_graph=(task_idx < len(self.modelpool.model_names) - 1))
            
            # Optimizer step and parameter updates
            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()
                
            with self.profile("merging weights"):
                module.merge_weights()  # Recompute merged weights with new parameters
                
            # Logging and progress tracking
            avg_loss = total_loss / len(self.modelpool.model_names)
            
            # Get current compression statistics
            compression_stats = module.get_compression_stats() 
            
            metrics = {
                "train/loss": avg_loss,
                "train/proj_params_mean": module.proj_params.mean().item(),
                "train/proj_params_std": module.proj_params.std().item(),
                "train/proj_params_min": module.proj_params.min().item(),
                "train/proj_params_max": module.proj_params.max().item(),
                "train/ada_weights_mean": module.ada_weights.mean().item(), 
                "train/ada_weights_std": module.ada_weights.std().item(),
                "train/compression_ratio": compression_stats["overall_compression_ratio"],
                "train/memory_savings": compression_stats["memory_savings"],
            }
            
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "compression": f"{compression_stats['overall_compression_ratio']:.3f}",
                "proj_mean": f"{module.proj_params.mean():.3f}"
            })
        
        log.info(get_memory_usage("After AdaFastFood adaptation, GPU memory usage:"))
        self.print_profile_summary()
        
        # Final compression analysis
        final_stats = module.get_compression_stats()
        log.info("Final AdaFastFood Statistics:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                log.info(f"  {key}: {value:.4f}")
        
        return module