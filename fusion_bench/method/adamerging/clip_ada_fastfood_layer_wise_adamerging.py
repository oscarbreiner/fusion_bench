"""
CLIP AdaFastFood Layer-wise Merging Algorithm

CLIP-specific implementation of the hybrid AdaMerging + FastFood approach.
Handles CLIP vision models with zero-shot classification heads.

Example Usage:

```bash
fusion_bench \
    method=adamerging/ada_fastfood_clip \
        method.proj_init_strategy=conservative \
        method.proj_init_value=0.3 \
        method.proj_lr=1e-3 \
        method.ada_lr=1e-3 \
        method.max_steps=1000 \
        method.save_merging_weights=ada_fastfood_weights.pt \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=ada_fastfood_merging
```
"""

import functools
import logging
from typing import Iterator

from torch.utils.data import DataLoader

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.utils.data import InfiniteDataLoader

from .ada_fastfood_layer_wise_adamerging import AdaFastFoodLayerWiseMergingAlgorithm

log = logging.getLogger(__name__)


class CLIPAdaFastFoodLayerWiseMergingAlgorithm(
    CLIPClassificationMixin,
    AdaFastFoodLayerWiseMergingAlgorithm,
):
    """
    CLIP-specific AdaFastFood merging algorithm.
    
    Combines:
    - CLIP zero-shot classification setup
    - Per-layer learnable subspace projections (FastFood)
    - Per-task per-layer AdaMerging coefficients
    - Test-time adaptation via entropy loss
    """
    
    def on_test_time_adaptation_start(self):
        """
        Setup CLIP-specific components before test-time adaptation.
        This includes loading the CLIP processor and constructing zero-shot classification heads.
        """
        log.info("Setting up CLIP zero-shot classification heads for AdaFastFood")
        self.setup_zero_shot_classification_head()
        log.info(f"Zero-shot heads configured for {len(self.modelpool.model_names)} tasks")

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str) -> Iterator:
        """
        Get cached shuffled test data loader for CLIP dataset.
        
        Args:
            task: Name of the task/dataset
            
        Returns:
            Iterator over infinite shuffled DataLoader
        """
        return super().get_shuffled_test_loader_iter(
            task,
            batch_size=self.config.get("batch_size", 16),
            num_workers=self.config.get("num_workers", 8),
        )