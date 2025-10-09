"""
CLIP-specific Learnable Fastfood Merging.

This module provides CLIP-specific implementation of learnable Fastfood merging,
including zero-shot classification head setup and logits computation.

Example usage:
```bash
fusion_bench \\
    method=fastfood_merging/learnable_clip \\
        method.init_proj_ratio=0.1 \\
        method.lr=0.01 \\
        method.max_steps=500 \\
    modelpool=clip-vit-base-patch32_TA8 \\
    taskpool=clip-vit-classification_TA8
```
"""

import functools
import logging
import os
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from fusion_bench.dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import timeit_context
from fusion_bench.utils.data import InfiniteDataLoader

from .learnable_fastfood_merging import LearnableFastfoodMergingAlgorithm

log = logging.getLogger(__name__)


class CLIPLearnableFastfoodMergingAlgorithm(
    CLIPClassificationMixin,
    LearnableFastfoodMergingAlgorithm,
):
    """
    CLIP-specific learnable Fastfood merging with test-time adaptation.
    
    This class extends LearnableFastfoodMergingAlgorithm with CLIP-specific
    functionality for loading datasets, constructing zero-shot classification
    heads, and computing logits.
    
    The projection ratios for each layer are learned by minimizing entropy loss
    on CLIP classification tasks.
    """
    
    _clip_processor: CLIPProcessor = None
    zeroshot_weights = {}
    
    def __init__(self, algorithm_config):
        """
        Initialize from config dict (compatibility with AlgorithmFactory).
        
        Args:
            algorithm_config: Either a DictConfig or dict with algorithm parameters
        """
        from omegaconf import OmegaConf
        
        # Convert DictConfig to plain dict/primitives
        if hasattr(algorithm_config, 'items'):
            # Convert to plain Python dict to avoid DictConfig issues
            config_dict = OmegaConf.to_container(algorithm_config, resolve=True)
            # Remove fields not needed by parent __init__
            config_dict.pop('name', None)
            config_dict.pop('fast_dev_run', None)
            super().__init__(**config_dict)
        else:
            # Fallback for direct instantiation
            super().__init__()
    
    def on_test_time_adaptation_start(self):
        """
        Setup for test-time adaptation on CLIP models.
        
        This method:
        1. Loads the CLIP processor
        2. Constructs zero-shot classification heads for each task
        3. Prepares visual projection and logit scale parameters
        """
        log.info("Setting up CLIP zero-shot classification")
        
        # Get pretrained model path
        clip_model_config = self.modelpool.get_model_config("_pretrained_")
        if isinstance(clip_model_config, str):
            pretrained_path = clip_model_config
        else:
            pretrained_path = (
                clip_model_config.pretrained_model_name_or_path
                if hasattr(clip_model_config, "pretrained_model_name_or_path")
                else clip_model_config.path
            )
        
        with timeit_context("Loading CLIP processor and pretrained model"):
            self._clip_processor = CLIPProcessor.from_pretrained(pretrained_path)
            clip_model: CLIPModel = CLIPModel.from_pretrained(pretrained_path)
            
            # Setup classifier for zero-shot weights
            clip_classifier = HFCLIPClassifier(clip_model, self._clip_processor)
            self.visual_projection = clip_model.visual_projection.requires_grad_(False)
            self.logit_scale_exp = clip_model.logit_scale.exp()
            
            # Move to device
            self.visual_projection = self.visual_projection.to(self.device)
            self.logit_scale_exp = self.logit_scale_exp.to(self.device)
        
        # Construct zero-shot classification heads for each task
        cache_dir = getattr(self.config, "cache_dir", "./cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        for task in self.modelpool.model_names:
            cache_file = os.path.join(
                cache_dir,
                f"{os.path.basename(pretrained_path)}_{task}_zeroshot_weights.pt",
            )
            
            if os.path.exists(cache_file):
                log.info(f"Loading cached zero-shot weights for task: {task}")
                zeroshot_weights = torch.load(cache_file, map_location="cpu")
            else:
                log.info(f"Constructing zero-shot classification head for task: {task}")
                classnames, templates = get_classnames_and_templates(task)
                clip_classifier.set_classification_task(classnames, templates)
                zeroshot_weights = clip_classifier.zeroshot_weights
                
                log.info(f"Saving zero-shot weights to {cache_file}")
                torch.save(zeroshot_weights, cache_file)
            
            self.zeroshot_weights[task] = zeroshot_weights.to(self.device)
        
        log.info(f"Prepared zero-shot heads for {len(self.zeroshot_weights)} tasks")
    
    @functools.cache
    def get_test_dataset(self, task: str) -> CLIPDataset:
        """
        Load the test dataset for a task (cached).
        
        Args:
            task: Task name
            
        Returns:
            CLIPDataset for the task
        """
        log.info(f"Loading test dataset: {task}")
        dataset = self.modelpool.load_test_dataset(task)
        return CLIPDataset(dataset, self._clip_processor)
    
    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str) -> Iterator:
        """
        Get an iterator over shuffled test data for a task.
        
        Args:
            task: Task name
            
        Returns:
            Iterator yielding batches of test data
        """
        loader = DataLoader(
            self.get_test_dataset(task),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return iter(InfiniteDataLoader(loader))
    
    def compute_logits(self, module, batch, task: str) -> Tensor:
        """
        Compute classification logits for CLIP.
        
        This method:
        1. Extracts image embeddings from the merged model
        2. Projects them using the visual projection
        3. Computes cosine similarity with text embeddings
        4. Scales by the learned temperature parameter
        
        Args:
            module: The learnable merged model
            batch: Batch of (images, labels)
            task: Task name
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]
        
        # Move images to the correct device
        images = images.to(self.device)
        
        # Get image embeddings from merged model
        # CLIP models return (image_embeds, pooled_output)
        image_embeds = module(images)[1]
        
        # Apply visual projection
        image_embeds = self.visual_projection(image_embeds)
        
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Compute cosine similarity (logits)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        logits_per_image = logits_per_text.t()
        
        return logits_per_image
