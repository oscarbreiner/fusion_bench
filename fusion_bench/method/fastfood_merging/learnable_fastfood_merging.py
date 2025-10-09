"""
Learnable Layer-wise Fastfood Merging Algorithm.

This module implements a test-time adaptation approach where the projection ratio
(subspace dimension) for each layer is learned by minimizing entropy loss on test data.
Unlike standard Fastfood merging which uses fixed projection ratios, this method
optimizes the projection ratios to find the optimal subspace size for each layer.

The merging process itself remains unchanged from the base Fastfood algorithm.
Only the projection ratios are trainable parameters.

Example usage:
    ```python
    from fusion_bench.method.fastfood_merging import LearnableFastfoodMergingAlgorithm
    
    algorithm = LearnableFastfoodMergingAlgorithm(
        init_proj_ratio=0.1,
        lr=0.01,
        max_steps=500,
        optimizer="adam",
    )
    merged_model = algorithm.run(modelpool)
    ```
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.wrappers.learnable_fastfood_fusion import (
    LearnableFastfoodMergedModel,
    get_learnable_projection_ratios,
)
from fusion_bench.utils.type import StateDictType

from .fastfood_utils import (
    EPS,
    create_fastfood_ops,
    zero_aware_aggregate,
    layer_key,
    normalize_weights,
    compute_global_dim,
)

log = logging.getLogger(__name__)


def entropy_loss(logits: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute entropy loss for test-time adaptation.
    
    Lower entropy indicates more confident predictions.
    
    Args:
        logits: Model output logits, shape (batch_size, num_classes)
        eps: Small constant for numerical stability
        
    Returns:
        Scalar entropy loss
    """
    assert logits.dim() == 2, f"Expected 2D logits, got shape {logits.shape}"
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()


@auto_register_config
class LearnableFastfoodMergingAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    """
    Learnable Layer-wise Fastfood Merging with Test-Time Adaptation.
    
    This algorithm performs Fastfood/SRHT-based model merging where the projection
    ratio for each layer is learned during test-time adaptation. The algorithm
    minimizes entropy loss on unlabeled test data to find optimal subspace dimensions.
    
    Key parameters:
        init_proj_ratio: Initial projection ratio for all layers (default: 0.1)
        lr: Learning rate for optimizer (default: 0.01)
        max_steps: Number of optimization steps (default: 500)
        optimizer: Optimizer type, currently only "adam" supported
        clamp_ratios: Whether to clamp projection ratios to valid range
        min_proj_ratio: Minimum allowed projection ratio (default: 0.01)
        max_proj_ratio: Maximum allowed projection ratio (default: 1.0)
        
        # Fastfood merging parameters (fixed during training):
        merge_func: Aggregation function in subspace ("sum", "mean", "ties_sum", etc.)
        use_G: Whether to use Gaussian scaling in Fastfood transform
        block_rows: Block size for processing (memory optimization)
        weights: Task weights for merging (None = uniform)
        scale: Post-merge scaling factor
        
        # Optional: Save learned projection ratios
        save_projection_ratios: Path to save learned ratios (optional)
    """
    
    def __init__(
        self,
        # Learning parameters
        init_proj_ratio: float = 0.1,
        lr: float = 0.01,
        max_steps: int = 500,
        optimizer: str = "adam",
        batch_size: int = 32,
        num_workers: int = 4,
        
        # Projection ratio constraints
        clamp_ratios: bool = True,
        min_proj_ratio: float = 0.01,
        max_proj_ratio: float = 1.0,
        
        # Fastfood merging parameters (fixed)
        merge_func: str = "sum",
        use_G: bool = False,
        device: str = "cuda",
        block_rows: int = 8192,
        weights: List[float] | None = None,
        scale: float = 1.0,
        
        # Analysis and saving
        save_projection_ratios: str | None = None,
        
        # Compatibility parameters (kept for config compatibility)
        merge_where: str = "subspace",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        
        # Learning parameters
        self.init_proj_ratio = float(init_proj_ratio)
        self.lr = float(lr)
        self.max_steps = int(max_steps)
        self.optimizer = str(optimizer)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        
        # Projection ratio constraints
        self.clamp_ratios = bool(clamp_ratios)
        self.min_proj_ratio = float(min_proj_ratio)
        self.max_proj_ratio = float(max_proj_ratio)
        
        # Fastfood parameters
        self.merge_func = str(merge_func).lower()
        self.use_G = bool(use_G)
        self.device = torch.device(device)
        self.block_rows = int(block_rows)
        self.weights = list(weights) if weights is not None else None
        self.scale = float(scale)
        
        # Saving
        self.save_projection_ratios = save_projection_ratios
        
        # Validate
        assert 0.0 < self.init_proj_ratio <= 1.0, "init_proj_ratio must be in (0, 1]"
        assert self.optimizer == "adam", f"Only 'adam' optimizer supported, got {self.optimizer}"
        assert merge_where == "subspace", "Only subspace merging is supported"
        
        log.info(f"Initialized LearnableFastfoodMergingAlgorithm with init_proj_ratio={self.init_proj_ratio}")
    
    def construct_learnable_merged_model(
        self,
        modelpool: BaseModelPool,
    ) -> LearnableFastfoodMergedModel:
        """
        Construct a learnable Fastfood merged model.
        
        This method:
        1. Loads base and donor models
        2. Identifies mergeable parameters and groups them by layer
        3. Creates learnable projection ratios for each layer
        4. Returns a wrapper model with the merge function
        
        Args:
            modelpool: Model pool containing pretrained and finetuned models
            
        Returns:
            LearnableFastfoodMergedModel with learnable projection ratios
        """
        with self.profile("loading models"):
            base_model = modelpool.load_model("_pretrained_")
            donor_names = list(modelpool.model_names)
            
            if len(donor_names) < 2:
                raise ValueError(f"Need ≥2 donors; got {len(donor_names)}")
            
            donors_sd: List[StateDictType] = [
                modelpool.load_model(n).state_dict(keep_vars=True)
                for n in donor_names
            ]
            base_sd: Dict[str, Tensor] = base_model.state_dict(keep_vars=True)
        
        with self.profile("identifying mergeable parameters"):
            # Find eligible tensors
            keys_all = list(base_sd.keys())
            keys_float = [
                k for k in keys_all
                if (k in donors_sd[0])
                and torch.is_floating_point(base_sd[k])
                and base_sd[k].ndim >= 1
                and all((k in d) and torch.is_floating_point(d[k]) and d[k].shape == base_sd[k].shape for d in donors_sd)
            ]
            
            if not keys_float:
                raise RuntimeError("No overlapping float tensors. Nothing to merge.")
            
            log.info(f"Found {len(keys_float)} mergeable parameters out of {len(keys_all)} total")
        
        with self.profile("grouping by layer"):
            # Group parameters by layer
            layer_groups = {}
            for k in keys_float:
                lkey = layer_key(k)
                if lkey not in layer_groups:
                    layer_groups[lkey] = []
                layer_groups[lkey].append(k)
            
            num_layers = len(layer_groups)
            log.info(f"Grouped parameters into {num_layers} layers")
            
            # Create ordered list of layer keys
            layer_keys = sorted(layer_groups.keys())
        
        with self.profile("initializing projection ratios"):
            # Initialize learnable projection ratios
            projection_ratios = get_learnable_projection_ratios(
                num_layers=num_layers,
                init_value=self.init_proj_ratio,
                dtype=torch.float32,
            )
            
            log.info(f"Initialized {num_layers} projection ratios at {self.init_proj_ratio}")
        
        # Store necessary data for merge function
        self._layer_keys = layer_keys
        self._layer_groups = layer_groups
        self._base_sd = base_sd
        self._donors_sd = donors_sd
        self._keys_float = keys_float
        self._donor_names = donor_names
        
        # Normalize weights
        K = len(donor_names)
        if self.weights is None:
            w = [1.0 / K] * K
        else:
            if len(self.weights) != K:
                raise ValueError(f"weights length {len(self.weights)} != num donors {K}")
            s = sum(self.weights) + EPS
            w = [wi / s for wi in self.weights]
        self._normalized_weights = w
        
        # Create merge function closure
        def merge_function(proj_ratios: Tensor) -> StateDictType:
            return self._merge_with_ratios(proj_ratios)
        
        # Create wrapper model
        module = LearnableFastfoodMergedModel(
            projection_ratios=projection_ratios,
            pretrained_model=base_model,
            merge_function=merge_function,
            clamp_ratios=self.clamp_ratios,
            min_ratio=self.min_proj_ratio,
            max_ratio=self.max_proj_ratio,
        )
        
        return module
    
    def _merge_with_ratios(self, proj_ratios: Tensor) -> StateDictType:
        """
        Perform Fastfood merging with given projection ratios.
        
        This is the core merging function that gets called during forward passes.
        It uses the current projection ratios to determine subspace dimensions.
        
        Args:
            proj_ratios: Projection ratio for each layer, shape (num_layers,)
            
        Returns:
            Merged state dictionary
        """
        # Get clamped ratios
        if self.clamp_ratios:
            proj_ratios = torch.clamp(proj_ratios, self.min_proj_ratio, self.max_proj_ratio)
        
        # Initialize merged state dict
        merged_sd = {}
        
        # Process each layer with its specific projection ratio
        for layer_idx, layer_key_name in enumerate(self._layer_keys):
            param_names = self._layer_groups[layer_key_name]
            proj_ratio_tensor = proj_ratios[layer_idx]  # Keep as tensor to maintain gradients!
            
            # Compute global dimension for this layer
            global_dim = compute_global_dim(self._base_sd, param_names)
            
            # Calculate target dimension (continuous)
            target_dim_float = global_dim * proj_ratio_tensor
            
            # Round to integer for Fastfood operations
            # Straight-through estimator: forward uses rounded, backward uses continuous
            proj_dim = max(1, min(int(target_dim_float.round().item()), global_dim))
            
            # Create single Fastfood operator with rounded dimension
            seed_key = layer_key_name
            fwd, lift = create_fastfood_ops(
                global_dim=global_dim,
                proj_dim=proj_dim,
                seed_key=seed_key,
                device=self.device,
                use_G=self.use_G,
            )
            
            # Collect task vectors for this layer
            task_vecs = []
            for donor_sd in self._donors_sd:
                layer_delta = []
                for pname in param_names:
                    delta = donor_sd[pname].to(self.device) - self._base_sd[pname].to(self.device)
                    layer_delta.append(delta.flatten())
                task_vecs.append(torch.cat(layer_delta))
            
            # Stack into (K, D) and project
            task_vecs_stacked = torch.stack(task_vecs, dim=0)  # (K, D)
            U = torch.stack([fwd(tv) for tv in task_vecs_stacked], dim=0)  # (K, m)
            
            # Merge in subspace
            z_merged = zero_aware_aggregate(
                U,
                merge_func=self.merge_func,
                weights=self._normalized_weights,
            )
            
            # Lift back to full dimension
            delta_merged = lift(z_merged) * self.scale
            
            # Straight-through estimator: 
            # Forward uses discrete proj_dim, backward flows through continuous proj_ratio_tensor
            # This allows gradients to flow back to proj_ratio_tensor for learning
            actual_ratio = proj_dim / global_dim
            if actual_ratio > 0:  # Avoid division by zero
                delta_merged = delta_merged * (proj_ratio_tensor / actual_ratio)
            
            # Unpack back to parameter shapes
            offset = 0
            for pname in param_names:
                pshape = self._base_sd[pname].shape
                numel = self._base_sd[pname].numel()
                delta_p = delta_merged[offset : offset + numel].reshape(pshape)
                merged_sd[pname] = self._base_sd[pname].to(self.device) + delta_p
                offset += numel
        
        # Copy non-merged parameters
        for k in self._base_sd.keys():
            if k not in merged_sd:
                merged_sd[k] = self._base_sd[k]
        
        return merged_sd
    
    def run(self, modelpool: BaseModelPool, **kwargs: Any) -> nn.Module:
        """
        Run the learnable Fastfood merging algorithm.
        
        Steps:
        1. Construct the learnable wrapped model
        2. Perform test-time adaptation to learn projection ratios
        3. Merge and unload the final model
        
        Args:
            modelpool: Model pool with pretrained and finetuned models
            
        Returns:
            Merged model with optimized projection ratios
        """
        log.info("Starting Learnable Fastfood Merging")
        modelpool = to_modelpool(modelpool)
        self.modelpool = modelpool
        
        with self.profile("construct learnable model"):
            module = self.construct_learnable_merged_model(modelpool)
        
        with self.profile("test-time adaptation"):
            module = self.test_time_adaptation(module)
        
        # Save learned projection ratios if requested
        if self.save_projection_ratios:
            log.info(f"Saving learned projection ratios to {self.save_projection_ratios}")
            torch.save(module.projection_ratios.detach().cpu(), self.save_projection_ratios)
        
        with self.profile("merge and unload"):
            merged_model = module.merge_and_unload()
        
        self.print_profile_summary()
        return merged_model
    
    # ==================== Abstract Methods to be Implemented by Subclasses ====================
    
    @abstractmethod
    def on_test_time_adaptation_start(self):
        """
        Hook called before test-time adaptation starts.
        
        Subclasses should implement this to set up task-specific components
        (e.g., zero-shot classification heads for CLIP).
        """
        pass
    
    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str):
        """
        Get an iterator over shuffled test data for a task.
        
        Args:
            task: Task name
            
        Returns:
            Iterator yielding batches of test data
        """
        pass
    
    @abstractmethod
    def compute_logits(self, module: LearnableFastfoodMergedModel, batch, task: str) -> Tensor:
        """
        Compute logits for a batch of data.
        
        Args:
            module: The learnable merged model
            batch: Batch of input data
            task: Task name
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        pass
    
    # ==================== Test-Time Adaptation ====================
    
    def test_time_adaptation(
        self, module: LearnableFastfoodMergedModel
    ) -> LearnableFastfoodMergedModel:
        """
        Perform test-time adaptation to learn optimal projection ratios.
        
        This method optimizes the projection ratios by minimizing entropy loss
        on unlabeled test data. The merging process itself is not changed,
        only the subspace dimensions are adapted.
        
        Args:
            module: Learnable Fastfood merged model
            
        Returns:
            Model with optimized projection ratios
        """
        log.info("Starting test-time adaptation for projection ratios")
        self.on_test_time_adaptation_start()
        
        # Setup optimizer
        optimizer = torch.optim.Adam([module.projection_ratios], lr=self.lr)
        log.info(f"Using {self.optimizer} optimizer with lr={self.lr}")
        
        # Move to device
        module = module.to(self.device)
        
        # Training mode
        module.train()
        
        # Initial merge
        module.merge_weights()
        
        # Optimization loop
        pbar = tqdm(
            range(self.max_steps),
            desc="Learning projection ratios",
            dynamic_ncols=True,
        )
        
        for step_idx in pbar:
            optimizer.zero_grad()
            
            # Accumulate loss across tasks
            total_loss = 0.0
            accumulated_loss = None
            
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, batch, task)
                    loss = entropy_loss(logits)
                    total_loss += loss.item()
                    
                    # Debug gradient connection
                    if step_idx == 0 and task == list(self.modelpool.model_names)[0]:
                        log.info(f"First loss requires_grad: {loss.requires_grad}")
                        log.info(f"Logits requires_grad: {logits.requires_grad}")
                        log.info(f"Projection ratios requires_grad: {module.projection_ratios.requires_grad}")
                        log.info(f"Projection ratios grad_fn: {module.projection_ratios.grad_fn}")
                
                # Accumulate losses for single backward pass
                if loss.requires_grad:
                    if accumulated_loss is None:
                        accumulated_loss = loss
                    else:
                        accumulated_loss = accumulated_loss + loss
                else:
                    log.warning(f"Task {task} produced loss without gradients: {loss.requires_grad}")
                    # Skip this loss if it doesn't require gradients
            
            with self.profile("backward pass"):
                # Single backward pass for all accumulated losses
                if accumulated_loss is not None:
                    if accumulated_loss.requires_grad:
                        accumulated_loss.backward()
                    else:
                        log.warning(f"Accumulated loss does not require grad: requires_grad={accumulated_loss.requires_grad}")
                        log.warning(f"Loss grad_fn: {accumulated_loss.grad_fn}")
                        log.warning("Skipping backward pass - no gradient connection to learnable parameters")
                else:
                    log.warning("No accumulated loss to backward through")
            
            with self.profile("optimizer step"):
                optimizer.step()
            
            # Merge weights after optimizer step, outside gradient computation
            with torch.no_grad():
                with self.profile("merge weights"):
                    module.merge_weights(force=True)
            
            # Logging
            stats = module.get_projection_ratio_stats()
            stats["train/loss"] = total_loss / len(self.modelpool.model_names)
            pbar.set_postfix(stats)
        
        log.info("Test-time adaptation completed")
        log.info(f"Final projection ratios - min: {stats['proj_ratio_min']:.4f}, "
                f"max: {stats['proj_ratio_max']:.4f}, mean: {stats['proj_ratio_mean']:.4f}")
        
        return module
