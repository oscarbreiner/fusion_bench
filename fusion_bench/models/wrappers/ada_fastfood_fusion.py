"""
AdaFastFood Fusion Wrapper: Hybrid of AdaMerging and FastFood Subspace Projection

This module implements a hybrid approach where:
1. Each layer learns its own optimal subspace dimension (projection ratio)
2. Within each subspace, standard AdaMerging coefficients are learned
3. FastFood transforms provide efficient subspace projections and lifting
"""

import functools
import hashlib
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.func import functional_call

from fusion_bench.models.utils import del_attr, get_attr, set_attr
from fusion_bench.utils.type import StateDictType, TorchModelType

__all__ = ["AdaFastFoodMergedModel"]

log = logging.getLogger(__name__)


def _next_pow2(n: int) -> int:
    """Find next power of 2 >= n"""
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def _fwht_inplace_ortho(x: Tensor) -> Tensor:
    """In-place orthonormal FWHT along the last dim (scale 1/sqrt(n))."""
    n = x.shape[-1]
    if n <= 1:
        return x
    h = 1
    while h < n:
        for i in range(0, n, h << 1):
            for j in range(h):
                u = x[..., i + j]
                v = x[..., i + j + h]
                x[..., i + j] = u + v
                x[..., i + j + h] = u - v
        h <<= 1
    x.mul_(1.0 / math.sqrt(n))
    return x


def _seed_from(s: str) -> int:
    """Generate deterministic seed from string"""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")


def _create_fastfood_ops(
    global_dim: int,
    proj_dim: int,
    seed_key: str,
    device: torch.device,
    use_G: bool = False,
):
    """
    Create FastFood forward and lift operations for a given dimension.
    
    Args:
        global_dim: Original parameter dimension
        proj_dim: Projected subspace dimension
        seed_key: Seed for deterministic random matrices
        device: Device to create tensors on
        use_G: Whether to use Gaussian scaling in FastFood
        
    Returns:
        Tuple of (forward_fn, lift_fn)
    """
    torch.manual_seed(_seed_from(seed_key))
    D = int(global_dim)
    L = _next_pow2(D)
    m = max(1, int(proj_dim))

    # FastFood parameters: B, G, Pi for SRHT
    B = (torch.randint(0, 2, (L,), dtype=torch.int8, device=device) * 2 - 1).to(
        dtype=torch.float32
    )
    G = (
        torch.randn(L, device=device, dtype=torch.float32)
        if use_G
        else torch.ones(L, device=device, dtype=torch.float32)
    )
    Pi = torch.randperm(L, device=device)
    inv_Pi = torch.argsort(Pi)

    # Random row subset for Johnson-Lindenstrauss
    row_idx = torch.randperm(L, device=device)[:m]
    scale = math.sqrt(L / m)

    def forward_fn(x_D: Tensor) -> Tensor:
        """Project tensor to subspace using FastFood"""
        assert x_D.shape[-1] == D, f"Expected dimension {D}, got {x_D.shape[-1]}"
        x = x_D.clone()
        
        # Pad to next power of 2 if needed
        if D < L:
            x = torch.nn.functional.pad(x, (0, L - D))
        
        x = x.to(torch.float32, copy=False)
        
        # FastFood transform: H * Pi * G * H * B
        x.mul_(B)
        _fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G)
        _fwht_inplace_ortho(x)
        
        # Johnson-Lindenstrauss projection
        x = x.index_select(dim=-1, index=row_idx)
        return (scale * x).contiguous()

    def lift_fn(y: Tensor) -> Tensor:
        """Lift tensor from subspace back to original space"""
        y = (y.to(torch.float32, copy=False) / scale)
        
        # Inverse Johnson-Lindenstrauss
        y_full = torch.zeros(y.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
        y_full.index_copy_(dim=-1, index=row_idx, source=y)
        
        # Inverse FastFood transform: B * H * Pi^-1 * G * H
        _fwht_inplace_ortho(y_full)
        y_full.mul_(G)
        y_full = y_full[..., inv_Pi]
        _fwht_inplace_ortho(y_full)
        y_full.mul_(B)
        
        # Truncate back to original dimension
        return y_full[..., :D].contiguous()

    return forward_fn, lift_fn


class AdaFastFoodMergedModel(nn.Module):
    """
    Hybrid AdaMerging + FastFood model that learns:
    1. Per-layer projection ratios (subspace dimensions)  
    2. Per-task per-layer AdaMerging coefficients within subspaces
    
    The model dynamically projects task vectors to learned subspaces,
    performs AdaMerging in compressed space, then lifts back to original space.
    """
    
    _merged_state_dict: StateDictType = None
    
    def __init__(
        self,
        pretrained_model: TorchModelType,
        finetuned_models: List[TorchModelType],
        proj_init_strategy: str = "conservative",
        proj_init_value: float = 0.3,
        ada_init_value: Optional[float] = None,
        use_G: bool = False,
        clamp_weights: bool = True,
        clamp_proj: bool = True,
        tie_weights: bool = False,
        strict: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize AdaFastFood merged model.
        
        Args:
            pretrained_model: Base pretrained model
            finetuned_models: List of fine-tuned models to merge
            proj_init_strategy: How to initialize projection ratios 
                ("conservative", "layer_dependent", "uniform")
            proj_init_value: Initial projection ratio value (for "conservative"/"uniform")
            ada_init_value: Initial AdaMerging coefficient (default: 1/num_tasks)
            use_G: Whether to use Gaussian scaling in FastFood
            clamp_weights: Whether to clamp AdaMerging weights to [0,1]
            clamp_proj: Whether to clamp projection ratios to [0.1,1.0]
            tie_weights: Pass to functional_call
            strict: Pass to functional_call  
            device: Device for computations
        """
        super().__init__()
        
        self.num_tasks = len(finetuned_models)
        self.use_G = use_G
        self.clamp_weights = clamp_weights
        self.clamp_proj = clamp_proj
        self.tie_weights = tie_weights
        self.strict = strict
        self.device = torch.device(device)
        
        # Extract task vectors and compute layer information
        self._extract_task_vectors(pretrained_model, finetuned_models)
        
        # Initialize learnable parameters
        self._init_learnable_params(proj_init_strategy, proj_init_value, ada_init_value)
        
        # Store models
        self.pretrained_model = pretrained_model.requires_grad_(False)
        for m in finetuned_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(finetuned_models)
        
    def _extract_task_vectors(self, pretrained_model: TorchModelType, finetuned_models: List[TorchModelType]):
        """Extract task vectors and organize by layer structure"""
        
        # Get all trainable parameters
        trainable_params = []
        param_shapes = []
        param_names = []
        
        for name, param in pretrained_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                param_shapes.append(param.shape)
                param_names.append(name)
        
        self.num_layers = len(trainable_params)
        self.param_shapes = param_shapes
        self.param_names = param_names
        
        print(f"AdaFastFood: Found {self.num_layers} trainable layers")
        print(f"Parameter shapes: {param_shapes[:5]}...")  # Show first 5
        
        # Compute task vectors (difference from pretrained)
        for name, param in pretrained_model.named_parameters():
            if not param.requires_grad:
                # Remove non-trainable parameters from finetuned models
                for m in finetuned_models:
                    del_attr(m, name.split("."))
            else:
                # Convert to task vectors (finetuned - pretrained)
                for m in finetuned_models:
                    finetuned_param = get_attr(m, name.split("."))
                    get_attr(m, name.split(".")).data = finetuned_param - param
                    
    def _init_learnable_params(self, proj_init_strategy: str, proj_init_value: float, ada_init_value: Optional[float]):
        """Initialize learnable projection ratios and AdaMerging weights"""
        
        # Initialize projection ratios per layer
        if proj_init_strategy == "conservative":
            proj_init = torch.full((self.num_layers,), proj_init_value)
        elif proj_init_strategy == "layer_dependent":
            # Early layers more compressed, later layers less compressed
            proj_init = torch.linspace(0.2, 0.7, self.num_layers)
        elif proj_init_strategy == "uniform":
            proj_init = torch.full((self.num_layers,), 0.5)
        else:
            raise ValueError(f"Unknown proj_init_strategy: {proj_init_strategy}")
            
        self.proj_params = nn.Parameter(proj_init)
        
        # Initialize AdaMerging weights per task per layer
        if ada_init_value is None:
            ada_init_value = 1.0 / self.num_tasks
            
        self.ada_weights = nn.Parameter(
            torch.full((self.num_tasks, self.num_layers), ada_init_value)
        )
        
        print(f"Initialized proj_params: {self.proj_params.shape} = {self.proj_params[:5].tolist()}...")
        print(f"Initialized ada_weights: {self.ada_weights.shape}")
        
    def _clamp_parameters(self):
        """Clamp parameters to valid ranges"""
        with torch.no_grad():
            if self.clamp_proj:
                self.proj_params.clamp_(0.1, 1.0)
            if self.clamp_weights:
                self.ada_weights.clamp_(0.0, 2.0)  # Allow some amplification
    
    @property
    def forward_model(self):
        """Get functional_call partial for merged model"""
        return functools.partial(
            functional_call,
            self.pretrained_model,
            self._merged_state_dict,
            tie_weights=self.tie_weights,
            strict=self.strict,
        )
    
    def merge_weights(self):
        """
        Perform hybrid AdaFastFood merging:
        1. Project each layer's task vectors to learned subspace
        2. Apply AdaMerging coefficients in subspace  
        3. Lift merged result back to original space
        4. Add to pretrained model parameters
        """
        # Clamp parameters to valid ranges
        self._clamp_parameters()
        
        # Start with pretrained model state
        state_dict = self.pretrained_model.state_dict(keep_vars=True)
        
        # Process each layer independently
        for layer_idx, param_name in enumerate(self.param_names):
            # Get current projection ratio and compute subspace dimension
            proj_ratio = self.proj_params[layer_idx]
            param_shape = self.param_shapes[layer_idx]
            original_dim = param_shape.numel()
            proj_dim = max(1, int(proj_ratio.item() * original_dim))
            
            # Create FastFood operators for this layer
            seed_key = f"layer_{layer_idx}_{param_name}"
            fwd_fn, lift_fn = _create_fastfood_ops(
                original_dim, proj_dim, seed_key, self.device, self.use_G
            )
            
            # Collect task vectors for this layer from all tasks
            task_vectors_layer = []
            for task_idx in range(self.num_tasks):
                task_param = get_attr(self.task_vectors[task_idx], param_name.split("."))
                # Flatten parameter to 1D for FastFood
                task_vector_flat = task_param.view(-1).to(self.device)
                task_vectors_layer.append(task_vector_flat)
            
            # Project all task vectors to subspace
            projected_task_vectors = []
            for tv in task_vectors_layer:
                projected_tv = fwd_fn(tv)
                projected_task_vectors.append(projected_tv)
            
            # Apply AdaMerging coefficients in subspace
            layer_ada_weights = self.ada_weights[:, layer_idx]  # Shape: (num_tasks,)
            merged_subspace = torch.zeros_like(projected_task_vectors[0])
            
            for task_idx, (weight, proj_tv) in enumerate(zip(layer_ada_weights, projected_task_vectors)):
                merged_subspace += weight * proj_tv
            
            # Lift merged result back to original space
            merged_flat = lift_fn(merged_subspace)
            merged_param = merged_flat.view(param_shape).to(state_dict[param_name].dtype)
            
            # Add merged task vector to pretrained parameter
            state_dict[param_name] = state_dict[param_name] + merged_param
            
        self._merged_state_dict = state_dict
        return state_dict
    
    def merge_and_unload(self):
        """Merge weights and return standalone model"""
        self.merge_weights()
        self.pretrained_model.load_state_dict(self._merged_state_dict)
        return self.pretrained_model
    
    def forward(self, *args, **kwargs):
        """Forward pass through merged model"""
        if self._merged_state_dict is None:
            self.merge_weights()
        return self.forward_model(args=args, kwargs=kwargs)
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics for analysis"""
        stats = {}
        total_original = 0
        total_compressed = 0
        
        for layer_idx, param_shape in enumerate(self.param_shapes):
            original_dim = param_shape.numel()
            proj_ratio = self.proj_params[layer_idx].item()
            compressed_dim = max(1, int(proj_ratio * original_dim))
            
            total_original += original_dim
            total_compressed += compressed_dim
            
            if layer_idx < 10:  # Show first 10 layers
                stats[f"layer_{layer_idx}_ratio"] = proj_ratio
                stats[f"layer_{layer_idx}_compression"] = compressed_dim / original_dim
        
        stats["overall_compression_ratio"] = total_compressed / total_original
        stats["memory_savings"] = 1 - (total_compressed / total_original)
        
        return stats