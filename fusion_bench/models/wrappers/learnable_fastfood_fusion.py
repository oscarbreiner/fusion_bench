"""
Wrapper model for learnable layer-wise Fastfood merging.

This module provides a wrapper that performs Fastfood/SRHT-based model merging
with learnable projection ratios for each layer. The projection ratios are
optimized during test-time adaptation to find optimal subspace dimensions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


def get_learnable_projection_ratios(
    num_layers: int,
    init_value: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Initialize learnable projection ratios for each layer.
    
    Args:
        num_layers: Number of layers in the model
        init_value: Initial projection ratio value (should be in [0, 1])
        dtype: Data type for the tensor
        
    Returns:
        Tensor of shape (num_layers,) with initial projection ratios
    """
    assert 0.0 < init_value <= 1.0, f"init_value must be in (0, 1], got {init_value}"
    assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"
    
    return torch.full((num_layers,), init_value, dtype=dtype)


class LearnableFastfoodMergedModel(nn.Module):
    """
    Wrapper model that performs Fastfood merging with learnable projection ratios.
    
    This model wraps the base merging functionality and makes the projection ratio
    for each layer a learnable parameter. During forward passes, it dynamically
    computes the merged model using the current projection ratios.
    
    The actual merging is delegated to a merge_function that takes projection ratios
    as input and returns a merged state dict.
    """
    
    def __init__(
        self,
        projection_ratios: Tensor,
        pretrained_model: nn.Module,
        merge_function: callable,
        clamp_ratios: bool = True,
        min_ratio: float = 0.01,
        max_ratio: float = 1.0,
    ):
        """
        Initialize the learnable Fastfood merged model.
        
        Args:
            projection_ratios: Initial projection ratios per layer, shape (num_layers,)
            pretrained_model: The base pretrained model
            merge_function: Function that performs merging given projection ratios.
                            Signature: merge_function(proj_ratios: Tensor) -> StateDictType
            clamp_ratios: Whether to clamp projection ratios to [min_ratio, max_ratio]
            min_ratio: Minimum projection ratio
            max_ratio: Maximum projection ratio
        """
        super().__init__()
        
        self.clamp_ratios = clamp_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
        # Make projection ratios learnable
        self.projection_ratios = nn.Parameter(projection_ratios, requires_grad=True)
        
        # Store the pretrained model and merge function
        self.pretrained_model = pretrained_model.requires_grad_(False)
        self.merge_function = merge_function
        
        # Cache for the current merged state dict
        self._merged_state_dict: Optional[StateDictType] = None
        self._last_ratios: Optional[Tensor] = None
        
        log.info(f"Initialized LearnableFastfoodMergedModel with {len(projection_ratios)} learnable projection ratios")
    
    def get_clamped_ratios(self) -> Tensor:
        """Get projection ratios, optionally clamped to valid range."""
        if self.clamp_ratios:
            return torch.clamp(self.projection_ratios, self.min_ratio, self.max_ratio)
        return self.projection_ratios
    
    def merge_weights(self, force: bool = False) -> None:
        """
        Compute merged weights using current projection ratios.
        
        Args:
            force: If True, recompute even if ratios haven't changed
        """
        current_ratios = self.get_clamped_ratios()
        
        # Check if we need to recompute
        if not force and self._last_ratios is not None:
            if torch.allclose(current_ratios, self._last_ratios, atol=1e-6):
                return
        
        # Compute merged state dict using the merge function
        self._merged_state_dict = self.merge_function(current_ratios)
        self._last_ratios = current_ratios.detach().clone()
    
    def forward(self, *args, **kwargs):
        """
        Forward pass using the merged model.
        
        This dynamically merges weights and uses functional forward pass
        to maintain gradient flow to projection_ratios.
        """
        # Ensure weights are merged with current ratios
        self.merge_weights()
        
        # CRITICAL: We cannot use param.data assignment as it detaches gradients!
        # Instead, we use torch.func.functional_call to run the model
        # with the merged weights while maintaining gradient flow.
        
        # Import functional_call for stateless execution
        from torch.func import functional_call
        
        # Use functional_call to execute the model with merged weights
        # This maintains gradient connections from merged weights to projection_ratios
        return functional_call(self.pretrained_model, self._merged_state_dict, args, kwargs)
    
    def merge_and_unload(self) -> nn.Module:
        """
        Merge weights permanently and return the base model.
        
        Returns:
            The pretrained model with merged weights loaded
        """
        log.info("Merging weights and unloading wrapper...")
        
        # Ensure final merge with current ratios
        self.merge_weights(force=True)
        
        # Load merged weights into pretrained model
        self.pretrained_model.load_state_dict(self._merged_state_dict, strict=False)
        
        log.info(f"Final projection ratios - min: {self._last_ratios.min().item():.4f}, "
                f"max: {self._last_ratios.max().item():.4f}, "
                f"mean: {self._last_ratios.mean().item():.4f}")
        
        return self.pretrained_model
    
    def get_projection_ratio_stats(self) -> Dict[str, float]:
        """Get statistics about current projection ratios."""
        ratios = self.get_clamped_ratios()
        return {
            "proj_ratio_min": ratios.min().item(),
            "proj_ratio_max": ratios.max().item(),
            "proj_ratio_mean": ratios.mean().item(),
            "proj_ratio_std": ratios.std().item(),
        }
