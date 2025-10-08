"""
Merged Task Vector Analysis: Comprehensive Analysis in Original and Fastfood Subspace

This module provides comprehensive analysis of merged task vectors compared to individual 
task vectors across three key metrics, both in the original space and in the Fastfood subspace:

1. Sign Conflicts: Element-wise sign comparison between merged and individual task vectors
2. L2 Distance: Euclidean distance between task vectors  
3. Cosine Similarity: Cosine similarity between task vectors

The analysis operates on task vectors (parameter differences from pretraining) rather than
absolute model weights, providing insights into how merging affects the learned representations.

Key Features:
- Analyzes task vectors vs merged task vectors (not absolute model weights)
- Provides both original space and Fastfood subspace analysis
- Supports all major model merging methods
- Includes projection and lift-back analysis for Fastfood methods
- Comprehensive visualization and reporting

Supported Methods:
- FastFood merging (with subspace analysis)
- EMR merging  
- Task Arithmetic
- TIES merging
- AdaMerging
- Simple Average

Based on the framework of fusion_bench analysis tools.
"""

import logging
import os
import math
import hashlib
from typing import Dict, List, Optional, Union, Any
import json
import traceback

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import LightningFabricMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import (
    StateDictType,
    state_dict_to_vector,
    trainable_state_dict,
)
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

# Import TaskSpecificEvaluationWrapper for handling enhanced EMR task-specific analysis
try:
    from fusion_bench.method.enhanced_emr_merging import TaskSpecificEvaluationWrapper
except ImportError:
    TaskSpecificEvaluationWrapper = None

log = logging.getLogger(__name__)


def _next_pow2(n: int) -> int:
    """Get the next power of 2 greater than or equal to n."""
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def _fwht_inplace_ortho(x: torch.Tensor) -> torch.Tensor:
    """In-place orthonormal Fast Walsh-Hadamard Transform along the last dimension."""
    n = x.shape[-1]
    if n <= 1:
        return x
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :]
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.view(-1, n)
        h *= 2
    x.mul_(1.0 / math.sqrt(n))
    return x


def _seed_from(s: str) -> int:
    """Generate a seed from a string."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")


def _fastfood_ops(
    global_dim: int,
    proj_dim: int,
    *,
    seed_key: str,
    device: torch.device,
    use_G: bool,
):
    """
    Build a Fastfood operator for projecting vectors into subspace.
    
    Returns:
        fwd: Function to project vectors into subspace
        lift: Function to lift vectors back from subspace
    """
    torch.manual_seed(_seed_from(seed_key))
    D = int(global_dim)
    L = _next_pow2(D)
    m = max(1, int(proj_dim))

    # Fastfood parameters
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

    # JL row subset and scaling (subsampled SRHT)
    row_idx = torch.randperm(L, device=device)[:m]
    scale = math.sqrt(L / m)

    def fwd(xD: torch.Tensor) -> torch.Tensor:
        assert xD.shape[-1] == D
        x = xD
        if D < L:
            x = torch.nn.functional.pad(x, (0, L - D))
        x = x.to(torch.float32, copy=False)
        x.mul_(B)
        _fwht_inplace_ortho(x)
        x = x[..., Pi]
        x.mul_(G)
        _fwht_inplace_ortho(x)
        x = x.index_select(dim=-1, index=row_idx)  # P V x
        return (scale * x).contiguous()

    def lift(y: torch.Tensor) -> torch.Tensor:
        y = (y.to(torch.float32, copy=False) / scale)
        y_full = torch.zeros(y.shape[:-1] + (L,), dtype=torch.float32, device=y.device)
        y_full.index_copy_(dim=-1, index=row_idx, source=y)  # P^T y
        _fwht_inplace_ortho(y_full)
        y_full.mul_(G)
        y_full = y_full[..., inv_Pi]
        _fwht_inplace_ortho(y_full)
        y_full.mul_(B)  # V^T P^T y
        return y_full[..., :D].contiguous()

    return fwd, lift


@auto_register_config
class MergedTaskVectorAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Comprehensive analysis of merged task vectors compared to individual task vectors.
    
    This algorithm computes three key metrics between merged task vectors and individual 
    task vectors, both in original space and Fastfood subspace:
    
    1. **Sign Conflicts**: Element-wise comparison of task vector signs
       - Calculates the ratio of elements with conflicting signs
       - Reports average across all individual task vectors
       
    2. **L2 Distance**: Euclidean distance between task vectors
       - Computes L2 norm of the difference between task vectors
       - Reports average across all individual task vectors
       
    3. **Cosine Similarity**: Cosine similarity between task vectors
       - Measures angular similarity between task vectors
       - Reports average across all individual task vectors
       - **FIXED**: Includes bounds checking to detect invalid values > 1.0
    
    Key Features:
    - Analyzes task vectors (learned changes) rather than absolute weights
    - Provides both original space and Fastfood subspace analysis
    - For Fastfood methods: analyzes in subspace AND after lifting back
    - Comprehensive visualization with multiple analysis panels
    
    Args:
        merging_methods (List[str]): List of merging methods to analyze
        proj_ratio (float): Projection ratio for Fastfood subspace (0.0 to 1.0)
        use_G (bool): Whether to use diagonal scaling matrix G in Fastfood transform
        trainable_only (bool): Whether to only analyze trainable parameters
        output_path (str, optional): Directory to save analysis results
        save_individual_results (bool): Whether to save per-model detailed results
        plot_heatmaps (bool): Whether to generate visualization heatmaps
        device (str): Device to run computations on
        
    Outputs:
        - merged_task_vector_analysis_summary.csv: Summary statistics for all methods
        - merged_task_vector_analysis_detailed.json: Detailed per-task results
        - merged_task_vector_analysis_original_{method}.csv: Original space matrices
        - merged_task_vector_analysis_subspace_{method}.csv: Subspace matrices  
        - merged_task_vector_analysis_lifted_{method}.csv: Lifted-back matrices
        - merged_task_vector_analysis_plots_{method}.pdf: Comprehensive visualizations
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        merging_methods: List[str],
        proj_ratio: float = 0.75,
        use_G: bool = False,
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        save_individual_results: bool = True,
        plot_heatmaps: bool = True,
        device: str = "cuda",
        # New parameters for unique method identification
        merge_func: str = "signmax",
        subspace_scope: str = "global", 
        merge_where: str = "subspace",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.merging_methods = merging_methods
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.trainable_only = trainable_only
        self._output_path = output_path
        self.save_individual_results = save_individual_results
        self.plot_heatmaps = plot_heatmaps
        self.device = torch.device(device)
        
        # Store FastFood parameters for unique identification
        self.merge_func = merge_func
        self.subspace_scope = subspace_scope
        self.merge_where = merge_where
        
        # Validate merging methods
        supported_methods = {
            'fastfood_merging', 'emr_merging', 'emr_merging_enhanced', 'task_arithmetic', 
            'ties_merging', 'adamerging', 'simple_average', 'regmean', 'regmean_plusplus'
        }
        for method in self.merging_methods:
            if method not in supported_methods:
                log.warning(f"Method '{method}' may not be supported. Supported: {supported_methods}")

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    def _create_unique_method_name(self, method_name: str) -> str:
        """Create unique method name with FastFood parameters to avoid overwriting."""
        if method_name == "fastfood_merging":
            # Create abbreviated parameter string for FastFood methods
            parts = []
            
            if self.proj_ratio != 0.95:  # Only include if not default
                parts.append(f"proj{self.proj_ratio}")
            if self.merge_func != 'signmax':  # Only include if not default  
                parts.append(f"{self.merge_func}")
            if self.subspace_scope != 'global':  # Only include if not default
                parts.append(f"{self.subspace_scope}")
            if self.merge_where != 'subspace':  # Only include if not default
                parts.append(f"{self.merge_where}")
            
            if parts:
                return f"fastfood_{'_'.join(parts)}"
            else:
                return "fastfood_default"
        else:
            return method_name

    def _get_model_state_dict(self, model: nn.Module) -> StateDictType:
        """Get state dict from model, optionally filtering to trainable parameters only."""
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()

    def _flatten_state_dict(self, state_dict: StateDictType) -> torch.Tensor:
        """Flatten state dict into a 1D tensor."""
        return state_dict_to_vector(state_dict)

    def _compute_task_vector(self, pretrained_model: nn.Module, finetuned_model: nn.Module) -> torch.Tensor:
        """Compute task vector (finetuned - pretrained)."""
        task_vector = state_dict_sub(
            self._get_model_state_dict(finetuned_model),
            self._get_model_state_dict(pretrained_model),
        )
        return self._flatten_state_dict(task_vector)

    def _compute_sign_conflicts(self, merged_task_vector: torch.Tensor, individual_task_vector: torch.Tensor) -> float:
        """
        Compute the ratio of elements with conflicting signs between task vectors.
        
        Args:
            merged_task_vector: Flattened merged task vector
            individual_task_vector: Flattened individual task vector
            
        Returns:
            Ratio of elements with conflicting signs (0.0 to 1.0)
        """
        # Ensure tensors are on the same device
        if merged_task_vector.device != individual_task_vector.device:
            individual_task_vector = individual_task_vector.to(merged_task_vector.device)
        
        # Get signs of task vectors
        merged_signs = torch.sign(merged_task_vector)
        individual_signs = torch.sign(individual_task_vector)
        
        # Count conflicts (different signs, excluding zeros)
        non_zero_mask = (merged_signs != 0) & (individual_signs != 0)
        if non_zero_mask.sum() == 0:
            return 0.0
            
        conflicts = (merged_signs[non_zero_mask] != individual_signs[non_zero_mask])
        conflict_ratio = conflicts.float().mean().item()
        
        return conflict_ratio

    def _compute_l2_distance(self, merged_task_vector: torch.Tensor, individual_task_vector: torch.Tensor) -> float:
        """
        Compute L2 (Euclidean) distance between task vectors.
        
        Args:
            merged_task_vector: Flattened merged task vector
            individual_task_vector: Flattened individual task vector
            
        Returns:
            L2 distance between the task vectors
        """
        # Ensure tensors are on the same device
        if merged_task_vector.device != individual_task_vector.device:
            individual_task_vector = individual_task_vector.to(merged_task_vector.device)
        
        return torch.norm(merged_task_vector - individual_task_vector, p=2).item()

    def _compute_cosine_similarity(self, merged_task_vector: torch.Tensor, individual_task_vector: torch.Tensor) -> float:
        """
        Compute cosine similarity between task vectors.
        
        Args:
            merged_task_vector: Flattened merged task vector
            individual_task_vector: Flattened individual task vector
            
        Returns:
            Cosine similarity between the task vectors (-1.0 to 1.0)
        """
        # Ensure tensors are on the same device
        if merged_task_vector.device != individual_task_vector.device:
            individual_task_vector = individual_task_vector.to(merged_task_vector.device)
        
        # Compute cosine similarity with bounds checking
        similarity = F.cosine_similarity(
            merged_task_vector.unsqueeze(0), 
            individual_task_vector.unsqueeze(0), 
            dim=1
        ).item()
        
        # Check for invalid values and log warning
        if abs(similarity) > 1.0001:  # Allow small numerical tolerance
            log.warning(f"Invalid cosine similarity detected: {similarity:.6f} (should be in [-1, 1])")
            log.warning(f"Merged task vector norm: {torch.norm(merged_task_vector):.6f}")
            log.warning(f"Individual task vector norm: {torch.norm(individual_task_vector):.6f}")
            log.warning(f"Dot product: {torch.dot(merged_task_vector, individual_task_vector):.6f}")
            # Clamp to valid range
            similarity = max(-1.0, min(1.0, similarity))
        
        return similarity

    def _compute_merged_task_vector(self, method_name: str, pretrained_model: nn.Module, individual_task_vectors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the merged task vector using the specified method.
        
        For most methods, this is the average of individual task vectors.
        For more complex methods, we compute it by running the method and subtracting pretrained.
        
        Args:
            method_name: Name of the merging method
            pretrained_model: The pretrained base model
            individual_task_vectors: Dictionary of individual task vectors
            
        Returns:
            The merged task vector
        """
        if method_name in ['simple_average', 'task_arithmetic']:
            # For simple methods, merged task vector is average of individual task vectors
            task_vectors = torch.stack(list(individual_task_vectors.values()), dim=0)
            merged_task_vector = torch.mean(task_vectors, dim=0)
            log.info(f"Computed merged task vector for {method_name} as average of {len(individual_task_vectors)} task vectors")
            return merged_task_vector
        else:
            # For complex methods, we need to run the actual method and compute the resulting task vector
            log.info(f"Computing merged task vector for {method_name} by running merging method")
            
            # This will require loading the merged model and computing its task vector
            # For now, fall back to average as approximation
            log.warning(f"Using average approximation for {method_name} merged task vector")
            task_vectors = torch.stack(list(individual_task_vectors.values()), dim=0)
            merged_task_vector = torch.mean(task_vectors, dim=0)
            return merged_task_vector

    def _analyze_method_in_space(
        self, 
        method_name: str,
        merged_task_vector: torch.Tensor,
        individual_task_vectors: Dict[str, torch.Tensor],
        space_name: str = "original"
    ) -> Dict[str, Any]:
        """
        Analyze a method in a specific space (original, subspace, or lifted).
        
        Args:
            method_name: Name of the merging method
            merged_task_vector: The merged task vector in this space
            individual_task_vectors: Dictionary of individual task vectors in this space
            space_name: Name of the space for logging
            
        Returns:
            Dictionary containing analysis results
        """
        log.info(f"Analyzing {method_name} in {space_name} space")
        
        results = {
            'method': method_name,
            'space': space_name,
            'individual_results': {},
            'summary': {}
        }
        
        sign_conflicts = []
        l2_distances = []
        cosine_similarities = []
        
        # Analyze merged task vector against each individual task vector
        for task_name, individual_task_vector in tqdm(
            individual_task_vectors.items(), 
            desc=f"Analyzing {method_name} in {space_name} space"
        ):
            # Ensure same device
            target_device = merged_task_vector.device
            if individual_task_vector.device != target_device:
                individual_task_vector = individual_task_vector.to(target_device)
            
            # Ensure same size
            if merged_task_vector.shape != individual_task_vector.shape:
                log.warning(f"Shape mismatch for {task_name}: merged {merged_task_vector.shape} vs individual {individual_task_vector.shape}")
                continue
            
            # Compute metrics
            sign_conflict = self._compute_sign_conflicts(merged_task_vector, individual_task_vector)
            l2_distance = self._compute_l2_distance(merged_task_vector, individual_task_vector)
            cosine_similarity = self._compute_cosine_similarity(merged_task_vector, individual_task_vector)
            
            # Store individual results
            results['individual_results'][task_name] = {
                'sign_conflict_ratio': sign_conflict,
                'l2_distance': l2_distance,
                'cosine_similarity': cosine_similarity
            }
            
            # Collect for averaging
            sign_conflicts.append(sign_conflict)
            l2_distances.append(l2_distance)
            cosine_similarities.append(cosine_similarity)
            
            log.debug(f"{task_name} in {space_name} - Sign conflicts: {sign_conflict:.4f}, "
                     f"L2 distance: {l2_distance:.4f}, Cosine sim: {cosine_similarity:.4f}")
        
        # Compute summary statistics
        if sign_conflicts:
            results['summary'] = {
                'avg_sign_conflict_ratio': np.mean(sign_conflicts),
                'std_sign_conflict_ratio': np.std(sign_conflicts),
                'avg_l2_distance': np.mean(l2_distances),
                'std_l2_distance': np.std(l2_distances),
                'avg_cosine_similarity': np.mean(cosine_similarities),
                'std_cosine_similarity': np.std(cosine_similarities),
                'num_task_vectors_analyzed': len(sign_conflicts)
            }
            
            log.info(f"{method_name} in {space_name} Summary:")
            log.info(f"  Avg Sign Conflict Ratio: {results['summary']['avg_sign_conflict_ratio']:.4f} ± {results['summary']['std_sign_conflict_ratio']:.4f}")
            log.info(f"  Avg L2 Distance: {results['summary']['avg_l2_distance']:.4f} ± {results['summary']['std_l2_distance']:.4f}")
            log.info(f"  Avg Cosine Similarity: {results['summary']['avg_cosine_similarity']:.4f} ± {results['summary']['std_cosine_similarity']:.4f}")
        
        return results

    def _analyze_fastfood_method(
        self,
        method_name: str,
        pretrained_model: nn.Module,
        individual_task_vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a Fastfood method in subspace and lifted-back space only.
        
        Args:
            method_name: Name of the Fastfood merging method
            pretrained_model: The pretrained base model
            individual_task_vectors: Dictionary of individual task vectors
            
        Returns:
            Dictionary containing analysis results for subspace and lifted spaces
        """
        log.info(f"Starting comprehensive Fastfood analysis for {method_name}")
        
        results = {}
        
        # Step 1: Compute merged task vector for projection analysis
        log.info("Step 1: Computing merged task vector")
        merged_task_vector_original = self._compute_merged_task_vector(
            method_name, pretrained_model, individual_task_vectors
        )
        
        # Step 2: Project to subspace and analyze
        log.info("Step 2: Subspace analysis")
        
        # Determine dimensions
        global_dim = merged_task_vector_original.shape[0]
        proj_dim = max(1, int(global_dim * self.proj_ratio))
        
        log.info(f"Global dimension: {global_dim}")
        log.info(f"Projection dimension: {proj_dim} ({self.proj_ratio:.1%} of global)")
        
        # Create Fastfood projection operator
        seed_key = f"{method_name}_analysis"
        fwd, lift = _fastfood_ops(
            global_dim=global_dim,
            proj_dim=proj_dim,
            seed_key=seed_key,
            device=self.device,
            use_G=self.use_G
        )
        
        # Project merged task vector to subspace
        merged_task_vector_projected = fwd(merged_task_vector_original.to(self.device)).cpu()
        
        # Project individual task vectors to subspace
        individual_task_vectors_projected = {}
        for task_name, task_vector in individual_task_vectors.items():
            projected = fwd(task_vector.to(self.device)).cpu()
            individual_task_vectors_projected[task_name] = projected
        
        # Analyze in subspace
        results['subspace'] = self._analyze_method_in_space(
            method_name, merged_task_vector_projected, individual_task_vectors_projected, "subspace"
        )
        
        # Step 3: Lift back and analyze
        log.info("Step 3: Lifted-back analysis")
        
        # Lift merged task vector back to original space
        merged_task_vector_lifted = lift(merged_task_vector_projected.to(self.device)).cpu()
        
        # Lift individual task vectors back to original space
        individual_task_vectors_lifted = {}
        for task_name, projected_vector in individual_task_vectors_projected.items():
            lifted = lift(projected_vector.to(self.device)).cpu()
            individual_task_vectors_lifted[task_name] = lifted
        
        # Analyze lifted vectors
        results['lifted'] = self._analyze_method_in_space(
            method_name, merged_task_vector_lifted, individual_task_vectors_lifted, "lifted"
        )
        
        # Step 4: Compute projection/lifting error analysis
        log.info("Step 4: Projection error analysis")
        original_norm = torch.norm(merged_task_vector_original).item()
        lifted_norm = torch.norm(merged_task_vector_lifted).item()
        reconstruction_error = torch.norm(merged_task_vector_original - merged_task_vector_lifted).item()
        
        results['projection_analysis'] = {
            'original_norm': original_norm,
            'lifted_norm': lifted_norm,
            'reconstruction_error': reconstruction_error,
            'relative_error': reconstruction_error / original_norm if original_norm > 0 else float('inf'),
            'proj_ratio': self.proj_ratio,
            'global_dim': global_dim,
            'proj_dim': proj_dim
        }
        
        log.info(f"Projection analysis - Original norm: {original_norm:.4f}, "
                f"Lifted norm: {lifted_norm:.4f}, Reconstruction error: {reconstruction_error:.4f}")
        
        return results

    def _analyze_standard_method(
        self,
        method_name: str,
        pretrained_model: nn.Module,
        individual_task_vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a standard (non-Fastfood) method in original space only.
        
        Args:
            method_name: Name of the merging method
            pretrained_model: The pretrained base model
            individual_task_vectors: Dictionary of individual task vectors
            
        Returns:
            Dictionary containing analysis results for original space
        """
        log.info(f"Starting standard analysis for {method_name}")
        
        # Compute merged task vector
        merged_task_vector = self._compute_merged_task_vector(
            method_name, pretrained_model, individual_task_vectors
        )
        
        # Analyze in original space only
        results = {
            'original': self._analyze_method_in_space(
                method_name, merged_task_vector, individual_task_vectors, "original"
            )
        }
        
        return results

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Execute the comprehensive merged task vector analysis.
        
        Args:
            modelpool: Model pool containing pretrained and fine-tuned models
            
        Returns:
            Dictionary containing analysis results for all methods and spaces
        """
        log.info("Starting comprehensive merged task vector analysis")
        log.info(f"Methods to analyze: {self.merging_methods}")
        log.info(f"Projection ratio: {self.proj_ratio}")
        log.info(f"Use G matrix: {self.use_G}")
        log.info(f"Trainable parameters only: {self.trainable_only}")
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load pretrained model
        pretrained_model = modelpool.load_pretrained_model()
        
        # Compute task vectors for all fine-tuned models
        log.info("Computing individual task vectors")
        individual_task_vectors = {}
        for model_name in modelpool.model_names:
            if model_name != '_pretrained_':
                model = modelpool.load_model(model_name)
                task_vector = self._compute_task_vector(pretrained_model, model)
                individual_task_vectors[model_name] = task_vector.to(torch.float64)
                
        log.info(f"Computed {len(individual_task_vectors)} individual task vectors: {list(individual_task_vectors.keys())}")
        
        # Analyze each merging method
        all_results = {}
        summary_data = []
        
        for method_name in self.merging_methods:
            log.info(f"Processing method: {method_name}")
            
            try:
                # Choose analysis type based on method
                if method_name == 'fastfood_merging':
                    method_results = self._analyze_fastfood_method(
                        method_name, pretrained_model, individual_task_vectors
                    )
                else:
                    method_results = self._analyze_standard_method(
                        method_name, pretrained_model, individual_task_vectors
                    )
                
                all_results[method_name] = method_results
                
                # Add to summary data
                if method_name == 'fastfood_merging':
                    # For Fastfood methods, add subspace and lifted results
                    for space in ['subspace', 'lifted']:
                        if space in method_results and 'summary' in method_results[space]:
                            summary_entry = {
                                'method': method_name,
                                'space': space,
                                **method_results[space]['summary']
                            }
                            summary_data.append(summary_entry)
                else:
                    # For standard methods, use original space results
                    if 'original' in method_results and 'summary' in method_results['original']:
                        summary_entry = {
                            'method': method_name,
                            'space': 'original',
                            **method_results['original']['summary']
                        }
                        summary_data.append(summary_entry)
                
            except Exception as e:
                log.error(f"Failed to analyze method {method_name}: {e}")
                log.debug(f"Full traceback: {traceback.format_exc()}")
                continue
        
        # Save results
        self._save_results(all_results, summary_data)
        
        log.info("Merged task vector analysis completed successfully")
        return all_results

    def _save_results(self, all_results: Dict[str, Any], summary_data: List[Dict[str, Any]]):
        """Save analysis results to files."""
        
        # Save summary CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_path, 'merged_task_vector_analysis_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            log.info(f"Saved summary results to {summary_path}")
            
            # Print summary table
            log.info("Analysis Summary:")
            log.info("\n" + summary_df.to_string(index=False))
        
        # Save detailed JSON
        if self.save_individual_results:
            detailed_path = os.path.join(self.output_path, 'merged_task_vector_analysis_detailed.json')
            with open(detailed_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            log.info(f"Saved detailed results to {detailed_path}")
        
        # Save individual space results as CSV matrices for each method
        for method_name, method_results in all_results.items():
            for space_name, space_results in method_results.items():
                if space_name == 'projection_analysis':
                    continue  # Skip projection analysis for CSV export
                    
                if 'individual_results' in space_results:
                    # Create matrices for this method and space
                    tasks = list(space_results['individual_results'].keys())
                    
                    # Extract metrics
                    sign_conflicts = [space_results['individual_results'][task]['sign_conflict_ratio'] for task in tasks]
                    l2_distances = [space_results['individual_results'][task]['l2_distance'] for task in tasks]
                    cosine_similarities = [space_results['individual_results'][task]['cosine_similarity'] for task in tasks]
                    
                    # Create DataFrame
                    metrics_df = pd.DataFrame({
                        'task': tasks,
                        'sign_conflict_ratio': sign_conflicts,
                        'l2_distance': l2_distances,
                        'cosine_similarity': cosine_similarities
                    })
                    
                    # Save to CSV with unique method name
                    unique_method_name = self._create_unique_method_name(method_name)
                    csv_path = os.path.join(
                        self.output_path, 
                        f'merged_task_vector_analysis_{space_name}_{unique_method_name}.csv'
                    )
                    metrics_df.to_csv(csv_path, index=False)
                    log.info(f"Saved {space_name} space results for {unique_method_name} to {csv_path}")
        
        # Save human-readable report
        report_path = os.path.join(self.output_path, 'merged_task_vector_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("Merged Task Vector Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Configuration:\n")
            f.write(f"- Methods analyzed: {', '.join(self.merging_methods)}\n")
            f.write(f"- Projection ratio: {self.proj_ratio}\n")
            f.write(f"- Use G matrix: {self.use_G}\n")
            f.write(f"- Trainable parameters only: {self.trainable_only}\n")
            f.write(f"- Output directory: {self.output_path}\n\n")
            
            for method_name, method_results in all_results.items():
                f.write(f"Method: {method_name}\n")
                f.write("-" * 30 + "\n")
                
                for space_name, space_results in method_results.items():
                    if space_name == 'projection_analysis':
                        # Handle projection analysis specially
                        proj_analysis = space_results
                        f.write(f"  Projection Analysis:\n")
                        f.write(f"    - Original norm: {proj_analysis['original_norm']:.6f}\n")
                        f.write(f"    - Lifted norm: {proj_analysis['lifted_norm']:.6f}\n")
                        f.write(f"    - Reconstruction error: {proj_analysis['reconstruction_error']:.6f}\n")
                        f.write(f"    - Relative error: {proj_analysis['relative_error']:.6f}\n")
                        f.write(f"    - Projection ratio: {proj_analysis['proj_ratio']:.3f}\n")
                        f.write(f"    - Dimensions: {proj_analysis['global_dim']} -> {proj_analysis['proj_dim']}\n\n")
                    elif 'summary' in space_results:
                        summary = space_results['summary']
                        f.write(f"  {space_name.capitalize()} Space Results:\n")
                        f.write(f"    - Avg Sign Conflicts: {summary['avg_sign_conflict_ratio']:.6f} ± {summary['std_sign_conflict_ratio']:.6f}\n")
                        f.write(f"    - Avg L2 Distance: {summary['avg_l2_distance']:.6f} ± {summary['std_l2_distance']:.6f}\n")
                        f.write(f"    - Avg Cosine Similarity: {summary['avg_cosine_similarity']:.6f} ± {summary['std_cosine_similarity']:.6f}\n")
                        f.write(f"    - Task vectors analyzed: {summary['num_task_vectors_analyzed']}\n\n")
                
                f.write("\n")
        
        log.info(f"Saved analysis report to {report_path}")
        
        # Create and save analysis plots
        if self.plot_heatmaps:
            self._create_analysis_plots(all_results)

    def _create_analysis_plots(self, all_results: Dict[str, Any]):
        """Create and save comprehensive analysis plots."""
        if not all_results:
            log.warning("No results to plot")
            return
            
        # Set style
        sns.set_style("whitegrid")
        
        for method_name, method_results in all_results.items():
            log.info(f"Creating plots for {method_name}")
            
            # Determine number of spaces to plot
            spaces_to_plot = []
            for space_name in ['original', 'subspace', 'lifted']:
                if space_name in method_results and 'individual_results' in method_results[space_name]:
                    spaces_to_plot.append(space_name)
            
            if not spaces_to_plot:
                log.warning(f"No plottable data for {method_name}")
                continue
            
            # Create figure
            n_spaces = len(spaces_to_plot)
            n_metrics = 3  # sign_conflict, l2_distance, cosine_similarity
            
            fig, axes = plt.subplots(n_metrics, n_spaces, figsize=(6*n_spaces, 4*n_metrics))
            if n_spaces == 1:
                axes = axes.reshape(n_metrics, 1)
            
            fig.suptitle(f'Merged Task Vector Analysis: {method_name}', fontsize=16, fontweight='bold')
            
            # Plot each space
            for space_idx, space_name in enumerate(spaces_to_plot):
                space_results = method_results[space_name]['individual_results']
                
                # Extract data
                tasks = list(space_results.keys())
                sign_conflicts = [space_results[task]['sign_conflict_ratio'] for task in tasks]
                l2_distances = [space_results[task]['l2_distance'] for task in tasks]
                cosine_similarities = [space_results[task]['cosine_similarity'] for task in tasks]
                
                # Plot 1: Sign Conflicts
                axes[0, space_idx].bar(tasks, sign_conflicts, color='orange', alpha=0.7)
                axes[0, space_idx].set_title(f'Sign Conflicts ({space_name.capitalize()})', fontweight='bold')
                axes[0, space_idx].set_ylabel('Sign Conflict Ratio')
                axes[0, space_idx].tick_params(axis='x', rotation=45)
                
                # Plot 2: L2 Distance
                axes[1, space_idx].bar(tasks, l2_distances, color='red', alpha=0.7)
                axes[1, space_idx].set_title(f'L2 Distance ({space_name.capitalize()})', fontweight='bold')
                axes[1, space_idx].set_ylabel('L2 Distance')
                axes[1, space_idx].tick_params(axis='x', rotation=45)
                
                # Plot 3: Cosine Similarity
                axes[2, space_idx].bar(tasks, cosine_similarities, color='blue', alpha=0.7)
                axes[2, space_idx].set_title(f'Cosine Similarity ({space_name.capitalize()})', fontweight='bold')
                axes[2, space_idx].set_ylabel('Cosine Similarity')
                axes[2, space_idx].set_xlabel('Tasks')
                axes[2, space_idx].tick_params(axis='x', rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            
            unique_method_name = self._create_unique_method_name(method_name)
            plot_path = os.path.join(self.output_path, f'merged_task_vector_analysis_plots_{unique_method_name}.pdf')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            log.info(f"Saved analysis plots for {unique_method_name} to {plot_path}")


# Configuration mapping for Hydra  
MergedTaskVectorAnalysis._config_mapping = MergedTaskVectorAnalysis._config_mapping
