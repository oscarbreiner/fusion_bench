"""
Task Vector Layer-wise Analysis

This module provides layer-wise analysis of task vectors to understand how different
merging methods affect different layers of neural networks. It computes sign conflicts,
cosine similarity, and L2 distance across different layers.

Key Features:
- Layer-wise sign conflict analysis
- Layer-wise cosine similarity analysis  
- Layer-wise L2 distance analysis
- Visualization with layer indices on x-axis and metric values on y-axis
- Support for both standard and Fastfood merging methods
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

log = logging.getLogger(__name__)


@auto_register_config
class TaskVectorLayerAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Layer-wise analysis of task vectors showing how metrics vary across network layers.
    
    This algorithm analyzes task vectors at the layer level rather than globally,
    providing insights into which layers show more conflicts or similarities between tasks.
    
    Key Metrics (computed per layer):
    1. **Sign Conflicts**: Ratio of parameters with different signs between task vectors
    2. **Cosine Similarity**: Angular similarity between task vectors
    3. **L2 Distance**: Euclidean distance between task vectors
    
    Outputs three plots with layer indices on x-axis and metric values on y-axis:
    - Sign conflicts across layers
    - Cosine similarity across layers  
    - L2 distance across layers
    
    Args:
        trainable_only (bool): Whether to only analyze trainable parameters
        output_path (str, optional): Directory to save analysis results
        method_name (str, optional): Method name for file naming
        device (str): Device to run computations on
        
    Outputs:
        - task_vector_layer_analysis_{method_name}.pdf: Layer-wise plots
        - task_vector_layer_analysis_{method_name}.csv: Detailed layer results
        - task_vector_layer_analysis_summary_{method_name}.csv: Summary statistics
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        device: str = "cuda",
        # Enhanced analysis options
        analyze_attention_mlp_separately: bool = True,
        analyze_layer_groups: bool = True,
        layer_group_strategy: str = "thirds",
        custom_layer_groups: Optional[List[List[int]]] = None,
        compute_layer_importance: bool = True,
        compute_gradient_flow_proxy: bool = True,
        compute_cross_layer_correlations: bool = True,
        compute_capacity_metrics: bool = True,
        attention_head_analysis: bool = True,
        analyze_qkv_separately: bool = True,
        position_encoding_analysis: bool = True,
        compute_statistical_tests: bool = True,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000,
        create_detailed_plots: bool = True,
        plot_correlation_matrices: bool = True,
        plot_importance_distributions: bool = True,
        plot_gradient_flow: bool = True,
        save_individual_layer_plots: bool = True,
        max_layers_for_detailed_analysis: int = 48,
        subsample_parameters: bool = True,
        max_parameters_per_layer: int = 10000,
        save_raw_data: bool = True,
        create_summary_report: bool = True,
        fastfood_aware_analysis: bool = True,
        analyze_subspace_effects_per_layer: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trainable_only = trainable_only
        self._output_path = output_path
        self.method_name = method_name or "enhanced_layer"
        self.device = torch.device(device)
        
        # Enhanced analysis parameters
        self.analyze_attention_mlp_separately = analyze_attention_mlp_separately
        self.analyze_layer_groups = analyze_layer_groups
        self.layer_group_strategy = layer_group_strategy
        self.custom_layer_groups = custom_layer_groups
        self.compute_layer_importance = compute_layer_importance
        self.compute_gradient_flow_proxy = compute_gradient_flow_proxy
        self.compute_cross_layer_correlations = compute_cross_layer_correlations
        self.compute_capacity_metrics = compute_capacity_metrics
        self.attention_head_analysis = attention_head_analysis
        self.analyze_qkv_separately = analyze_qkv_separately
        self.position_encoding_analysis = position_encoding_analysis
        self.compute_statistical_tests = compute_statistical_tests
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.create_detailed_plots = create_detailed_plots
        self.plot_correlation_matrices = plot_correlation_matrices
        self.plot_importance_distributions = plot_importance_distributions
        self.plot_gradient_flow = plot_gradient_flow
        self.save_individual_layer_plots = save_individual_layer_plots
        self.max_layers_for_detailed_analysis = max_layers_for_detailed_analysis
        self.subsample_parameters = subsample_parameters
        self.max_parameters_per_layer = max_parameters_per_layer
        self.save_raw_data = save_raw_data
        self.create_summary_report = create_summary_report
        self.fastfood_aware_analysis = fastfood_aware_analysis
        self.analyze_subspace_effects_per_layer = analyze_subspace_effects_per_layer

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    def _get_model_state_dict(self, model: nn.Module) -> StateDictType:
        """Get state dict from model, optionally filtering to trainable parameters only."""
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()

    def _extract_layer_number(self, param_name: str) -> int:
        """
        Extract layer number from parameter name for CLIP models.
        
        Examples:
        - 'visual.transformer.resblocks.0.attn.in_proj_weight' -> 0
        - 'visual.transformer.resblocks.11.mlp.c_fc.weight' -> 11
        - 'visual.class_embedding' -> -1 (non-layer parameter)
        """
        # Pattern for CLIP transformer blocks
        match = re.search(r'\.resblocks\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        
        # Pattern for other transformer architectures
        match = re.search(r'\.layers\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        
        # Pattern for BERT-style models  
        match = re.search(r'\.layer\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        
        # Non-layer parameters (embeddings, final layers, etc.)
        return -1

    def _extract_component_type(self, param_name: str) -> str:
        """
        Extract component type from parameter name (attention, mlp, etc.).
        
        Returns:
            Component type: 'attention', 'mlp', 'norm', 'embedding', 'head', 'other'
        """
        param_lower = param_name.lower()
        
        if any(x in param_lower for x in ['attn', 'attention', 'self_attn', 'cross_attn']):
            if any(x in param_lower for x in ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']):
                return f"attention_{param_lower.split('.')[-2]}" if '.' in param_lower else "attention_qkv"
            elif 'out_proj' in param_lower or 'o_proj' in param_lower:
                return "attention_output"
            else:
                return "attention"
        elif any(x in param_lower for x in ['mlp', 'ffn', 'feed_forward', 'c_fc', 'c_proj']):
            return "mlp"
        elif any(x in param_lower for x in ['norm', 'ln_', 'layer_norm', 'layernorm']):
            return "normalization"
        elif any(x in param_lower for x in ['embed', 'position', 'pos_embed', 'class_embedding']):
            return "embedding"
        elif any(x in param_lower for x in ['head', 'classifier', 'final']):
            return "classification_head"
        else:
            return "other"

    def _compute_layer_importance_score(self, layer_params: Dict[str, torch.Tensor]) -> float:
        """
        Compute importance score for a layer based on parameter magnitudes and variance.
        
        This helps identify which layers are most critical for task-specific adaptations.
        """
        if not layer_params:
            return 0.0
        
        total_magnitude = 0.0
        total_variance = 0.0
        total_params = 0
        
        for param_tensor in layer_params.values():
            # L2 norm (magnitude)
            magnitude = torch.norm(param_tensor).item()
            total_magnitude += magnitude
            
            # Parameter variance (diversity of weights)
            variance = torch.var(param_tensor).item()
            total_variance += variance
            
            total_params += param_tensor.numel()
        
        # Combine magnitude and variance, normalized by parameter count
        if total_params > 0:
            importance = (total_magnitude + total_variance) / total_params
        else:
            importance = 0.0
        
        return importance

    def _compute_gradient_flow_proxy(self, layer_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute proxy metrics for gradient flow characteristics.
        
        These metrics help understand optimization landscapes and learning dynamics.
        """
        if not layer_params:
            return {"gradient_norm": 0.0, "hessian_trace_approx": 0.0, "condition_number_approx": 0.0}
        
        # Flatten all parameters in the layer
        all_params = torch.cat([p.flatten() for p in layer_params.values()])
        
        # Gradient norm proxy (using parameter magnitudes as proxy)
        gradient_norm = torch.norm(all_params).item()
        
        # Hessian trace approximation (using parameter variance)
        hessian_trace_approx = torch.var(all_params).item()
        
        # Condition number approximation (ratio of max to min singular values proxy)
        param_abs = torch.abs(all_params)
        param_nonzero = param_abs[param_abs > 1e-8]
        if len(param_nonzero) > 0:
            condition_number_approx = param_nonzero.max().item() / param_nonzero.min().item()
        else:
            condition_number_approx = 1.0
        
        return {
            "gradient_norm": gradient_norm,
            "hessian_trace_approx": hessian_trace_approx,
            "condition_number_approx": condition_number_approx
        }

    def _compute_layer_capacity_metrics(self, layer_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute metrics related to layer capacity and utilization.
        
        These metrics help understand how efficiently each layer is being used.
        """
        if not layer_params:
            return {"effective_rank": 0.0, "sparsity": 0.0, "weight_distribution_entropy": 0.0}
        
        # Flatten all parameters
        all_params = torch.cat([p.flatten() for p in layer_params.values()])
        
        # Sparsity (fraction of near-zero weights)
        sparsity = (torch.abs(all_params) < 1e-6).float().mean().item()
        
        # Weight distribution entropy (diversity of weight values)
        param_abs = torch.abs(all_params) + 1e-8
        param_probs = param_abs / torch.sum(param_abs)
        entropy = -torch.sum(param_probs * torch.log(param_probs)).item()
        
        # Effective rank approximation (for matrices, use spectral properties)
        effective_rank = 0.0
        matrix_count = 0
        
        for param_tensor in layer_params.values():
            if param_tensor.dim() >= 2:
                # Reshape to matrix if needed
                matrix = param_tensor.view(param_tensor.size(0), -1)
                
                # SVD for effective rank
                try:
                    U, S, V = torch.svd(matrix.float())
                    # Effective rank using 90% of spectral energy
                    total_energy = torch.sum(S**2)
                    cumsum_energy = torch.cumsum(S**2, dim=0)
                    effective_rank += torch.sum(cumsum_energy < 0.9 * total_energy).item() + 1
                    matrix_count += 1
                except:
                    # Fallback for numerical issues
                    effective_rank += min(matrix.size())
                    matrix_count += 1
        
        if matrix_count > 0:
            effective_rank /= matrix_count
        
        return {
            "effective_rank": effective_rank,
            "sparsity": sparsity,
            "weight_distribution_entropy": entropy
        }

    def _create_layer_groups(self, layer_indices: List[int]) -> Dict[str, List[int]]:
        """Create meaningful layer groupings for analysis."""
        if not layer_indices:
            return {}
        
        layer_indices = sorted(layer_indices)
        num_layers = len(layer_indices)
        
        if self.layer_group_strategy == "thirds":
            third = num_layers // 3
            groups = {
                "early": layer_indices[:third],
                "middle": layer_indices[third:2*third],
                "late": layer_indices[2*third:]
            }
        elif self.layer_group_strategy == "quartiles":
            quarter = num_layers // 4
            groups = {
                "first_quarter": layer_indices[:quarter],
                "second_quarter": layer_indices[quarter:2*quarter],
                "third_quarter": layer_indices[2*quarter:3*quarter],
                "fourth_quarter": layer_indices[3*quarter:]
            }
        elif self.layer_group_strategy == "custom" and self.custom_layer_groups:
            groups = {}
            for i, group in enumerate(self.custom_layer_groups):
                groups[f"custom_group_{i}"] = [idx for idx in group if idx in layer_indices]
        else:
            # Default: just use all layers as one group
            groups = {"all": layer_indices}
        
        return groups

    def _group_params_by_layer(self, state_dict: StateDictType) -> Dict[int, Dict[str, torch.Tensor]]:
        """Group state dict parameters by layer number."""
        layers = {}
        
        for param_name, param_tensor in state_dict.items():
            layer_idx = self._extract_layer_number(param_name)
            
            if layer_idx not in layers:
                layers[layer_idx] = {}
            
            layers[layer_idx][param_name] = param_tensor
        
        return layers

    def _compute_task_vector_by_layer(
        self, 
        pretrained_model: nn.Module, 
        finetuned_model: nn.Module
    ) -> Dict[int, torch.Tensor]:
        """Compute task vectors grouped by layer."""
        
        # Get state dicts
        pretrained_state = self._get_model_state_dict(pretrained_model)
        finetuned_state = self._get_model_state_dict(finetuned_model)
        
        # Compute task vector
        task_vector_state = state_dict_sub(finetuned_state, pretrained_state)
        
        # Group by layer
        layer_groups = self._group_params_by_layer(task_vector_state)
        
        # Flatten each layer's parameters into a single vector
        layer_vectors = {}
        for layer_idx, layer_params in layer_groups.items():
            if layer_params:  # Skip empty layers
                vectors = []
                for param_tensor in layer_params.values():
                    vectors.append(param_tensor.flatten())
                layer_vectors[layer_idx] = torch.cat(vectors)
        
        return layer_vectors

    def _compute_layer_metrics(
        self, 
        task_vectors_by_layer: Dict[str, Dict[int, torch.Tensor]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute pairwise metrics between all task vectors for each layer.
        
        Args:
            task_vectors_by_layer: Dict mapping task_name -> layer_idx -> vector
            
        Returns:
            Dict mapping layer_idx -> metrics dict
        """
        
        # Get all layer indices that exist across all tasks
        all_layers = set()
        for task_vectors in task_vectors_by_layer.values():
            all_layers.update(task_vectors.keys())
        
        # Remove non-layer parameters (layer -1)
        layer_indices = sorted([layer for layer in all_layers if layer >= 0])
        
        results = {}
        
        for layer_idx in tqdm(layer_indices, desc="Analyzing layers"):
            # Get task vectors for this layer from all tasks that have it
            layer_task_vectors = {}
            for task_name, task_layers in task_vectors_by_layer.items():
                if layer_idx in task_layers:
                    layer_task_vectors[task_name] = task_layers[layer_idx]
            
            if len(layer_task_vectors) < 2:
                log.warning(f"Layer {layer_idx} has fewer than 2 tasks, skipping")
                continue
            
            # Compute pairwise metrics for this layer
            task_names = list(layer_task_vectors.keys())
            num_tasks = len(task_names)
            
            sign_conflicts = []
            cosine_similarities = []
            l2_distances = []
            
            for i in range(num_tasks):
                for j in range(i + 1, num_tasks):
                    vec1 = layer_task_vectors[task_names[i]]
                    vec2 = layer_task_vectors[task_names[j]]
                    
                    # Ensure same size
                    if vec1.shape != vec2.shape:
                        log.warning(f"Layer {layer_idx}: Size mismatch between {task_names[i]} and {task_names[j]}")
                        continue
                    
                    # Sign conflicts
                    sign_conflicts.append(self._compute_sign_conflicts(vec1, vec2))
                    
                    # Cosine similarity
                    cosine_similarities.append(self._compute_cosine_similarity(vec1, vec2))
                    
                    # L2 distance
                    l2_distances.append(self._compute_l2_distance(vec1, vec2))
            
            # Aggregate metrics for this layer
            results[layer_idx] = {
                'avg_sign_conflicts': np.mean(sign_conflicts) if sign_conflicts else 0.0,
                'std_sign_conflicts': np.std(sign_conflicts) if sign_conflicts else 0.0,
                'avg_cosine_similarity': np.mean(cosine_similarities) if cosine_similarities else 0.0,
                'std_cosine_similarity': np.std(cosine_similarities) if cosine_similarities else 0.0,
                'avg_l2_distance': np.mean(l2_distances) if l2_distances else 0.0,
                'std_l2_distance': np.std(l2_distances) if l2_distances else 0.0,
                'num_task_pairs': len(sign_conflicts),
                'num_parameters': vec1.numel() if layer_task_vectors else 0,
            }
        
        return results

    def _compute_sign_conflicts(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute ratio of elements with different signs."""
        if vec1.numel() == 0:
            return 0.0
        
        # Only consider non-zero elements
        mask = (vec1 != 0) & (vec2 != 0)
        if mask.sum() == 0:
            return 0.0
        
        conflicts = ((vec1[mask] > 0) != (vec2[mask] > 0)).sum().item()
        return conflicts / mask.sum().item()

    def _compute_cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute cosine similarity."""
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return torch.dot(vec1, vec2).item() / (norm1.item() * norm2.item())

    def _compute_l2_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute L2 distance."""
        return torch.norm(vec1 - vec2).item()

    def _create_layer_plots(self, layer_results: Dict[int, Dict[str, float]]):
        """Create and save layer-wise analysis plots."""
        
        if not layer_results:
            log.warning("No layer results to plot")
            return
        
        # Extract data for plotting
        layer_indices = sorted(layer_results.keys())
        sign_conflicts = [layer_results[idx]['avg_sign_conflicts'] for idx in layer_indices]
        sign_conflicts_std = [layer_results[idx]['std_sign_conflicts'] for idx in layer_indices]
        
        cosine_similarities = [layer_results[idx]['avg_cosine_similarity'] for idx in layer_indices]
        cosine_similarities_std = [layer_results[idx]['std_cosine_similarity'] for idx in layer_indices]
        
        l2_distances = [layer_results[idx]['avg_l2_distance'] for idx in layer_indices]
        l2_distances_std = [layer_results[idx]['std_l2_distance'] for idx in layer_indices]
        
        # Create three-panel plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Sign Conflicts
        axes[0].errorbar(layer_indices, sign_conflicts, yerr=sign_conflicts_std, 
                        marker='o', linestyle='-', color='orange', capsize=5)
        axes[0].set_title('Sign Conflicts Across Layers', fontweight='bold')
        axes[0].set_ylabel('Sign Conflict Ratio')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Cosine Similarity
        axes[1].errorbar(layer_indices, cosine_similarities, yerr=cosine_similarities_std,
                        marker='o', linestyle='-', color='blue', capsize=5)
        axes[1].set_title('Cosine Similarity Across Layers', fontweight='bold')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-1, 1)
        
        # Plot 3: L2 Distance
        axes[2].errorbar(layer_indices, l2_distances, yerr=l2_distances_std,
                        marker='o', linestyle='-', color='red', capsize=5)
        axes[2].set_title('L2 Distance Across Layers', fontweight='bold')
        axes[2].set_ylabel('L2 Distance')
        axes[2].set_xlabel('Layer Index')
        axes[2].grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'Task Vector Layer-wise Analysis - {self.method_name}', 
                    fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plot
        plot_path = os.path.join(self.output_path, f'task_vector_layer_analysis_{self.method_name}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        log.info(f"Saved layer analysis plots to {plot_path}")

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Execute layer-wise task vector analysis.
        
        Args:
            modelpool: Model pool containing pretrained and fine-tuned models
            
        Returns:
            The pretrained model from the model pool
        """
        log.info("Starting layer-wise task vector analysis")
        log.info(f"Method: {self.method_name}")
        log.info(f"Trainable parameters only: {self.trainable_only}")
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load pretrained model
        pretrained_model = modelpool.load_pretrained_model()
        
        # Compute task vectors by layer for each fine-tuned model
        task_vectors_by_layer = {}
        
        for name, finetuned_model in tqdm(modelpool.named_models(), 
                                        desc="Computing task vectors by layer"):
            layer_vectors = self._compute_task_vector_by_layer(pretrained_model, finetuned_model)
            task_vectors_by_layer[name] = layer_vectors
            log.info(f"Computed task vectors for {name}: {len(layer_vectors)} layers")
        
        # Compute layer-wise metrics
        log.info("Computing layer-wise metrics")
        layer_results = self._compute_layer_metrics(task_vectors_by_layer)
        
        # Save detailed results
        if layer_results:
            # Create detailed DataFrame
            detailed_data = []
            for layer_idx, metrics in layer_results.items():
                detailed_data.append({
                    'layer_index': layer_idx,
                    'avg_sign_conflicts': metrics['avg_sign_conflicts'],
                    'std_sign_conflicts': metrics['std_sign_conflicts'],
                    'avg_cosine_similarity': metrics['avg_cosine_similarity'],
                    'std_cosine_similarity': metrics['std_cosine_similarity'],
                    'avg_l2_distance': metrics['avg_l2_distance'],
                    'std_l2_distance': metrics['std_l2_distance'],
                    'num_task_pairs': metrics['num_task_pairs'],
                    'num_parameters': metrics['num_parameters']
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_path = os.path.join(self.output_path, 
                                       f'task_vector_layer_analysis_{self.method_name}.csv')
            detailed_df.to_csv(detailed_path, index=False)
            log.info(f"Saved detailed layer analysis to {detailed_path}")
            
            # Create summary statistics
            summary_data = {
                'method_name': self.method_name,
                'total_layers_analyzed': len(layer_results),
                'avg_sign_conflicts_across_layers': detailed_df['avg_sign_conflicts'].mean(),
                'std_sign_conflicts_across_layers': detailed_df['avg_sign_conflicts'].std(),
                'avg_cosine_similarity_across_layers': detailed_df['avg_cosine_similarity'].mean(),
                'std_cosine_similarity_across_layers': detailed_df['avg_cosine_similarity'].std(),
                'avg_l2_distance_across_layers': detailed_df['avg_l2_distance'].mean(),
                'std_l2_distance_across_layers': detailed_df['avg_l2_distance'].std(),
                'total_parameters_analyzed': detailed_df['num_parameters'].sum()
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_path = os.path.join(self.output_path, 
                                      f'task_vector_layer_analysis_summary_{self.method_name}.csv')
            summary_df.to_csv(summary_path, index=False)
            log.info(f"Saved summary analysis to {summary_path}")
            
            # Create visualization
            self._create_layer_plots(layer_results)
            
            # Print summary
            log.info("Layer-wise Analysis Summary:")
            log.info(f"  - Total layers analyzed: {len(layer_results)}")
            log.info(f"  - Average sign conflicts: {summary_data['avg_sign_conflicts_across_layers']:.4f}")
            log.info(f"  - Average cosine similarity: {summary_data['avg_cosine_similarity_across_layers']:.4f}")
            log.info(f"  - Average L2 distance: {summary_data['avg_l2_distance_across_layers']:.4f}")
        
        else:
            log.warning("No layer results computed - check model architecture compatibility")
        
        log.info("Layer-wise task vector analysis completed successfully")
        return pretrained_model


# Configuration mapping for Hydra
TaskVectorLayerAnalysis._config_mapping = TaskVectorLayerAnalysis._config_mapping
