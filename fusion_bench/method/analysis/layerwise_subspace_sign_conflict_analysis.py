"""
Layer-wise Subspace Sign Conflict Analysis

This module analyzes how sign conflicts between task vectors change per layer when 
projected into FastFood subspaces. It provides insights into:

1. Magnitude of sign conflicts per layer in original vs subspace
2. Pattern shifts that explain interference between task vectors  
3. Layer-specific behavior of FastFood projection effects
4. Correlation between layer depth and subspace sign conflict changes

Key Features:
- Per-layer sign conflict analysis in original and FastFood subspace
- Visualization of sign conflict changes across network layers
- Statistical analysis of pattern shifts and interference effects
- Support for different projection ratios and FastFood configurations
- Comprehensive reporting with layer-wise breakdowns

Usage:
    from fusion_bench.method.analysis.layerwise_subspace_sign_conflict_analysis import LayerwiseSubspaceSignConflictAnalysis
    
    analyzer = LayerwiseSubspaceSignConflictAnalysis(
        proj_ratios=[0.1, 0.25, 0.5, 0.75],
        use_G=True,
        output_path="./layerwise_sign_analysis"
    )
    analyzer.run(modelpool)
"""

import hashlib
import logging
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
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
        x[..., 0, :], x[..., 1, :] = (a + b) / math.sqrt(2), (a - b) / math.sqrt(2)
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
class LayerwiseSubspaceSignConflictAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Analyze sign conflicts between task vectors per layer in FastFood subspace.
    
    This analysis reveals how FastFood projection affects sign conflicts between 
    task vectors at different network layers, providing insights into:
    
    1. **Layer-specific sign conflict patterns**: How conflicts change across layers
    2. **Subspace projection effects**: Original vs projected sign conflict ratios
    3. **Interference patterns**: Which layers show the most/least interference
    4. **Projection ratio sensitivity**: How different compression ratios affect conflicts
    
    The analysis computes pairwise sign conflicts between all task vectors for each layer,
    both in the original space and after FastFood projection, then analyzes the changes.
    
    Args:
        proj_ratios (List[float]): List of projection ratios to analyze (0.0 to 1.0)
        use_G (bool): Whether to use Gaussian scaling in FastFood transform
        trainable_only (bool): Whether to only analyze trainable parameters
        output_path (str, optional): Directory to save analysis results
        method_name (str, optional): Name for output files
        device (str): Device to run computations on
        min_layer_params (int): Minimum parameters per layer to include in analysis
        max_layers_analyze (int): Maximum number of layers to analyze (for efficiency)
        
    Outputs:
        - layerwise_sign_conflicts_original_{method_name}.csv: Original space conflicts per layer
        - layerwise_sign_conflicts_subspace_{method_name}.csv: Subspace conflicts per layer  
        - layerwise_sign_conflict_changes_{method_name}.csv: Change ratios per layer
        - layerwise_subspace_sign_analysis_{method_name}.pdf: Comprehensive visualization
        - layerwise_sign_conflict_summary_{method_name}.json: Statistical summary
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        proj_ratios: List[float] = [0.1, 0.25, 0.5, 0.75],
        use_G: bool = True,
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        device: str = "cuda",
        min_layer_params: int = 1000,
        max_layers_analyze: int = 50,
        analyze_component_types: bool = True,
        compute_statistical_significance: bool = True,
        create_detailed_plots: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proj_ratios = proj_ratios
        self.use_G = use_G
        self.trainable_only = trainable_only
        self._output_path = output_path
        self.method_name = method_name or "layerwise_subspace_sign"
        self.device = torch.device(device)
        self.min_layer_params = min_layer_params
        self.max_layers_analyze = max_layers_analyze
        self.analyze_component_types = analyze_component_types
        self.compute_statistical_significance = compute_statistical_significance
        self.create_detailed_plots = create_detailed_plots

    @property
    def output_path(self):
        if self._output_path:
            return Path(self._output_path)
        return Path(self.fabric.logger.log_dir) / "layerwise_subspace_sign_analysis"

    def get_state_dict(self, model: nn.Module) -> StateDictType:
        """Extract state dictionary from model."""
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()

    def get_task_vector(
        self, pretrained_model: nn.Module, finetuned_model: nn.Module
    ) -> StateDictType:
        """Compute task vector (finetuned - pretrained)."""
        return state_dict_sub(
            self.get_state_dict(finetuned_model),
            self.get_state_dict(pretrained_model),
        )

    def _extract_layer_info(self, param_name: str) -> Tuple[int, str]:
        """Extract layer number and component type from parameter name."""
        # Common patterns for transformer layers
        layer_patterns = [
            r"layers?\.(\d+)\.",
            r"blocks?\.(\d+)\.",
            r"encoder\.layers?\.(\d+)\.",
            r"decoder\.layers?\.(\d+)\.",
            r"transformer\.h\.(\d+)\.",
            r"model\.layers?\.(\d+)\.",
        ]
        
        import re
        layer_num = -1
        for pattern in layer_patterns:
            match = re.search(pattern, param_name)
            if match:
                layer_num = int(match.group(1))
                break
        
        # Extract component type
        component_type = "other"
        if "attention" in param_name or "attn" in param_name:
            component_type = "attention"
        elif "mlp" in param_name or "feed_forward" in param_name or "ffn" in param_name:
            component_type = "mlp"
        elif "norm" in param_name or "ln" in param_name or "layernorm" in param_name:
            component_type = "norm"
        elif "embed" in param_name:
            component_type = "embedding"
        elif "head" in param_name or "classifier" in param_name:
            component_type = "head"
        
        return layer_num, component_type

    def _group_parameters_by_layer(self, task_vector: StateDictType) -> Dict[int, Dict[str, torch.Tensor]]:
        """Group parameters by layer number."""
        layer_groups = defaultdict(dict)
        
        for param_name, param_tensor in task_vector.items():
            layer_num, component_type = self._extract_layer_info(param_name)
            
            # Skip parameters that don't belong to a specific layer
            if layer_num == -1:
                continue
                
            # Filter out very small layers
            if param_tensor.numel() < self.min_layer_params:
                continue
                
            layer_groups[layer_num][param_name] = param_tensor
        
        # Only keep layers with sufficient parameters
        filtered_groups = {}
        for layer_num, params in layer_groups.items():
            total_params = sum(p.numel() for p in params.values())
            if total_params >= self.min_layer_params:
                filtered_groups[layer_num] = params
        
        return filtered_groups

    def _compute_layer_sign_conflicts(self, task_vectors_by_layer: Dict[str, Dict[int, torch.Tensor]]) -> Dict[int, Dict[str, float]]:
        """Compute pairwise sign conflicts for each layer."""
        model_names = list(task_vectors_by_layer.keys())
        layer_results = {}
        
        # Get all layer numbers that exist across models
        all_layers = set()
        for model_vectors in task_vectors_by_layer.values():
            all_layers.update(model_vectors.keys())
        
        for layer_num in sorted(all_layers):
            # Skip if we have too many layers (for efficiency)
            if len(layer_results) >= self.max_layers_analyze:
                break
                
            # Get layer vectors for all models that have this layer
            layer_vectors = {}
            for model_name in model_names:
                if layer_num in task_vectors_by_layer[model_name]:
                    layer_vectors[model_name] = task_vectors_by_layer[model_name][layer_num]
            
            # Need at least 2 models for comparison
            if len(layer_vectors) < 2:
                continue
            
            # Compute pairwise sign conflicts
            conflicts = []
            model_pairs = []
            
            model_list = list(layer_vectors.keys())
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    model1, model2 = model_list[i], model_list[j]
                    vec1, vec2 = layer_vectors[model1], layer_vectors[model2]
                    
                    # Compute sign conflict
                    signs1 = torch.sign(vec1)
                    signs2 = torch.sign(vec2)
                    
                    # Only consider non-zero elements
                    nonzero_mask = (signs1 != 0) & (signs2 != 0)
                    if nonzero_mask.sum() > 0:
                        conflicts_mask = signs1[nonzero_mask] != signs2[nonzero_mask]
                        conflict_rate = conflicts_mask.float().mean().item()
                    else:
                        conflict_rate = 0.0
                    
                    conflicts.append(conflict_rate)
                    model_pairs.append(f"{model1}_vs_{model2}")
            
            if conflicts:
                layer_results[layer_num] = {
                    'mean_conflict_rate': np.mean(conflicts),
                    'std_conflict_rate': np.std(conflicts),
                    'min_conflict_rate': np.min(conflicts),
                    'max_conflict_rate': np.max(conflicts),
                    'num_comparisons': len(conflicts),
                    'total_parameters': sum(v.numel() for v in layer_vectors.values()),
                }
        
        return layer_results

    def _analyze_subspace_projection(
        self, 
        task_vectors_by_layer: Dict[str, Dict[int, torch.Tensor]], 
        proj_ratio: float
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
        """Analyze sign conflicts in original and subspace for each layer."""
        
        # Compute original space conflicts
        original_conflicts = self._compute_layer_sign_conflicts(task_vectors_by_layer)
        
        # Project to subspace and compute conflicts
        subspace_conflicts = {}
        
        for layer_num in original_conflicts.keys():
            # Get layer vectors for all models
            layer_vectors = {}
            for model_name, model_vectors in task_vectors_by_layer.items():
                if layer_num in model_vectors:
                    layer_vectors[model_name] = model_vectors[layer_num]
            
            if len(layer_vectors) < 2:
                continue
            
            # Determine global dimension for this layer
            total_dim = sum(v.numel() for v in layer_vectors.values()) // len(layer_vectors)
            proj_dim = max(1, int(total_dim * proj_ratio))
            
            # Create FastFood operator
            seed_key = f"layer_{layer_num}_proj_{proj_ratio}"
            fwd, lift = _fastfood_ops(
                global_dim=total_dim,
                proj_dim=proj_dim,
                seed_key=seed_key,
                device=self.device,
                use_G=self.use_G,
            )
            
            # Project vectors and compute conflicts
            projected_vectors = {}
            for model_name, vec in layer_vectors.items():
                vec_flat = vec.flatten().to(self.device)
                if vec_flat.shape[0] == total_dim:
                    projected_vectors[model_name] = fwd(vec_flat)
                
            if len(projected_vectors) >= 2:
                # Compute conflicts in subspace
                conflicts = []
                model_list = list(projected_vectors.keys())
                
                for i in range(len(model_list)):
                    for j in range(i + 1, len(model_list)):
                        model1, model2 = model_list[i], model_list[j]
                        vec1, vec2 = projected_vectors[model1], projected_vectors[model2]
                        
                        # Compute sign conflict
                        signs1 = torch.sign(vec1)
                        signs2 = torch.sign(vec2)
                        
                        # Only consider non-zero elements
                        nonzero_mask = (signs1 != 0) & (signs2 != 0)
                        if nonzero_mask.sum() > 0:
                            conflicts_mask = signs1[nonzero_mask] != signs2[nonzero_mask]
                            conflict_rate = conflicts_mask.float().mean().item()
                        else:
                            conflict_rate = 0.0
                        
                        conflicts.append(conflict_rate)
                
                if conflicts:
                    subspace_conflicts[layer_num] = {
                        'mean_conflict_rate': np.mean(conflicts),
                        'std_conflict_rate': np.std(conflicts),
                        'min_conflict_rate': np.min(conflicts),
                        'max_conflict_rate': np.max(conflicts),
                        'num_comparisons': len(conflicts),
                        'projection_dim': proj_dim,
                        'original_dim': total_dim,
                        'compression_ratio': proj_dim / total_dim,
                    }
        
        return original_conflicts, subspace_conflicts

    def _compute_change_statistics(
        self, 
        original_conflicts: Dict[int, Dict[str, float]], 
        subspace_conflicts: Dict[int, Dict[str, float]]
    ) -> Dict[int, Dict[str, float]]:
        """Compute statistics about how sign conflicts change from original to subspace."""
        change_stats = {}
        
        for layer_num in original_conflicts.keys():
            if layer_num in subspace_conflicts:
                orig = original_conflicts[layer_num]['mean_conflict_rate']
                proj = subspace_conflicts[layer_num]['mean_conflict_rate']
                
                # Compute various change metrics
                absolute_change = proj - orig
                relative_change = (proj - orig) / (orig + 1e-8)  # Add small epsilon
                amplification_ratio = proj / (orig + 1e-8)
                
                change_stats[layer_num] = {
                    'original_conflict_rate': orig,
                    'subspace_conflict_rate': proj,
                    'absolute_change': absolute_change,
                    'relative_change': relative_change,
                    'amplification_ratio': amplification_ratio,
                    'change_direction': 'increase' if absolute_change > 0 else 'decrease',
                    'compression_ratio': subspace_conflicts[layer_num]['compression_ratio'],
                }
        
        return change_stats

    def _create_comprehensive_plots(
        self, 
        all_results: Dict[float, Dict], 
        summary_stats: Dict[str, Any]
    ):
        """Create comprehensive visualization of layer-wise subspace sign conflict analysis."""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_path / f"layerwise_subspace_sign_analysis_{self.method_name}.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Plot 1: Original vs Subspace Conflict Rates by Layer
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Layer-wise Sign Conflict Analysis: Original vs FastFood Subspace', fontsize=16)
            
            # Collect data for plotting
            layer_nums = []
            proj_ratios_list = []
            original_rates = []
            subspace_rates = []
            change_ratios = []
            
            for proj_ratio, results in all_results.items():
                for layer_num, stats in results['change_statistics'].items():
                    layer_nums.append(layer_num)
                    proj_ratios_list.append(proj_ratio)
                    original_rates.append(stats['original_conflict_rate'])
                    subspace_rates.append(stats['subspace_conflict_rate'])
                    change_ratios.append(stats['relative_change'])
            
            # Convert to DataFrame for easier plotting
            plot_df = pd.DataFrame({
                'layer': layer_nums,
                'proj_ratio': proj_ratios_list,
                'original_rate': original_rates,
                'subspace_rate': subspace_rates,
                'relative_change': change_ratios,
            })
            
            # Plot 1a: Original conflict rates by layer
            ax1 = axes[0, 0]
            for proj_ratio in self.proj_ratios:
                subset = plot_df[plot_df['proj_ratio'] == proj_ratio]
                if not subset.empty:
                    ax1.plot(subset['layer'], subset['original_rate'], 
                            marker='o', alpha=0.7, label=f'Proj {proj_ratio}')
            ax1.set_xlabel('Layer Number')
            ax1.set_ylabel('Original Sign Conflict Rate')
            ax1.set_title('Original Space Sign Conflicts by Layer')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 1b: Subspace conflict rates by layer
            ax2 = axes[0, 1]
            for proj_ratio in self.proj_ratios:
                subset = plot_df[plot_df['proj_ratio'] == proj_ratio]
                if not subset.empty:
                    ax2.plot(subset['layer'], subset['subspace_rate'], 
                            marker='s', alpha=0.7, label=f'Proj {proj_ratio}')
            ax2.set_xlabel('Layer Number')
            ax2.set_ylabel('Subspace Sign Conflict Rate')
            ax2.set_title('FastFood Subspace Sign Conflicts by Layer')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 1c: Relative change by layer
            ax3 = axes[1, 0]
            for proj_ratio in self.proj_ratios:
                subset = plot_df[plot_df['proj_ratio'] == proj_ratio]
                if not subset.empty:
                    ax3.plot(subset['layer'], subset['relative_change'], 
                            marker='^', alpha=0.7, label=f'Proj {proj_ratio}')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Layer Number')
            ax3.set_ylabel('Relative Change in Sign Conflicts')
            ax3.set_title('Change in Sign Conflicts (Subspace vs Original)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 1d: Heatmap of relative changes
            ax4 = axes[1, 1]
            if not plot_df.empty:
                pivot_df = plot_df.pivot(index='layer', columns='proj_ratio', values='relative_change')
                sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax4)
                ax4.set_title('Heatmap: Relative Change by Layer and Projection Ratio')
                ax4.set_xlabel('Projection Ratio')
                ax4.set_ylabel('Layer Number')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Statistical Summary
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Statistical Summary: Sign Conflict Pattern Analysis', fontsize=16)
            
            # Plot 2a: Distribution of changes by projection ratio
            ax1 = axes[0, 0]
            for proj_ratio in self.proj_ratios:
                subset = plot_df[plot_df['proj_ratio'] == proj_ratio]
                if not subset.empty:
                    ax1.hist(subset['relative_change'], alpha=0.6, bins=20, 
                            label=f'Proj {proj_ratio}', density=True)
            ax1.set_xlabel('Relative Change in Sign Conflicts')
            ax1.set_ylabel('Density')
            ax1.set_title('Distribution of Sign Conflict Changes')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2b: Correlation between layer depth and change
            ax2 = axes[0, 1]
            for proj_ratio in self.proj_ratios:
                subset = plot_df[plot_df['proj_ratio'] == proj_ratio]
                if not subset.empty:
                    ax2.scatter(subset['layer'], subset['relative_change'], 
                               alpha=0.6, label=f'Proj {proj_ratio}')
            
            # Add trend line
            if not plot_df.empty:
                z = np.polyfit(plot_df['layer'], plot_df['relative_change'], 1)
                p = np.poly1d(z)
                ax2.plot(plot_df['layer'].unique(), p(plot_df['layer'].unique()), 
                        "r--", alpha=0.8, label='Trend')
            
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Layer Number')
            ax2.set_ylabel('Relative Change')
            ax2.set_title('Layer Depth vs Sign Conflict Change')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 2c: Original vs Subspace scatter
            ax3 = axes[1, 0]
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.proj_ratios)))
            for i, proj_ratio in enumerate(self.proj_ratios):
                subset = plot_df[plot_df['proj_ratio'] == proj_ratio]
                if not subset.empty:
                    ax3.scatter(subset['original_rate'], subset['subspace_rate'], 
                               alpha=0.6, color=colors[i], label=f'Proj {proj_ratio}')
            
            # Add diagonal line (no change)
            max_val = max(plot_df['original_rate'].max(), plot_df['subspace_rate'].max())
            ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No Change')
            ax3.set_xlabel('Original Sign Conflict Rate')
            ax3.set_ylabel('Subspace Sign Conflict Rate')
            ax3.set_title('Original vs Subspace Sign Conflicts')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 2d: Summary statistics text
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create summary text
            summary_text = "Summary Statistics:\n\n"
            if 'overall_stats' in summary_stats:
                stats = summary_stats['overall_stats']
                summary_text += f"Total Layers Analyzed: {stats.get('total_layers', 0)}\n"
                summary_text += f"Mean Original Conflict: {stats.get('mean_original_conflict', 0):.4f}\n"
                summary_text += f"Mean Subspace Conflict: {stats.get('mean_subspace_conflict', 0):.4f}\n"
                summary_text += f"Mean Relative Change: {stats.get('mean_relative_change', 0):.4f}\n\n"
            
            summary_text += "Key Findings:\n"
            for proj_ratio in self.proj_ratios:
                if proj_ratio in summary_stats.get('by_projection_ratio', {}):
                    ratio_stats = summary_stats['by_projection_ratio'][proj_ratio]
                    summary_text += f"• Proj {proj_ratio}: {ratio_stats.get('mean_change', 0):.3f} avg change\n"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Additional detailed plots if requested
            if self.create_detailed_plots and len(all_results) > 1:
                self._create_detailed_comparison_plots(pdf, all_results, plot_df)
        
        log.info(f"Comprehensive plots saved to: {pdf_path}")

    def _create_detailed_comparison_plots(self, pdf: PdfPages, all_results: Dict, plot_df: pd.DataFrame):
        """Create additional detailed comparison plots."""
        
        # Plot 3: Layer-wise comparison across projection ratios
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create box plots for each projection ratio
        proj_data = []
        proj_labels = []
        
        for proj_ratio in self.proj_ratios:
            subset = plot_df[plot_df['proj_ratio'] == proj_ratio]
            if not subset.empty:
                proj_data.append(subset['relative_change'].values)
                proj_labels.append(f'Proj {proj_ratio}')
        
        if proj_data:
            bp = ax.boxplot(proj_data, labels=proj_labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(proj_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Change')
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Relative Change in Sign Conflicts')
        ax.set_title('Distribution of Sign Conflict Changes by Projection Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """Run the layer-wise subspace sign conflict analysis."""
        
        log.info(f"Starting layer-wise subspace sign conflict analysis with {len(self.proj_ratios)} projection ratios")
        
        # Get models
        pretrained_model = modelpool.load_model("_pretrained_")
        model_names = [name for name in modelpool.model_names if name != "_pretrained_"]
        
        if len(model_names) < 2:
            raise ValueError("Need at least 2 fine-tuned models for sign conflict analysis")
        
        log.info(f"Analyzing {len(model_names)} fine-tuned models: {model_names}")
        
        # Compute task vectors for each model
        task_vectors = {}
        for model_name in tqdm(model_names, desc="Computing task vectors"):
            finetuned_model = modelpool.load_model(model_name)
            task_vector = self.get_task_vector(pretrained_model, finetuned_model)
            task_vectors[model_name] = task_vector
        
        # Group task vectors by layer
        task_vectors_by_layer = {}
        for model_name, task_vector in task_vectors.items():
            layer_groups = self._group_parameters_by_layer(task_vector)
            # Convert to flattened tensors per layer
            layer_tensors = {}
            for layer_num, layer_params in layer_groups.items():
                # Concatenate all parameters for this layer
                layer_tensor = torch.cat([p.flatten() for p in layer_params.values()])
                layer_tensors[layer_num] = layer_tensor
            task_vectors_by_layer[model_name] = layer_tensors
        
        log.info(f"Grouped parameters into layers. Found {len(set().union(*[layers.keys() for layers in task_vectors_by_layer.values()]))} unique layers")
        
        # Analyze for each projection ratio
        all_results = {}
        for proj_ratio in tqdm(self.proj_ratios, desc="Analyzing projection ratios"):
            log.info(f"Analyzing projection ratio: {proj_ratio}")
            
            original_conflicts, subspace_conflicts = self._analyze_subspace_projection(
                task_vectors_by_layer, proj_ratio
            )
            
            change_statistics = self._compute_change_statistics(original_conflicts, subspace_conflicts)
            
            all_results[proj_ratio] = {
                'original_conflicts': original_conflicts,
                'subspace_conflicts': subspace_conflicts,
                'change_statistics': change_statistics,
            }
            
            log.info(f"  Projection ratio {proj_ratio}: analyzed {len(change_statistics)} layers")
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(all_results)
        
        # Save results
        self._save_results(all_results, summary_stats)
        
        # Create visualizations
        if self.create_detailed_plots:
            self._create_comprehensive_plots(all_results, summary_stats)
        
        log.info("Layer-wise subspace sign conflict analysis completed")
        
        return pretrained_model

    def _compute_summary_statistics(self, all_results: Dict[float, Dict]) -> Dict[str, Any]:
        """Compute summary statistics across all projection ratios and layers."""
        
        summary_stats = {
            'by_projection_ratio': {},
            'overall_stats': {},
            'layer_patterns': {},
        }
        
        all_changes = []
        all_original = []
        all_subspace = []
        total_layers = 0
        
        for proj_ratio, results in all_results.items():
            change_stats = results['change_statistics']
            
            if change_stats:
                changes = [stats['relative_change'] for stats in change_stats.values()]
                original_rates = [stats['original_conflict_rate'] for stats in change_stats.values()]
                subspace_rates = [stats['subspace_conflict_rate'] for stats in change_stats.values()]
                
                summary_stats['by_projection_ratio'][proj_ratio] = {
                    'num_layers': len(changes),
                    'mean_change': np.mean(changes),
                    'std_change': np.std(changes),
                    'median_change': np.median(changes),
                    'min_change': np.min(changes),
                    'max_change': np.max(changes),
                    'layers_increased': sum(1 for c in changes if c > 0),
                    'layers_decreased': sum(1 for c in changes if c < 0),
                }
                
                all_changes.extend(changes)
                all_original.extend(original_rates)
                all_subspace.extend(subspace_rates)
                total_layers = max(total_layers, len(changes))
        
        # Overall statistics
        if all_changes:
            summary_stats['overall_stats'] = {
                'total_layers': total_layers,
                'mean_original_conflict': np.mean(all_original),
                'mean_subspace_conflict': np.mean(all_subspace),
                'mean_relative_change': np.mean(all_changes),
                'std_relative_change': np.std(all_changes),
                'fraction_increased': sum(1 for c in all_changes if c > 0) / len(all_changes),
                'fraction_decreased': sum(1 for c in all_changes if c < 0) / len(all_changes),
            }
        
        return summary_stats

    def _save_results(self, all_results: Dict[float, Dict], summary_stats: Dict[str, Any]):
        """Save analysis results to files."""
        
        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results for each projection ratio
        for proj_ratio, results in all_results.items():
            # Original space conflicts
            original_df = pd.DataFrame(results['original_conflicts']).T
            original_df.to_csv(
                output_path / f"layerwise_sign_conflicts_original_proj{proj_ratio}_{self.method_name}.csv"
            )
            
            # Subspace conflicts
            subspace_df = pd.DataFrame(results['subspace_conflicts']).T
            subspace_df.to_csv(
                output_path / f"layerwise_sign_conflicts_subspace_proj{proj_ratio}_{self.method_name}.csv"
            )
            
            # Change statistics
            change_df = pd.DataFrame(results['change_statistics']).T
            change_df.to_csv(
                output_path / f"layerwise_sign_conflict_changes_proj{proj_ratio}_{self.method_name}.csv"
            )
        
        # Save summary statistics
        import json
        with open(output_path / f"layerwise_sign_conflict_summary_{self.method_name}.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        log.info(f"Results saved to: {output_path}")


# Configuration mapping for Hydra
LayerwiseSubspaceSignConflictAnalysis._config_mapping = LayerwiseSubspaceSignConflictAnalysis._config_mapping