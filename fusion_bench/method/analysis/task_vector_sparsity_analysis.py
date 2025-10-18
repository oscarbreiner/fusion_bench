"""
Task Vector Sparsity Analysis

This module analyzes the sparsity patterns of task vectors at the layer level, providing insights into:

1. Sparsity scores per layer per task vector (percentage of near-zero parameters)
2. Layer-wise sparsity distribution patterns across different tasks
3. Task-specific sparsity characteristics and their relationship to performance
4. Sparsity evolution across network depth (early vs late layers)
5. Component-wise sparsity analysis (attention vs MLP in transformers)

Key Features:
- Per-layer sparsity analysis for each task vector
- Multiple sparsity metrics: L0 (exact zeros), L0.5, L1-based, and adaptive threshold
- Visualization of sparsity patterns across layers and tasks
- Statistical analysis of sparsity distributions
- Support for different sparsity threshold strategies
- Comprehensive reporting with layer-wise and task-wise breakdowns

The sparsity analysis helps understand:
- Which layers are most affected by task-specific fine-tuning
- How different tasks utilize network capacity differently  
- Whether certain layers become specialized vs generalized across tasks
- The relationship between sparsity patterns and task performance

Usage:
    from fusion_bench.method.analysis.task_vector_sparsity_analysis import TaskVectorSparsityAnalysis
    
    analyzer = TaskVectorSparsityAnalysis(
        sparsity_thresholds=[1e-6, 1e-5, 1e-4, 1e-3],
        analyze_components_separately=True,
        output_path="./sparsity_analysis"
    )
    analyzer.run(modelpool)
"""

import logging
import math
import re
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


@auto_register_config
class TaskVectorSparsityAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Comprehensive sparsity analysis of task vectors at the layer level.
    
    This algorithm analyzes the sparsity patterns of task vectors (parameter differences
    between fine-tuned and pretrained models) across different layers of neural networks.
    
    Key Metrics (computed per layer per task):
    1. **L0 Sparsity**: Exact percentage of zero parameters
    2. **Threshold Sparsity**: Percentage of parameters below various thresholds
    3. **Adaptive Sparsity**: Data-driven threshold based on parameter distribution
    4. **Component Sparsity**: Separate analysis for attention vs MLP components
    
    Outputs:
    - Sparsity heatmaps showing layer vs task patterns
    - Distribution plots of sparsity across layers and tasks
    - Component-wise sparsity analysis for transformer architectures
    - Statistical summaries and correlation analysis
    - CSV files with detailed per-layer per-task sparsity scores
    
    Args:
        sparsity_thresholds (List[float]): Thresholds for near-zero parameter detection
        analyze_components_separately (bool): Whether to analyze attention/MLP separately  
        adaptive_threshold_method (str): Method for adaptive threshold ("percentile", "std", "iqr")
        adaptive_percentile (float): Percentile for adaptive thresholding (if using percentile method)
        trainable_only (bool): Whether to only analyze trainable parameters
        output_path (str, optional): Directory to save analysis results
        device (str): Device to run computations on
        create_visualizations (bool): Whether to create plots and visualizations
        
    Outputs:
        - task_vector_sparsity_heatmap.pdf: Layer vs task sparsity heatmap
        - task_vector_sparsity_distributions.pdf: Sparsity distribution plots
        - task_vector_sparsity_components.pdf: Component-wise analysis (if enabled)
        - task_vector_sparsity_detailed.csv: Per-layer per-task sparsity scores
        - task_vector_sparsity_summary.csv: Summary statistics
        - task_vector_sparsity_report.txt: Human-readable analysis report
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        sparsity_thresholds: List[float] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        analyze_components_separately: bool = True,
        adaptive_threshold_method: str = "percentile",  # "percentile", "std", "iqr"
        adaptive_percentile: float = 5.0,  # 5th percentile for adaptive thresholding
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        device: str = "cuda",
        create_visualizations: bool = True,
        save_raw_data: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sparsity_thresholds = sparsity_thresholds
        self.analyze_components_separately = analyze_components_separately
        self.adaptive_threshold_method = adaptive_threshold_method
        self.adaptive_percentile = adaptive_percentile
        self.trainable_only = trainable_only
        self._output_path = output_path
        self.device = device
        self.create_visualizations = create_visualizations
        self.save_raw_data = save_raw_data

    @property
    def output_path(self) -> Path:
        if self._output_path is not None:
            return Path(self._output_path)
        return Path(self.fabric.logger.log_dir)

    def _get_model_state_dict(self, model: nn.Module) -> StateDictType:
        """Get model state dict, optionally filtered to trainable parameters only."""
        if self.trainable_only:
            return trainable_state_dict(model)
        return model.state_dict()

    def _extract_layer_number(self, param_name: str) -> int:
        """
        Extract layer number from parameter name for various architectures.
        
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
        
        # Pattern for GPT-style models
        match = re.search(r'\.h\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        
        # Return -1 for non-layer parameters (embeddings, classifiers, etc.)
        return -1

    def _extract_component_type(self, param_name: str) -> str:
        """
        Extract component type from parameter name (attention vs MLP vs other).
        
        Returns:
            str: 'attention', 'mlp', 'norm', 'embedding', 'classifier', or 'other'
        """
        param_lower = param_name.lower()
        
        # Attention components
        if any(attn_key in param_lower for attn_key in 
               ['attn', 'attention', 'self_attn', 'cross_attn', 'multihead']):
            return 'attention'
        
        # MLP/Feed-forward components  
        if any(mlp_key in param_lower for mlp_key in
               ['mlp', 'fc', 'ffn', 'feed_forward', 'intermediate', 'dense']):
            return 'mlp'
        
        # Normalization layers
        if any(norm_key in param_lower for norm_key in
               ['norm', 'layernorm', 'layer_norm', 'batchnorm', 'groupnorm']):
            return 'norm'
        
        # Embeddings
        if any(emb_key in param_lower for emb_key in
               ['embedding', 'embed', 'positional', 'token']):
            return 'embedding'
        
        # Classifiers/Output layers
        if any(cls_key in param_lower for cls_key in
               ['classifier', 'head', 'output', 'logits', 'pred']):
            return 'classifier'
        
        return 'other'

    def _group_params_by_layer_and_component(self, state_dict: StateDictType) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """Group state dict parameters by layer number and component type."""
        layers = defaultdict(lambda: defaultdict(dict))
        
        for param_name, param_tensor in state_dict.items():
            layer_idx = self._extract_layer_number(param_name)
            component_type = self._extract_component_type(param_name)
            
            layers[layer_idx][component_type][param_name] = param_tensor
        
        return dict(layers)

    def _compute_sparsity_metrics(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Compute various sparsity metrics for a tensor.
        
        Returns:
            Dict with sparsity metrics:
            - l0_sparsity: Percentage of exact zeros
            - threshold_sparsity_{threshold}: Percentage below each threshold
            - adaptive_sparsity: Percentage below adaptive threshold
            - mean_abs_value: Mean absolute value of parameters
            - std_abs_value: Standard deviation of absolute values
        """
        metrics = {}
        
        # Convert to float for numerical stability
        tensor_flat = tensor.flatten().float()
        abs_tensor = torch.abs(tensor_flat)
        total_params = tensor_flat.numel()
        
        # L0 sparsity (exact zeros)
        exact_zeros = (tensor_flat == 0).sum().item()
        metrics['l0_sparsity'] = exact_zeros / total_params * 100.0
        
        # Threshold-based sparsity
        for threshold in self.sparsity_thresholds:
            near_zeros = (abs_tensor < threshold).sum().item()
            metrics[f'threshold_sparsity_{threshold}'] = near_zeros / total_params * 100.0
        
        # Adaptive threshold sparsity
        if self.adaptive_threshold_method == "percentile":
            adaptive_threshold = torch.quantile(abs_tensor, self.adaptive_percentile / 100.0).item()
        elif self.adaptive_threshold_method == "std":
            mean_abs = abs_tensor.mean().item()
            std_abs = abs_tensor.std().item()
            adaptive_threshold = mean_abs - 2 * std_abs  # 2-sigma below mean
            adaptive_threshold = max(adaptive_threshold, 0)  # Ensure non-negative
        elif self.adaptive_threshold_method == "iqr":
            q1 = torch.quantile(abs_tensor, 0.25).item()
            q3 = torch.quantile(abs_tensor, 0.75).item()
            iqr = q3 - q1
            adaptive_threshold = q1 - 1.5 * iqr  # Outlier detection threshold
            adaptive_threshold = max(adaptive_threshold, 0)
        else:
            adaptive_threshold = self.sparsity_thresholds[0]  # Fallback
        
        adaptive_sparse = (abs_tensor < adaptive_threshold).sum().item()
        metrics['adaptive_sparsity'] = adaptive_sparse / total_params * 100.0
        metrics['adaptive_threshold'] = adaptive_threshold
        
        # Additional statistics
        metrics['mean_abs_value'] = abs_tensor.mean().item()
        metrics['std_abs_value'] = abs_tensor.std().item()
        metrics['max_abs_value'] = abs_tensor.max().item()
        metrics['total_parameters'] = total_params
        
        return metrics

    def _compute_task_vector_sparsity(
        self, 
        pretrained_model: nn.Module, 
        finetuned_model: nn.Module,
        task_name: str
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, Dict[str, float]]]]:
        """
        Compute sparsity metrics for task vector grouped by layer and optionally by component.
        
        Returns:
            Tuple of:
            - Layer-level sparsity metrics: Dict[layer_idx -> metrics]
            - Component-level sparsity metrics: Dict[layer_idx -> component -> metrics]
        """
        
        # Get state dicts
        pretrained_state = self._get_model_state_dict(pretrained_model)
        finetuned_state = self._get_model_state_dict(finetuned_model)
        
        # Compute task vector
        task_vector_state = state_dict_sub(finetuned_state, pretrained_state)
        
        # Group by layer and component
        layer_component_groups = self._group_params_by_layer_and_component(task_vector_state)
        
        layer_sparsity = {}
        component_sparsity = {}
        
        for layer_idx, components in layer_component_groups.items():
            # Layer-level sparsity (all parameters in the layer combined)
            layer_tensors = []
            for component_params in components.values():
                for param_tensor in component_params.values():
                    layer_tensors.append(param_tensor.flatten())
            
            if layer_tensors:
                combined_layer_tensor = torch.cat(layer_tensors)
                layer_sparsity[layer_idx] = self._compute_sparsity_metrics(combined_layer_tensor)
            
            # Component-level sparsity (if requested)
            if self.analyze_components_separately:
                component_sparsity[layer_idx] = {}
                for component_type, component_params in components.items():
                    if component_params:  # Skip empty components
                        component_tensors = []
                        for param_tensor in component_params.values():
                            component_tensors.append(param_tensor.flatten())
                        
                        if component_tensors:
                            combined_component_tensor = torch.cat(component_tensors)
                            component_sparsity[layer_idx][component_type] = self._compute_sparsity_metrics(combined_component_tensor)
        
        return layer_sparsity, component_sparsity

    def run(self, modelpool: BaseModelPool):
        """
        Execute the comprehensive task vector sparsity analysis.

        This method:
        1. Loads the pretrained base model from the model pool
        2. Computes task vectors for each fine-tuned model
        3. Analyzes sparsity patterns layer-by-layer and task-by-task
        4. Generates comprehensive visualizations and reports
        5. Saves detailed results as CSV files

        Args:
            modelpool (BaseModelPool): Pool containing pretrained and fine-tuned models

        Returns:
            Dict: Summary of sparsity analysis results
        """
        print("Starting Task Vector Sparsity Analysis...")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        pretrained_model = modelpool.load_pretrained_model()
        
        # Store results
        all_layer_sparsity = {}  # task_name -> layer_idx -> metrics
        all_component_sparsity = {}  # task_name -> layer_idx -> component -> metrics
        task_names = []
        
        # Analyze each task
        for task_name, finetuned_model in tqdm(
            modelpool.named_models(), 
            total=len(modelpool),
            desc="Computing sparsity for each task"
        ):
            print(f"Analyzing sparsity for task: {task_name}")
            task_names.append(task_name)
            
            layer_sparsity, component_sparsity = self._compute_task_vector_sparsity(
                pretrained_model, finetuned_model, task_name
            )
            
            all_layer_sparsity[task_name] = layer_sparsity
            all_component_sparsity[task_name] = component_sparsity
        
        print(f"✓ Sparsity analysis completed for {len(modelpool)} tasks")
        
        # Save detailed results
        if self.save_raw_data:
            self._save_detailed_results(all_layer_sparsity, all_component_sparsity)
        
        # Generate summary statistics
        summary_stats = self._compute_summary_statistics(all_layer_sparsity, all_component_sparsity)
        
        # Create visualizations
        if self.create_visualizations:
            self._create_visualizations(all_layer_sparsity, all_component_sparsity, summary_stats)
        
        # Generate analysis report
        self._generate_report(summary_stats, task_names)
        
        print(f"✓ Task Vector Sparsity Analysis completed. Results saved to: {self.output_path}")
        
        return {
            "layer_sparsity": all_layer_sparsity,
            "component_sparsity": all_component_sparsity,
            "summary_stats": summary_stats,
            "output_path": str(self.output_path),
        }

    def _save_detailed_results(self, all_layer_sparsity: Dict, all_component_sparsity: Dict):
        """Save detailed sparsity results to CSV files."""
        
        # Prepare layer-level data
        layer_data = []
        for task_name, layer_metrics in all_layer_sparsity.items():
            for layer_idx, metrics in layer_metrics.items():
                row = {"task": task_name, "layer": layer_idx}
                row.update(metrics)
                layer_data.append(row)
        
        layer_df = pd.DataFrame(layer_data)
        layer_csv_path = self.output_path / "task_vector_sparsity_detailed.csv"
        layer_df.to_csv(layer_csv_path, index=False)
        print(f"✓ Detailed layer sparsity saved to: {layer_csv_path}")
        
        # Prepare component-level data (if available)
        if self.analyze_components_separately and all_component_sparsity:
            component_data = []
            for task_name, layer_components in all_component_sparsity.items():
                for layer_idx, components in layer_components.items():
                    for component_type, metrics in components.items():
                        row = {"task": task_name, "layer": layer_idx, "component": component_type}
                        row.update(metrics)
                        component_data.append(row)
            
            component_df = pd.DataFrame(component_data)
            component_csv_path = self.output_path / "task_vector_sparsity_components_detailed.csv"
            component_df.to_csv(component_csv_path, index=False)
            print(f"✓ Detailed component sparsity saved to: {component_csv_path}")

    def _compute_summary_statistics(self, all_layer_sparsity: Dict, all_component_sparsity: Dict) -> Dict:
        """Compute summary statistics across all tasks and layers."""
        
        summary = {
            "layer_stats": {},
            "task_stats": {},
            "component_stats": {},
            "global_stats": {}
        }
        
        # Collect all sparsity values for different metrics
        metric_values = defaultdict(list)
        layer_averages = defaultdict(list)  # layer_idx -> list of average sparsity values across tasks
        task_averages = defaultdict(list)   # task_name -> list of average sparsity values across layers
        
        # Process layer-level data
        for task_name, layer_metrics in all_layer_sparsity.items():
            task_sparsity_values = []
            
            for layer_idx, metrics in layer_metrics.items():
                # Use adaptive sparsity as the main metric
                sparsity_value = metrics.get('adaptive_sparsity', 0.0)
                
                metric_values['adaptive_sparsity'].append(sparsity_value)
                layer_averages[layer_idx].append(sparsity_value)
                task_sparsity_values.append(sparsity_value)
                
                # Collect other metrics
                for metric_name, value in metrics.items():
                    if metric_name.endswith('_sparsity'):
                        metric_values[metric_name].append(value)
            
            if task_sparsity_values:
                task_averages[task_name] = np.mean(task_sparsity_values)
        
        # Layer-wise statistics
        for layer_idx, values in layer_averages.items():
            summary["layer_stats"][layer_idx] = {
                "mean_sparsity": np.mean(values),
                "std_sparsity": np.std(values),
                "min_sparsity": np.min(values),
                "max_sparsity": np.max(values),
                "num_tasks": len(values)
            }
        
        # Task-wise statistics  
        for task_name, avg_sparsity in task_averages.items():
            summary["task_stats"][task_name] = {
                "mean_sparsity": avg_sparsity,
                "num_layers": len(all_layer_sparsity.get(task_name, {}))
            }
        
        # Global statistics
        for metric_name, values in metric_values.items():
            if values:
                summary["global_stats"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75)
                }
        
        # Component statistics (if available)
        if self.analyze_components_separately and all_component_sparsity:
            component_averages = defaultdict(list)  # component_type -> list of sparsity values
            
            for task_name, layer_components in all_component_sparsity.items():
                for layer_idx, components in layer_components.items():
                    for component_type, metrics in components.items():
                        sparsity_value = metrics.get('adaptive_sparsity', 0.0)
                        component_averages[component_type].append(sparsity_value)
            
            for component_type, values in component_averages.items():
                summary["component_stats"][component_type] = {
                    "mean_sparsity": np.mean(values),
                    "std_sparsity": np.std(values),
                    "min_sparsity": np.min(values),
                    "max_sparsity": np.max(values),
                    "count": len(values)
                }
        
        # Save summary statistics
        summary_df = pd.DataFrame([summary["global_stats"]]).T
        summary_csv_path = self.output_path / "task_vector_sparsity_summary.csv"
        summary_df.to_csv(summary_csv_path)
        
        return summary

    def _create_visualizations(self, all_layer_sparsity: Dict, all_component_sparsity: Dict, summary_stats: Dict):
        """Create comprehensive visualization plots."""
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        with PdfPages(self.output_path / "task_vector_sparsity_analysis.pdf") as pdf:
            
            # 1. Layer vs Task Sparsity Heatmap
            self._plot_sparsity_heatmap(all_layer_sparsity, pdf)
            
            # 2. Sparsity Distribution Plots
            self._plot_sparsity_distributions(all_layer_sparsity, pdf)
            
            # 3. Layer-wise Sparsity Progression
            self._plot_layer_sparsity_progression(all_layer_sparsity, pdf)
            
            # 4. Component-wise Analysis (if available)
            if self.analyze_components_separately and all_component_sparsity:
                self._plot_component_sparsity_analysis(all_component_sparsity, pdf)
            
            # 5. Threshold Comparison Plot
            self._plot_threshold_comparison(all_layer_sparsity, pdf)

    def _plot_sparsity_heatmap(self, all_layer_sparsity: Dict, pdf: PdfPages):
        """Create heatmap showing sparsity across layers and tasks."""
        
        # Prepare data for heatmap
        tasks = list(all_layer_sparsity.keys())
        all_layers = set()
        for layer_metrics in all_layer_sparsity.values():
            all_layers.update(layer_metrics.keys())
        
        layers = sorted([l for l in all_layers if l >= 0])  # Exclude non-layer parameters
        
        # Create sparsity matrix
        sparsity_matrix = np.zeros((len(tasks), len(layers)))
        sparsity_matrix.fill(np.nan)  # Use NaN for missing data
        
        for i, task in enumerate(tasks):
            for j, layer in enumerate(layers):
                if layer in all_layer_sparsity[task]:
                    sparsity_matrix[i, j] = all_layer_sparsity[task][layer].get('adaptive_sparsity', 0.0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.5), max(8, len(tasks) * 0.3)))
        
        sns.heatmap(
            sparsity_matrix,
            xticklabels=[f"Layer {l}" for l in layers],
            yticklabels=tasks,
            cmap='RdYlBu_r',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Sparsity (%)'},
            ax=ax,
            annot=True,
            fmt='.1f'
        )
        
        ax.set_title('Task Vector Sparsity Heatmap\n(Adaptive Threshold)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Network Layer', fontsize=12)
        ax.set_ylabel('Task', fontsize=12)
        
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_sparsity_distributions(self, all_layer_sparsity: Dict, pdf: PdfPages):
        """Plot distribution of sparsity values."""
        
        # Collect all sparsity values
        all_sparsity_values = []
        task_sparsity_values = defaultdict(list)
        
        for task_name, layer_metrics in all_layer_sparsity.items():
            for layer_idx, metrics in layer_metrics.items():
                if layer_idx >= 0:  # Skip non-layer parameters
                    sparsity = metrics.get('adaptive_sparsity', 0.0)
                    all_sparsity_values.append(sparsity)
                    task_sparsity_values[task_name].append(sparsity)
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall distribution
        axes[0, 0].hist(all_sparsity_values, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Overall Sparsity Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Sparsity (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(all_sparsity_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_sparsity_values):.1f}%')
        axes[0, 0].legend()
        
        # Box plot by task
        task_data = [values for values in task_sparsity_values.values()]
        task_labels = list(task_sparsity_values.keys())
        
        axes[0, 1].boxplot(task_data, labels=task_labels)
        axes[0, 1].set_title('Sparsity Distribution by Task', fontweight='bold')
        axes[0, 1].set_ylabel('Sparsity (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Violin plot by task (if we have enough tasks)
        if len(task_labels) <= 10:
            data_for_violin = []
            labels_for_violin = []
            for task, values in task_sparsity_values.items():
                data_for_violin.extend(values)
                labels_for_violin.extend([task] * len(values))
            
            violin_df = pd.DataFrame({'Task': labels_for_violin, 'Sparsity': data_for_violin})
            sns.violinplot(data=violin_df, x='Task', y='Sparsity', ax=axes[1, 0])
            axes[1, 0].set_title('Sparsity Distribution by Task (Violin Plot)', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Too many tasks for violin plot', ha='center', va='center')
            axes[1, 0].set_title('Sparsity Distribution by Task (Violin Plot)', fontweight='bold')
        
        # Q-Q plot for normality check
        from scipy.stats import probplot
        probplot(all_sparsity_values, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
        
        plt.suptitle('Task Vector Sparsity Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_layer_sparsity_progression(self, all_layer_sparsity: Dict, pdf: PdfPages):
        """Plot how sparsity changes across layers."""
        
        # Collect data
        all_layers = set()
        for layer_metrics in all_layer_sparsity.values():
            all_layers.update(layer_metrics.keys())
        
        layers = sorted([l for l in all_layers if l >= 0])
        
        # Calculate mean sparsity per layer across all tasks
        layer_mean_sparsity = []
        layer_std_sparsity = []
        
        for layer in layers:
            layer_values = []
            for task_metrics in all_layer_sparsity.values():
                if layer in task_metrics:
                    layer_values.append(task_metrics[layer].get('adaptive_sparsity', 0.0))
            
            if layer_values:
                layer_mean_sparsity.append(np.mean(layer_values))
                layer_std_sparsity.append(np.std(layer_values))
            else:
                layer_mean_sparsity.append(0)
                layer_std_sparsity.append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot individual task lines (with transparency)
        for task_name, layer_metrics in all_layer_sparsity.items():
            task_sparsity = []
            task_layers = []
            for layer in layers:
                if layer in layer_metrics:
                    task_sparsity.append(layer_metrics[layer].get('adaptive_sparsity', 0.0))
                    task_layers.append(layer)
            
            if task_layers:
                ax.plot(task_layers, task_sparsity, alpha=0.3, linewidth=1, label=f'{task_name}')
        
        # Plot mean line with error bars
        ax.errorbar(layers, layer_mean_sparsity, yerr=layer_std_sparsity, 
                   color='black', linewidth=3, marker='o', markersize=8,
                   capsize=5, capthick=2, label='Mean ± Std')
        
        ax.set_title('Sparsity Progression Across Layers', fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Sparsity (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only show legend if we have few tasks (to avoid clutter)
        if len(all_layer_sparsity) <= 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_component_sparsity_analysis(self, all_component_sparsity: Dict, pdf: PdfPages):
        """Plot component-wise sparsity analysis."""
        
        # Collect component data
        component_data = defaultdict(list)
        
        for task_name, layer_components in all_component_sparsity.items():
            for layer_idx, components in layer_components.items():
                if layer_idx >= 0:  # Skip non-layer parameters
                    for component_type, metrics in components.items():
                        component_data[component_type].append(
                            metrics.get('adaptive_sparsity', 0.0)
                        )
        
        # Create component comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot by component
        component_types = list(component_data.keys())
        component_values = [component_data[comp] for comp in component_types]
        
        axes[0].boxplot(component_values, labels=component_types)
        axes[0].set_title('Sparsity by Component Type', fontweight='bold')
        axes[0].set_ylabel('Sparsity (%)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Mean sparsity by component (bar plot)
        component_means = [np.mean(values) for values in component_values]
        component_stds = [np.std(values) for values in component_values]
        
        bars = axes[1].bar(component_types, component_means, yerr=component_stds, 
                          capsize=5, alpha=0.7, edgecolor='black')
        axes[1].set_title('Mean Sparsity by Component Type', fontweight='bold')
        axes[1].set_ylabel('Sparsity (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, component_means):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{mean_val:.1f}%', ha='center', va='bottom')
        
        plt.suptitle('Component-wise Sparsity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _plot_threshold_comparison(self, all_layer_sparsity: Dict, pdf: PdfPages):
        """Compare sparsity across different thresholds."""
        
        # Collect data for different thresholds
        threshold_data = defaultdict(list)
        
        for task_name, layer_metrics in all_layer_sparsity.items():
            for layer_idx, metrics in layer_metrics.items():
                if layer_idx >= 0:  # Skip non-layer parameters
                    for metric_name, value in metrics.items():
                        if metric_name.endswith('_sparsity') and 'threshold' in metric_name:
                            threshold_data[metric_name].append(value)
                    
                    # Add adaptive sparsity
                    threshold_data['adaptive_sparsity'].append(
                        metrics.get('adaptive_sparsity', 0.0)
                    )
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        threshold_names = list(threshold_data.keys())
        threshold_means = [np.mean(threshold_data[name]) for name in threshold_names]
        threshold_stds = [np.std(threshold_data[name]) for name in threshold_names]
        
        # Clean up threshold names for display
        display_names = []
        for name in threshold_names:
            if name == 'adaptive_sparsity':
                display_names.append('Adaptive')
            elif 'threshold_sparsity' in name:
                threshold_val = name.replace('threshold_sparsity_', '')
                display_names.append(f'Threshold {threshold_val}')
            else:
                display_names.append(name)
        
        bars = ax.bar(range(len(threshold_names)), threshold_means, 
                     yerr=threshold_stds, capsize=5, alpha=0.7, edgecolor='black')
        
        ax.set_title('Sparsity Comparison Across Different Thresholds', fontweight='bold')
        ax.set_ylabel('Mean Sparsity (%)')
        ax.set_xlabel('Threshold Type')
        ax.set_xticks(range(len(threshold_names)))
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, mean_val) in enumerate(zip(bars, threshold_means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean_val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _generate_report(self, summary_stats: Dict, task_names: List[str]):
        """Generate a human-readable analysis report."""
        
        report_path = self.output_path / "task_vector_sparsity_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TASK VECTOR SPARSITY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Tasks: {len(task_names)}\n")
            f.write(f"Tasks Analyzed: {', '.join(task_names)}\n")
            f.write(f"Sparsity Thresholds: {self.sparsity_thresholds}\n")
            f.write(f"Adaptive Threshold Method: {self.adaptive_threshold_method}\n")
            f.write(f"Trainable Parameters Only: {self.trainable_only}\n")
            f.write(f"Component Analysis Enabled: {self.analyze_components_separately}\n\n")
            
            # Global Statistics
            f.write("GLOBAL SPARSITY STATISTICS\n")
            f.write("-" * 40 + "\n")
            
            if 'adaptive_sparsity' in summary_stats['global_stats']:
                stats = summary_stats['global_stats']['adaptive_sparsity']
                f.write(f"Adaptive Sparsity (main metric):\n")
                f.write(f"  Mean: {stats['mean']:.2f}%\n")
                f.write(f"  Std:  {stats['std']:.2f}%\n")
                f.write(f"  Min:  {stats['min']:.2f}%\n")
                f.write(f"  Max:  {stats['max']:.2f}%\n")
                f.write(f"  Median: {stats['median']:.2f}%\n")
                f.write(f"  Q25: {stats['q25']:.2f}%\n")
                f.write(f"  Q75: {stats['q75']:.2f}%\n\n")
            
            # Task-wise Statistics
            f.write("TASK-WISE SPARSITY STATISTICS\n")
            f.write("-" * 40 + "\n")
            
            for task_name, stats in summary_stats['task_stats'].items():
                f.write(f"{task_name}:\n")
                f.write(f"  Mean Sparsity: {stats['mean_sparsity']:.2f}%\n")
                f.write(f"  Layers Analyzed: {stats['num_layers']}\n\n")
            
            # Layer-wise Statistics
            f.write("LAYER-WISE SPARSITY STATISTICS\n")
            f.write("-" * 40 + "\n")
            
            for layer_idx in sorted(summary_stats['layer_stats'].keys()):
                if layer_idx >= 0:  # Skip non-layer parameters
                    stats = summary_stats['layer_stats'][layer_idx]
                    f.write(f"Layer {layer_idx}:\n")
                    f.write(f"  Mean Sparsity: {stats['mean_sparsity']:.2f}%\n")
                    f.write(f"  Std Sparsity:  {stats['std_sparsity']:.2f}%\n")
                    f.write(f"  Min Sparsity:  {stats['min_sparsity']:.2f}%\n")
                    f.write(f"  Max Sparsity:  {stats['max_sparsity']:.2f}%\n")
                    f.write(f"  Tasks: {stats['num_tasks']}\n\n")
            
            # Component Statistics (if available)
            if self.analyze_components_separately and summary_stats['component_stats']:
                f.write("COMPONENT-WISE SPARSITY STATISTICS\n")
                f.write("-" * 40 + "\n")
                
                for component_type, stats in summary_stats['component_stats'].items():
                    f.write(f"{component_type.capitalize()}:\n")
                    f.write(f"  Mean Sparsity: {stats['mean_sparsity']:.2f}%\n")
                    f.write(f"  Std Sparsity:  {stats['std_sparsity']:.2f}%\n")
                    f.write(f"  Min Sparsity:  {stats['min_sparsity']:.2f}%\n")
                    f.write(f"  Max Sparsity:  {stats['max_sparsity']:.2f}%\n")
                    f.write(f"  Count: {stats['count']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Analysis report saved to: {report_path}")


# Configuration mapping for Hydra
TaskVectorSparsityAnalysis._config_mapping = TaskVectorSparsityAnalysis._config_mapping