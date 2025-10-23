"""
Random Subspace Task Vector Analysis

This module analyzes task vectors by projecting them into random subspaces using
various structured random projections (SRHT, FWHT, DCT, DHT) and compares geometric
properties in the original space versus projected subspaces.

Key Features:
- Multiple random projection types (SRHT, FWHT, DCT, DHT)
- Configurable projection dimensions for multi-scale analysis
- Comparative metrics: L2 distance, cosine similarity, sign conflicts
- Visualization of metric preservation across subspace dimensions
- Statistical analysis of dimension reduction effects on task vector geometry

Metrics Computed:
1. L2 Distance: Euclidean distance between task vectors
2. Cosine Similarity: Angular similarity (normalized dot product)
3. Sign Conflicts: Proportion of parameters with opposing signs
4. Metric Preservation: How well each metric is preserved after projection

Analysis Outputs:
- CSV files with pairwise metrics for each subspace dimension
- Comparison plots showing metric preservation vs projection dimension
- Correlation analysis between original and projected metrics
- Distribution plots of metric changes across different projections

Usage:
    from fusion_bench.method.analysis import RandomSubspaceTaskVectorAnalysis
    
    analyzer = RandomSubspaceTaskVectorAnalysis(
        proj_ratios=[0.1, 0.25, 0.5, 0.75, 0.9],
        transform_type="srht",
        num_random_seeds=5,
        output_path="./random_subspace_analysis"
    )
    analyzer.run(modelpool)
"""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

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

# Import projection utilities from fastfood_merging
from fusion_bench.method.fastfood_merging.fastfood_utils import (
    create_projection_ops,
    next_pow2,
)

log = logging.getLogger(__name__)


def _seed_from_string(s: str) -> int:
    """Generate a deterministic seed from a string."""
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")


@auto_register_config
class RandomSubspaceTaskVectorAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Analyzes task vectors in random subspaces using structured projections.
    
    This algorithm projects task vectors into multiple random subspaces of varying
    dimensions and analyzes how geometric properties (L2 distance, cosine similarity,
    sign conflicts) are preserved. This provides insights into:
    
    1. Intrinsic dimensionality of task vector differences
    2. Robustness of merging strategies to dimension reduction
    3. Optimal subspace dimensions for efficient merging
    4. Stability of geometric relationships under projection
    
    Args:
        proj_ratios (List[float]): List of projection ratios to analyze (0.0-1.0)
        transform_type (str): Type of random projection ("srht", "fwht", "dct", "dht")
        num_random_seeds (int): Number of random seeds to average over
        trainable_only (bool): Only analyze trainable parameters
        max_points_per_model (int, optional): Max parameters to sample for memory efficiency
        output_path (str, optional): Directory to save outputs
        method_name (str, optional): Name prefix for output files
        device (str): Device for computations ("cuda" or "cpu")
        plot_heatmaps (bool): Whether to generate heatmap visualizations
        plot_preservation (bool): Whether to plot metric preservation curves
        plot_distributions (bool): Whether to plot metric distribution changes
        compute_correlations (bool): Whether to compute correlation with original metrics
        
    Outputs:
        - random_subspace_metrics_summary_{method_name}.csv: Aggregated results
        - random_subspace_analysis_{method_name}.pdf: Comprehensive visualizations
        - subspace_metrics_{ratio}_{seed}_{method_name}.csv: Per-projection metrics
        
    Returns:
        The pretrained model from the model pool.
        
    Example:
        ```python
        >>> analyzer = RandomSubspaceTaskVectorAnalysis(
        ...     proj_ratios=[0.1, 0.25, 0.5, 0.75],
        ...     transform_type="srht",
        ...     num_random_seeds=3,
        ...     plot_heatmaps=True,
        ...     output_path="/path/to/outputs"
        ... )
        >>> result = analyzer.run(modelpool)
        ```
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        proj_ratios: List[float] = None,
        transform_type: str = "srht",
        num_random_seeds: int = 3,
        trainable_only: bool = True,
        max_points_per_model: Optional[int] = None,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        device: str = "cuda",
        plot_heatmaps: bool = True,
        plot_preservation: bool = True,
        plot_distributions: bool = True,
        compute_correlations: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_path = output_path
        self.proj_ratios = proj_ratios or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.transform_type = transform_type.lower()
        self.num_random_seeds = num_random_seeds
        self.trainable_only = trainable_only
        self.max_points_per_model = max_points_per_model
        self.method_name = method_name or "default"
        self.device = torch.device(device)
        self.plot_heatmaps = plot_heatmaps
        self.plot_preservation = plot_preservation
        self.plot_distributions = plot_distributions
        self.compute_correlations = compute_correlations

        # Validate transform type
        valid_transforms = ["srht", "fwht", "dct", "dht"]
        if self.transform_type not in valid_transforms:
            raise ValueError(
                f"Invalid transform_type '{self.transform_type}'. "
                f"Must be one of {valid_transforms}"
            )

    @property
    def output_path(self):
        if self._output_path is not None:
            return Path(self._output_path)
        elif self.fabric is not None:
            return Path(self.fabric.logger.log_dir)
        else:
            return Path("./outputs")

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Main analysis pipeline.
        
        1. Extract task vectors from model pool
        2. Compute metrics in original space (baseline)
        3. For each projection ratio and random seed:
           - Project task vectors into subspace
           - Compute metrics in projected space
           - Compare with original metrics
        4. Aggregate results and generate visualizations
        """
        log.info("=" * 80)
        log.info("Random Subspace Task Vector Analysis")
        log.info("=" * 80)
        log.info(f"Transform type: {self.transform_type}")
        log.info(f"Projection ratios: {self.proj_ratios}")
        log.info(f"Random seeds: {self.num_random_seeds}")
        log.info(f"Output path: {self.output_path}")
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Get pretrained and finetuned models
        pretrained_model = modelpool.load_model("_pretrained_")
        model_names = [name for name in modelpool.model_names if name != "_pretrained_"]
        
        log.info(f"Number of task models: {len(model_names)}")
        log.info(f"Model names: {model_names}")
        
        # Extract task vectors
        log.info("Extracting task vectors...")
        task_vectors = []
        for name in tqdm(model_names, desc="Loading models"):
            finetuned_model = modelpool.load_model(name)
            task_vector = self.get_task_vector(pretrained_model, finetuned_model)
            task_vectors.append(task_vector)
        
        task_vectors_tensor = torch.stack(task_vectors).to(self.device)
        log.info(f"Task vectors shape: {task_vectors_tensor.shape}")
        
        # Compute original space metrics (baseline)
        log.info("Computing metrics in original space...")
        original_metrics = self._compute_all_metrics(
            task_vectors_tensor, model_names, space_name="original"
        )
        
        # Analyze subspaces
        log.info("Analyzing random subspaces...")
        subspace_results = []
        all_projected_metrics = {}  # Store for heatmap generation
        
        for proj_ratio in tqdm(self.proj_ratios, desc="Projection ratios"):
            ratio_metrics = []  # Store all seeds for this ratio
            
            for seed_idx in tqdm(range(self.num_random_seeds), desc="Random seeds", leave=False):
                seed_key = f"{self.transform_type}_ratio{proj_ratio}_seed{seed_idx}"
                
                # Project task vectors
                projected_vectors = self._project_task_vectors(
                    task_vectors_tensor, proj_ratio, seed_key
                )
                
                # Compute metrics in projected space
                projected_metrics = self._compute_all_metrics(
                    projected_vectors, model_names, 
                    space_name=f"proj_{proj_ratio}_seed{seed_idx}"
                )
                
                ratio_metrics.append(projected_metrics)
                
                # Compare with original metrics
                comparison = self._compare_metrics(
                    original_metrics, projected_metrics, proj_ratio, seed_idx
                )
                subspace_results.append(comparison)
                
                # Save individual projection metrics
                self._save_projection_metrics(
                    projected_metrics, proj_ratio, seed_idx
                )
            
            # Store averaged metrics for this ratio (for heatmaps)
            all_projected_metrics[proj_ratio] = self._average_metrics_across_seeds(ratio_metrics)
        
        # Aggregate results
        log.info("Aggregating results...")
        summary_df = pd.DataFrame(subspace_results)
        summary_path = self.output_path / f"random_subspace_metrics_summary_{self.method_name}.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info(f"Saved summary to {summary_path}")
        
        # Generate visualizations
        log.info("Generating visualizations...")
        self._generate_visualizations(
            original_metrics, all_projected_metrics, summary_df, model_names
        )
        
        log.info("=" * 80)
        log.info("Random Subspace Analysis Complete!")
        log.info("=" * 80)
        
        return pretrained_model

    def _project_task_vectors(
        self,
        task_vectors: torch.Tensor,
        proj_ratio: float,
        seed_key: str,
    ) -> torch.Tensor:
        """
        Project task vectors into a random subspace.
        
        Args:
            task_vectors: Task vectors [num_models, vector_dim]
            proj_ratio: Projection ratio (0.0-1.0)
            seed_key: Seed key for deterministic projection
            
        Returns:
            Projected task vectors [num_models, proj_dim]
        """
        num_models, global_dim = task_vectors.shape
        proj_dim = max(1, int(global_dim * proj_ratio))
        
        # Create projection operators
        fwd, lift = create_projection_ops(
            global_dim=global_dim,
            proj_dim=proj_dim,
            transform_type=self.transform_type,
            seed_key=seed_key,
            device=self.device,
        )
        
        # Project all task vectors
        projected = torch.stack([fwd(tv) for tv in task_vectors])
        
        return projected

    def _compute_all_metrics(
        self,
        task_vectors: torch.Tensor,
        model_names: List[str],
        space_name: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all pairwise metrics for task vectors.
        
        Args:
            task_vectors: Task vectors [num_models, vector_dim]
            model_names: Names of models
            space_name: Name of the space (for logging)
            
        Returns:
            Dictionary containing DataFrames for each metric
        """
        num_models = len(model_names)
        
        # Initialize metric matrices
        cos_sim_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        l2_dist_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        sign_conflict_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        jaccard_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        
        # Compute pairwise metrics
        for i in range(num_models):
            for j in range(i + 1, num_models):
                vec_i = task_vectors[i]
                vec_j = task_vectors[j]
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    vec_i.unsqueeze(0), vec_j.unsqueeze(0)
                ).item()
                cos_sim_matrix[i, j] = cos_sim
                cos_sim_matrix[j, i] = cos_sim
                
                # L2 distance
                l2_dist = torch.norm(vec_i - vec_j, p=2).item()
                l2_dist_matrix[i, j] = l2_dist
                l2_dist_matrix[j, i] = l2_dist
                
                # Sign conflicts
                signs_i = torch.sign(vec_i)
                signs_j = torch.sign(vec_j)
                conflicts = (signs_i * signs_j < 0).float().mean().item()
                sign_conflict_matrix[i, j] = conflicts
                sign_conflict_matrix[j, i] = conflicts
                
                # Jaccard similarity (based on sign patterns)
                # Convert to binary: positive=1, non-positive=0
                pos_i = (vec_i > 0).float()
                pos_j = (vec_j > 0).float()
                intersection = (pos_i * pos_j).sum().item()
                union = ((pos_i + pos_j) > 0).float().sum().item()
                jaccard = intersection / union if union > 0 else 0.0
                jaccard_matrix[i, j] = jaccard
                jaccard_matrix[j, i] = jaccard
        
        # Set diagonal values
        for i in range(num_models):
            cos_sim_matrix[i, i] = 1.0
            sign_conflict_matrix[i, i] = 0.0
            l2_dist_matrix[i, i] = 0.0
            jaccard_matrix[i, i] = 1.0
        
        # Convert to DataFrames
        cos_sim_df = pd.DataFrame(
            cos_sim_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names,
        )
        l2_dist_df = pd.DataFrame(
            l2_dist_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names,
        )
        sign_conflict_df = pd.DataFrame(
            sign_conflict_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names,
        )
        jaccard_df = pd.DataFrame(
            jaccard_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names,
        )
        
        return {
            "cosine_similarity": cos_sim_df,
            "l2_distance": l2_dist_df,
            "sign_conflicts": sign_conflict_df,
            "jaccard_similarity": jaccard_df,
        }

    def _compare_metrics(
        self,
        original_metrics: Dict[str, pd.DataFrame],
        projected_metrics: Dict[str, pd.DataFrame],
        proj_ratio: float,
        seed_idx: int,
    ) -> Dict[str, Any]:
        """
        Compare original and projected metrics.
        
        Returns a dictionary with comparison statistics.
        """
        comparison = {
            "proj_ratio": proj_ratio,
            "seed_idx": seed_idx,
            "transform_type": self.transform_type,
        }
        
        for metric_name in ["cosine_similarity", "l2_distance", "sign_conflicts", "jaccard_similarity"]:
            orig = original_metrics[metric_name].values
            proj = projected_metrics[metric_name].values
            
            # Get upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(orig, dtype=bool), k=1)
            orig_vals = orig[mask]
            proj_vals = proj[mask]
            
            # Compute comparison statistics
            if metric_name in ["cosine_similarity", "jaccard_similarity"]:
                # For similarity metrics, compute correlation and MAE
                correlation = np.corrcoef(orig_vals, proj_vals)[0, 1]
                mae = np.mean(np.abs(orig_vals - proj_vals))
                rmse = np.sqrt(np.mean((orig_vals - proj_vals) ** 2))
                
                comparison[f"{metric_name}_correlation"] = correlation
                comparison[f"{metric_name}_mae"] = mae
                comparison[f"{metric_name}_rmse"] = rmse
                comparison[f"{metric_name}_orig_mean"] = np.mean(orig_vals)
                comparison[f"{metric_name}_proj_mean"] = np.mean(proj_vals)
                
            elif metric_name == "l2_distance":
                # For L2 distance, normalize by original magnitude
                correlation = np.corrcoef(orig_vals, proj_vals)[0, 1]
                relative_error = np.mean(np.abs(orig_vals - proj_vals) / (orig_vals + 1e-8))
                
                comparison[f"{metric_name}_correlation"] = correlation
                comparison[f"{metric_name}_relative_error"] = relative_error
                comparison[f"{metric_name}_orig_mean"] = np.mean(orig_vals)
                comparison[f"{metric_name}_proj_mean"] = np.mean(proj_vals)
                
            elif metric_name == "sign_conflicts":
                # For sign conflicts, compute agreement
                correlation = np.corrcoef(orig_vals, proj_vals)[0, 1]
                mae = np.mean(np.abs(orig_vals - proj_vals))
                
                comparison[f"{metric_name}_correlation"] = correlation
                comparison[f"{metric_name}_mae"] = mae
                comparison[f"{metric_name}_orig_mean"] = np.mean(orig_vals)
                comparison[f"{metric_name}_proj_mean"] = np.mean(proj_vals)
        
        return comparison

    def _average_metrics_across_seeds(
        self,
        metrics_list: List[Dict[str, pd.DataFrame]],
    ) -> Dict[str, pd.DataFrame]:
        """
        Average metrics across multiple random seeds.
        
        Args:
            metrics_list: List of metric dictionaries from different seeds
            
        Returns:
            Dictionary of averaged metric DataFrames
        """
        if not metrics_list:
            return {}
        
        # Initialize with zeros
        averaged = {}
        metric_names = metrics_list[0].keys()
        
        for metric_name in metric_names:
            # Stack all DataFrames for this metric
            stacked = np.stack([m[metric_name].values for m in metrics_list])
            # Average across seeds (axis 0)
            avg_values = np.mean(stacked, axis=0)
            # Create DataFrame with same index/columns
            averaged[metric_name] = pd.DataFrame(
                avg_values,
                index=metrics_list[0][metric_name].index,
                columns=metrics_list[0][metric_name].columns,
            )
        
        return averaged

    def _save_projection_metrics(
        self,
        metrics: Dict[str, pd.DataFrame],
        proj_ratio: float,
        seed_idx: int,
    ):
        """Save metrics for a specific projection configuration."""
        # Save CSV files
        for metric_name, df in metrics.items():
            filename = (
                f"subspace_{metric_name}_ratio{proj_ratio:.2f}_seed{seed_idx}_{self.method_name}.csv"
            )
            filepath = self.output_path / filename
            df.to_csv(filepath)
        
        # Save JSON files for easier programmatic access
        import json
        json_filename = f"subspace_metrics_ratio{proj_ratio:.2f}_seed{seed_idx}_{self.method_name}.json"
        json_filepath = self.output_path / json_filename
        
        json_data = {
            metric_name: df.to_dict(orient='index')
            for metric_name, df in metrics.items()
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)

    def _generate_visualizations(
        self,
        original_metrics: Dict[str, pd.DataFrame],
        all_projected_metrics: Dict[float, Dict[str, pd.DataFrame]],
        summary_df: pd.DataFrame,
        model_names: List[str],
    ):
        """Generate comprehensive visualizations including per-ratio heatmaps."""
        pdf_path = self.output_path / f"random_subspace_analysis_{self.method_name}.pdf"
        
        with PdfPages(pdf_path) as pdf:
            # Plot 1: Metric preservation vs projection ratio
            if self.plot_preservation:
                self._plot_metric_preservation(summary_df, pdf)
            
            # Plot 2: Heatmaps for original space
            if self.plot_heatmaps:
                self._plot_original_heatmaps(original_metrics, model_names, pdf)
            
            # NEW: Plot 3: Heatmaps for each projection ratio
            if self.plot_heatmaps:
                for proj_ratio in sorted(all_projected_metrics.keys()):
                    self._plot_projected_heatmaps(
                        all_projected_metrics[proj_ratio],
                        model_names,
                        proj_ratio,
                        pdf,
                    )
            
            # Plot 4: Distribution of metric changes
            if self.plot_distributions:
                self._plot_metric_distributions(summary_df, pdf)
            
            # Plot 5: Comparison across transform types (if multiple seeds)
            if self.num_random_seeds > 1:
                self._plot_seed_variability(summary_df, pdf)
        
        log.info(f"Saved visualizations to {pdf_path}")
        
        # Save heatmap data as JSON
        self._save_heatmap_json(original_metrics, all_projected_metrics)

    def _plot_metric_preservation(self, summary_df: pd.DataFrame, pdf: PdfPages):
        """Plot how metrics are preserved across projection ratios."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Metric Preservation in Random Subspaces ({self.transform_type.upper()})",
            fontsize=16,
            fontweight="bold",
        )
        
        # Aggregate by projection ratio (average over seeds)
        agg_df = summary_df.groupby("proj_ratio").agg({
            "cosine_similarity_correlation": ["mean", "std"],
            "l2_distance_correlation": ["mean", "std"],
            "sign_conflicts_correlation": ["mean", "std"],
            "jaccard_similarity_correlation": ["mean", "std"],
            "cosine_similarity_mae": ["mean", "std"],
            "jaccard_similarity_mae": ["mean", "std"],
        }).reset_index()
        
        proj_ratios = agg_df["proj_ratio"].values
        
        # Plot 1: Correlation preservation
        ax = axes[0, 0]
        metrics_to_plot = [
            ("cosine_similarity_correlation", "Cosine Similarity", "blue"),
            ("l2_distance_correlation", "L2 Distance", "red"),
            ("sign_conflicts_correlation", "Sign Conflicts", "green"),
            ("jaccard_similarity_correlation", "Jaccard Similarity", "orange"),
        ]
        
        for metric_key, label, color in metrics_to_plot:
            means = agg_df[(metric_key, "mean")].values
            stds = agg_df[(metric_key, "std")].values
            ax.plot(proj_ratios, means, marker="o", label=label, color=color, linewidth=2)
            ax.fill_between(
                proj_ratios,
                means - stds,
                means + stds,
                alpha=0.2,
                color=color,
            )
        
        ax.set_xlabel("Projection Ratio", fontsize=12)
        ax.set_ylabel("Correlation with Original", fontsize=12)
        ax.set_title("Metric Correlation Preservation", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Plot 2: Cosine similarity MAE
        ax = axes[0, 1]
        means = agg_df[("cosine_similarity_mae", "mean")].values
        stds = agg_df[("cosine_similarity_mae", "std")].values
        ax.plot(proj_ratios, means, marker="o", color="purple", linewidth=2)
        ax.fill_between(proj_ratios, means - stds, means + stds, alpha=0.2, color="purple")
        ax.set_xlabel("Projection Ratio", fontsize=12)
        ax.set_ylabel("Mean Absolute Error", fontsize=12)
        ax.set_title("Cosine Similarity MAE", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Jaccard similarity MAE
        ax = axes[1, 0]
        if "jaccard_similarity_mae" in agg_df.columns.get_level_values(0):
            means = agg_df[("jaccard_similarity_mae", "mean")].values
            stds = agg_df[("jaccard_similarity_mae", "std")].values
            ax.plot(proj_ratios, means, marker="o", color="orange", linewidth=2)
            ax.fill_between(proj_ratios, means - stds, means + stds, alpha=0.2, color="orange")
        ax.set_xlabel("Projection Ratio", fontsize=12)
        ax.set_ylabel("Mean Absolute Error", fontsize=12)
        ax.set_title("Jaccard Similarity MAE", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Mean metric values in original space
        ax = axes[1, 1]
        orig_means = summary_df.groupby("proj_ratio").first()
        metrics_orig = [
            ("cosine_similarity_orig_mean", "Cosine Similarity", "blue"),
            ("jaccard_similarity_orig_mean", "Jaccard Similarity", "orange"),
            ("sign_conflicts_orig_mean", "Sign Conflicts", "green"),
        ]
        for metric_key, label, color in metrics_orig:
            if metric_key in orig_means.columns:
                values = orig_means[metric_key].values
                ax.plot(proj_ratios, values, marker="s", label=label, color=color, linewidth=2)
        ax.set_xlabel("Projection Ratio", fontsize=12)
        ax.set_ylabel("Mean Value", fontsize=12)
        ax.set_title("Original Space Metric Values", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _plot_original_heatmaps(
        self,
        original_metrics: Dict[str, pd.DataFrame],
        model_names: List[str],
        pdf: PdfPages,
    ):
        """Plot heatmaps of original space metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            "Task Vector Metrics in Original Space",
            fontsize=16,
            fontweight="bold",
        )
        
        metrics_to_plot = [
            ("cosine_similarity", "Cosine Similarity", "coolwarm"),
            ("l2_distance", "L2 Distance", "viridis"),
            ("sign_conflicts", "Sign Conflicts", "YlOrRd"),
            ("jaccard_similarity", "Jaccard Similarity", "RdYlGn"),
        ]
        
        for ax, (metric_key, title, cmap) in zip(axes.flat, metrics_to_plot):
            df = original_metrics[metric_key]
            
            # Set vmin/vmax based on metric type
            if metric_key == "cosine_similarity":
                vmin, vmax = -1, 1
            elif metric_key == "jaccard_similarity":
                vmin, vmax = 0, 1
            else:
                vmin, vmax = df.min().min(), df.max().max()
            
            sns.heatmap(
                df,
                annot=len(model_names) <= 10,
                fmt=".3f",
                cmap=cmap,
                ax=ax,
                cbar_kws={"label": title},
                square=True,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Model", fontsize=11)
            ax.set_ylabel("Model", fontsize=11)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _plot_projected_heatmaps(
        self,
        projected_metrics: Dict[str, pd.DataFrame],
        model_names: List[str],
        proj_ratio: float,
        pdf: PdfPages,
    ):
        """Plot heatmaps for a specific projection ratio (averaged across seeds)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            f"Task Vector Metrics in Projected Space (ratio={proj_ratio:.2f}, {self.transform_type.upper()})",
            fontsize=16,
            fontweight="bold",
        )
        
        metrics_to_plot = [
            ("cosine_similarity", "Cosine Similarity", "coolwarm"),
            ("l2_distance", "L2 Distance", "viridis"),
            ("sign_conflicts", "Sign Conflicts", "YlOrRd"),
            ("jaccard_similarity", "Jaccard Similarity", "RdYlGn"),
        ]
        
        for ax, (metric_key, title, cmap) in zip(axes.flat, metrics_to_plot):
            df = projected_metrics[metric_key]
            
            # Set vmin/vmax based on metric type
            if metric_key == "cosine_similarity":
                vmin, vmax = -1, 1
            elif metric_key == "jaccard_similarity":
                vmin, vmax = 0, 1
            else:
                vmin, vmax = df.min().min(), df.max().max()
            
            sns.heatmap(
                df,
                annot=len(model_names) <= 10,
                fmt=".3f",
                cmap=cmap,
                ax=ax,
                cbar_kws={"label": title},
                square=True,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{title} @ {proj_ratio:.0%} dim", fontsize=13, fontweight="bold")
            ax.set_xlabel("Model", fontsize=11)
            ax.set_ylabel("Model", fontsize=11)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _save_heatmap_json(
        self,
        original_metrics: Dict[str, pd.DataFrame],
        all_projected_metrics: Dict[float, Dict[str, pd.DataFrame]],
    ):
        """Save all heatmap data as JSON for programmatic access."""
        import json
        
        json_data = {
            "original_space": {
                metric_name: df.to_dict(orient='index')
                for metric_name, df in original_metrics.items()
            },
            "projected_spaces": {
                f"ratio_{proj_ratio:.2f}": {
                    metric_name: df.to_dict(orient='index')
                    for metric_name, df in metrics.items()
                }
                for proj_ratio, metrics in all_projected_metrics.items()
            },
            "metadata": {
                "transform_type": self.transform_type,
                "proj_ratios": list(self.proj_ratios),  # Convert ListConfig to list
                "num_random_seeds": self.num_random_seeds,
                "method_name": self.method_name,
            }
        }
        
        json_path = self.output_path / f"all_heatmaps_{self.method_name}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        log.info(f"Saved heatmap JSON to {json_path}")

    def _plot_metric_distributions(self, summary_df: pd.DataFrame, pdf: PdfPages):
        """Plot distributions of metric changes across projections."""
        # Create two pages of distribution plots
        
        # Page 1: Correlation distributions
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle(
            f"Correlation Distributions ({self.transform_type.upper()})",
            fontsize=16,
            fontweight="bold",
        )
        
        correlation_metrics = [
            ("cosine_similarity_correlation", "Cosine Similarity", axes1[0, 0]),
            ("l2_distance_correlation", "L2 Distance", axes1[0, 1]),
            ("sign_conflicts_correlation", "Sign Conflicts", axes1[1, 0]),
            ("jaccard_similarity_correlation", "Jaccard Similarity", axes1[1, 1]),
        ]
        
        for metric_key, title, ax in correlation_metrics:
            if metric_key in summary_df.columns:
                for ratio in sorted(summary_df["proj_ratio"].unique()):
                    data = summary_df[summary_df["proj_ratio"] == ratio][metric_key]
                    ax.hist(data, alpha=0.5, label=f"Ratio {ratio:.2f}", bins=10)
                ax.set_xlabel("Correlation", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"{title} Correlation Distribution", fontsize=13, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)
        
        # Page 2: MAE distributions
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle(
            f"MAE Distributions ({self.transform_type.upper()})",
            fontsize=16,
            fontweight="bold",
        )
        
        mae_metrics = [
            ("cosine_similarity_mae", "Cosine Similarity", axes2[0, 0]),
            ("jaccard_similarity_mae", "Jaccard Similarity", axes2[0, 1]),
        ]
        
        for metric_key, title, ax in mae_metrics:
            if metric_key in summary_df.columns:
                for ratio in sorted(summary_df["proj_ratio"].unique()):
                    data = summary_df[summary_df["proj_ratio"] == ratio][metric_key]
                    ax.hist(data, alpha=0.5, label=f"Ratio {ratio:.2f}", bins=10)
                ax.set_xlabel("Mean Absolute Error", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"{title} MAE Distribution", fontsize=13, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for ax in [axes2[1, 0], axes2[1, 1]]:
            ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    def _plot_seed_variability(self, summary_df: pd.DataFrame, pdf: PdfPages):
        """Plot variability across random seeds."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Variability Across Random Seeds ({self.transform_type.upper()})",
            fontsize=16,
            fontweight="bold",
        )
        
        # Compute seed statistics for each ratio
        agg_dict = {
            "cosine_similarity_correlation": ["mean", "std", "min", "max"],
            "l2_distance_correlation": ["mean", "std", "min", "max"],
            "sign_conflicts_correlation": ["mean", "std", "min", "max"],
            "cosine_similarity_mae": ["mean", "std", "min", "max"],
        }
        
        # Add Jaccard if available
        if "jaccard_similarity_correlation" in summary_df.columns:
            agg_dict["jaccard_similarity_correlation"] = ["mean", "std", "min", "max"]
            agg_dict["jaccard_similarity_mae"] = ["mean", "std", "min", "max"]
        
        seed_stats = summary_df.groupby("proj_ratio").agg(agg_dict).reset_index()
        
        proj_ratios = seed_stats["proj_ratio"].values
        
        metrics_to_plot = [
            (
                "cosine_similarity_correlation",
                "Cosine Similarity Correlation",
                axes[0, 0],
                "blue",
            ),
            (
                "l2_distance_correlation",
                "L2 Distance Correlation",
                axes[0, 1],
                "red",
            ),
            (
                "sign_conflicts_correlation",
                "Sign Conflicts Correlation",
                axes[1, 0],
                "green",
            ),
            (
                "jaccard_similarity_correlation",
                "Jaccard Similarity Correlation",
                axes[1, 1],
                "orange",
            ),
        ]
        
        for metric_key, title, ax, color in metrics_to_plot:
            if metric_key in seed_stats.columns.get_level_values(0):
                means = seed_stats[(metric_key, "mean")].values
                stds = seed_stats[(metric_key, "std")].values
                mins = seed_stats[(metric_key, "min")].values
                maxs = seed_stats[(metric_key, "max")].values
                
                ax.plot(proj_ratios, means, marker="o", color=color, linewidth=2, label="Mean")
                ax.fill_between(proj_ratios, mins, maxs, alpha=0.2, color=color, label="Min-Max Range")
                ax.fill_between(
                    proj_ratios,
                    means - stds,
                    means + stds,
                    alpha=0.4,
                    color=color,
                    label="±1 Std Dev",
                )
                
                ax.set_xlabel("Projection Ratio", fontsize=12)
                ax.set_ylabel(title, fontsize=12)
                ax.set_title(title, fontsize=13, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def get_task_vector(
        self, pretrained_model: nn.Module, finetuned_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute task vector as difference between finetuned and pretrained parameters.
        
        Args:
            pretrained_model: Base pretrained model
            finetuned_model: Fine-tuned model
            
        Returns:
            Task vector as a 1D tensor
        """
        pretrained_sd = self.get_state_dict(pretrained_model)
        finetuned_sd = self.get_state_dict(finetuned_model)
        
        # Compute difference
        task_vector_sd = state_dict_sub(finetuned_sd, pretrained_sd)
        
        # Flatten to vector
        task_vector = state_dict_to_vector(task_vector_sd)
        
        return task_vector

    def get_state_dict(self, model: nn.Module):
        """Get state dict (trainable only if specified)."""
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()
