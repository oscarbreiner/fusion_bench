"""
Linear Weights Similarity Analysis (Original Domain)

Computes pairwise similarities between task vectors considering ONLY 2D linear weights
in the original parameter space (no projection).

Metrics computed:
1. Cosine Similarity: Angular similarity between task vectors
2. L2 Distance: Euclidean distance between task vectors
3. Sign Conflicts: Ratio of parameters with conflicting signs
4. Jaccard Similarity: Intersection over union of significant parameters

This analysis focuses exclusively on linear transformation weights (2D tensors),
excluding biases, normalization parameters, and embeddings.
"""

import hashlib
import logging
import math
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import (
    StateDictType,
    trainable_state_dict,
)
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

log = logging.getLogger(__name__)


@auto_register_config
class LinearWeightsSimilarityOriginal(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    """
    Computes pairwise similarity metrics between task vectors for 2D linear weights only.
    
    This analysis filters the model to only consider 2D weight tensors (linear layers),
    excluding biases, layer norms, embeddings, and other non-linear parameters.
    
    Args:
        plot_heatmap (bool): Whether to generate and save heatmap visualizations
        trainable_only (bool): If True, only consider trainable parameters (default: True)
        output_path (str, optional): Directory to save outputs. If None, uses fabric logger directory
        method_name (str, optional): Name suffix for output files
        jaccard_threshold (float): Threshold for considering parameters "significant" in Jaccard similarity (default: 0.01)
        
    Outputs:
        - linear_weights_cosine_similarity_{method_name}.csv: Pairwise cosine similarity matrix
        - linear_weights_l2_distance_{method_name}.csv: Pairwise L2 distance matrix
        - linear_weights_sign_conflicts_{method_name}.csv: Pairwise sign conflict matrix
        - linear_weights_jaccard_similarity_{method_name}.csv: Pairwise Jaccard similarity matrix
        - linear_weights_analysis_{method_name}.pdf: Four-panel heatmap visualization
        
    Example:
        ```python
        >>> algorithm = LinearWeightsSimilarityOriginal(
        ...     plot_heatmap=True,
        ...     trainable_only=True,
        ...     method_name="original_space",
        ...     output_path="/path/to/outputs"
        ... )
        >>> result = algorithm.run(modelpool)
        ```
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        plot_heatmap: bool,
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        jaccard_threshold: float = 0.01,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_path = output_path
        self.plot_heatmap = plot_heatmap
        self.trainable_only = trainable_only
        self.method_name = method_name if method_name is not None else "original_linear_weights"
        self.jaccard_threshold = float(jaccard_threshold)
        self.device = torch.device(device)

    @property
    def output_path(self):
        if self._output_path is not None:
            return self._output_path
        if hasattr(self, "fabric") and self.fabric is not None:
            return self.fabric.logger.log_dir
        return None

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Run similarity analysis on linear weights only.
        
        Args:
            modelpool: Model pool containing pretrained and fine-tuned models
            
        Returns:
            The pretrained model from the model pool
        """
        print(f"\n{'='*80}")
        print(f"Linear Weights Similarity Analysis (Original Domain)")
        print(f"{'='*80}")
        
        # Load models
        with self.profile("loading models"):
            pretrained_model = modelpool.load_model("_pretrained_")
            model_names = [name for name in modelpool.model_names]
            models = {name: modelpool.load_model(name) for name in model_names}

        print(f"Loaded {len(models)} fine-tuned models")

        # Extract task vectors for 2D linear weights only
        with self.profile("computing task vectors"):
            task_vectors = []
            linear_keys = None
            
            for name in tqdm(model_names, desc="Computing task vectors"):
                task_vector = self.get_task_vector(pretrained_model, models[name])
                
                # Filter to only 2D tensors (linear weights)
                if linear_keys is None:
                    linear_keys = [k for k, v in task_vector.items() if v.ndim == 2]
                    print(f"\nFiltered to {len(linear_keys)} 2D linear weight tensors out of {len(task_vector)} total parameters")
                    print(f"Example linear keys: {linear_keys[:5]}")
                
                # Flatten and concatenate only 2D tensors
                task_vector_flat = torch.cat([
                    task_vector[k].flatten().to(self.device) 
                    for k in linear_keys
                ])
                task_vectors.append(task_vector_flat)
            
            task_vectors = torch.stack(task_vectors)  # [num_models, total_params]
            print(f"Task vectors shape: {task_vectors.shape}")
            print(f"Total parameters in linear weights: {task_vectors.shape[1]:,}")

        # Analyze and save matrices
        with self.profile("computing similarities"):
            self._analyze_and_save_matrices(task_vectors, modelpool)

        self.print_profile_summary()
        return pretrained_model

    def _analyze_and_save_matrices(self, task_vectors: torch.Tensor, modelpool: BaseModelPool):
        """
        Analyze task vectors and save similarity matrices.
        
        Args:
            task_vectors: Task vectors to analyze [num_models, vector_dim]
            modelpool: Model pool for naming
        """
        num_models = len(modelpool)
        model_names = [name for name in modelpool.model_names]
        
        # Initialize matrices
        cos_sim_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        l2_dist_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        sign_conflict_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        jaccard_sim_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        
        # Compute all pairwise metrics
        print("\nComputing pairwise metrics...")
        for i in tqdm(range(num_models), desc="Computing similarities"):
            for j in range(i, num_models):
                vec_i = task_vectors[i]
                vec_j = task_vectors[j]
                
                # Cosine Similarity
                cos_sim = self._compute_cosine_similarity(vec_i, vec_j)
                cos_sim_matrix[i, j] = cos_sim_matrix[j, i] = cos_sim
                
                # L2 Distance
                l2_dist = torch.norm(vec_i - vec_j).item()
                l2_dist_matrix[i, j] = l2_dist_matrix[j, i] = l2_dist
                
                # Sign Conflicts
                if i != j:
                    sign_conflict = self._compute_sign_conflict(vec_i, vec_j)
                    sign_conflict_matrix[i, j] = sign_conflict_matrix[j, i] = sign_conflict
                
                # Jaccard Similarity
                jaccard_sim = self._compute_jaccard_similarity(vec_i, vec_j)
                jaccard_sim_matrix[i, j] = jaccard_sim_matrix[j, i] = jaccard_sim

        # Convert to DataFrames
        cos_sim_df = pd.DataFrame(
            cos_sim_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names
        )
        
        l2_dist_df = pd.DataFrame(
            l2_dist_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names
        )
        
        sign_conflict_df = pd.DataFrame(
            sign_conflict_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names
        )
        
        jaccard_sim_df = pd.DataFrame(
            jaccard_sim_matrix.cpu().numpy(),
            index=model_names,
            columns=model_names
        )

        # Save to CSV files
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            
            cos_sim_path = os.path.join(
                self.output_path, f"linear_weights_cosine_similarity_{self.method_name}.csv"
            )
            cos_sim_df.to_csv(cos_sim_path)
            print(f"Saved cosine similarity to {cos_sim_path}")
            
            l2_dist_path = os.path.join(
                self.output_path, f"linear_weights_l2_distance_{self.method_name}.csv"
            )
            l2_dist_df.to_csv(l2_dist_path)
            print(f"Saved L2 distance to {l2_dist_path}")
            
            sign_conflict_path = os.path.join(
                self.output_path, f"linear_weights_sign_conflicts_{self.method_name}.csv"
            )
            sign_conflict_df.to_csv(sign_conflict_path)
            print(f"Saved sign conflicts to {sign_conflict_path}")
            
            jaccard_sim_path = os.path.join(
                self.output_path, f"linear_weights_jaccard_similarity_{self.method_name}.csv"
            )
            jaccard_sim_df.to_csv(jaccard_sim_path)
            print(f"Saved Jaccard similarity to {jaccard_sim_path}")

        # Plot heatmaps
        if self.plot_heatmap:
            self._plot_heatmaps(cos_sim_df, l2_dist_df, sign_conflict_df, jaccard_sim_df)

    def _compute_cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()

    def _compute_sign_conflict(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute sign conflict ratio between two vectors."""
        signs1 = torch.sign(vec1)
        signs2 = torch.sign(vec2)
        
        # Only consider non-zero elements
        mask = (signs1 != 0) & (signs2 != 0)
        if mask.sum() == 0:
            return 0.0
        
        conflicts = (signs1[mask] != signs2[mask]).sum().item()
        total = mask.sum().item()
        
        return conflicts / total

    def _compute_jaccard_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute Jaccard similarity between two vectors.
        
        Treats parameters as "significant" if their absolute value exceeds the threshold,
        then computes intersection over union of significant parameters.
        """
        # Identify significant parameters
        sig1 = torch.abs(vec1) > self.jaccard_threshold
        sig2 = torch.abs(vec2) > self.jaccard_threshold
        
        # Compute intersection and union
        intersection = (sig1 & sig2).sum().item()
        union = (sig1 | sig2).sum().item()
        
        if union == 0:
            return 1.0  # Both empty sets
        
        return intersection / union

    def _plot_heatmaps(self, cos_sim_df: pd.DataFrame, l2_dist_df: pd.DataFrame, 
                      sign_conflict_df: pd.DataFrame, jaccard_sim_df: pd.DataFrame):
        """Plot four-panel heatmap visualization."""
        if self.output_path is None:
            print("Warning: output_path is None, skipping plot generation")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Cosine Similarity
        sns.heatmap(
            cos_sim_df, annot=True, fmt=".3f", cmap="RdYlGn", 
            vmin=-1, vmax=1, ax=axes[0, 0], cbar_kws={'label': 'Cosine Similarity'}
        )
        axes[0, 0].set_title("Cosine Similarity (Linear Weights)", fontsize=14, fontweight='bold')
        
        # L2 Distance
        sns.heatmap(
            l2_dist_df, annot=True, fmt=".2e", cmap="YlOrRd", 
            ax=axes[0, 1], cbar_kws={'label': 'L2 Distance'}
        )
        axes[0, 1].set_title("L2 Distance (Linear Weights)", fontsize=14, fontweight='bold')
        
        # Sign Conflicts
        sns.heatmap(
            sign_conflict_df, annot=True, fmt=".3f", cmap="RdYlGn_r", 
            vmin=0, vmax=1, ax=axes[1, 0], cbar_kws={'label': 'Sign Conflict Ratio'}
        )
        axes[1, 0].set_title("Sign Conflicts (Linear Weights)", fontsize=14, fontweight='bold')
        
        # Jaccard Similarity
        sns.heatmap(
            jaccard_sim_df, annot=True, fmt=".3f", cmap="RdYlGn", 
            vmin=0, vmax=1, ax=axes[1, 1], cbar_kws={'label': 'Jaccard Similarity'}
        )
        axes[1, 1].set_title("Jaccard Similarity (Linear Weights)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = os.path.join(
            self.output_path, f"linear_weights_analysis_{self.method_name}.pdf"
        )
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_file}")
        plt.close()

    def get_task_vector(
        self, pretrained_model: nn.Module, finetuned_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task vector as the difference between fine-tuned and pretrained models.
        
        Args:
            pretrained_model: Base pretrained model
            finetuned_model: Fine-tuned model
            
        Returns:
            Dictionary of task vectors (parameter differences)
        """
        pretrained_state_dict = self.get_state_dict(pretrained_model)
        finetuned_state_dict = self.get_state_dict(finetuned_model)
        
        task_vector = state_dict_sub(finetuned_state_dict, pretrained_state_dict)
        return task_vector

    def get_state_dict(self, model: nn.Module):
        """Get state dict, optionally filtering to trainable parameters only."""
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()
