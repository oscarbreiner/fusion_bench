"""
Fastfood Subspace Task Vector Analysis

This module provides analysis of task vectors projected into the Fastfood subspace.
It computes the same metrics as the regular task vector analysis (cosine similarity, 
L2 distance, sign conflicts) but operates on task vectors that have been projected 
into the intrinsic subspace used by the Fastfood merging algorithm.

This allows us to understand how task vectors relate to each other in the compressed
subspace where the actual merging occurs.
"""

import logging
import os
import math
import hashlib
from typing import Dict, List, Optional, Tuple, Any

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
        lift: Function to lift vectors back from subspace (not used in this analysis)
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
class FastfoodSubspaceTaskVectorAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Task vector analysis in Fastfood subspace.
    
    This algorithm computes task vectors and projects them into the Fastfood subspace
    before computing pairwise similarities. This reveals how task vectors relate to 
    each other in the compressed space where Fastfood merging actually operates.
    
    Args:
        proj_ratio (float): Projection ratio for subspace dimensionality (0.0 to 1.0)
        use_G (bool): Whether to use diagonal scaling matrix G in Fastfood transform
        plot_heatmap (bool): Whether to generate and save visualization heatmaps
        trainable_only (bool): Whether to only analyze trainable parameters
        output_path (str, optional): Directory to save analysis results
        method_name (str, optional): Name for output files (e.g., "fastfood_subspace")
        device (str): Device to run computations on
        
    Outputs:
        - task_vector_subspace_cos_similarity_{method_name}.csv: Cosine similarity matrix in subspace
        - task_vector_subspace_l2_distance_{method_name}.csv: L2 distance matrix in subspace  
        - task_vector_subspace_sign_conflicts_{method_name}.csv: Sign conflict matrix in subspace
        - task_vector_subspace_analysis_{method_name}.pdf: Three-panel heatmap visualization
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        proj_ratio: float = 0.75,
        use_G: bool = False,
        plot_heatmap: bool = True,
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.plot_heatmap = plot_heatmap
        self.trainable_only = trainable_only
        self._output_path = output_path
        self.method_name = method_name or "fastfood_subspace"
        self.device = torch.device(device)

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    def get_state_dict(self, model: nn.Module) -> StateDictType:
        """Extract state dictionary from model."""
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()

    def get_task_vector(
        self, pretrained_model: nn.Module, finetuned_model: nn.Module
    ) -> torch.Tensor:
        """Compute task vector (finetuned - pretrained)."""
        task_vector = state_dict_sub(
            self.get_state_dict(finetuned_model),
            self.get_state_dict(pretrained_model),
        )
        return state_dict_to_vector(task_vector)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Execute the Fastfood subspace task vector analysis.
        
        Args:
            modelpool: Model pool containing pretrained and fine-tuned models
            
        Returns:
            The pretrained model from the model pool
        """
        log.info("Starting Fastfood subspace task vector analysis")
        log.info(f"Projection ratio: {self.proj_ratio}")
        log.info(f"Use G matrix: {self.use_G}")
        
        # Load pretrained model
        pretrained_model = modelpool.load_pretrained_model()

        # Compute task vectors for all fine-tuned models
        task_vectors = []
        model_names = []
        
        for name, finetuned_model in tqdm(
            modelpool.named_models(), 
            desc="Computing task vectors",
            total=len(modelpool)
        ):
            log.info(f"Computing task vector for {name}")
            task_vector = self.get_task_vector(pretrained_model, finetuned_model)
            task_vectors.append(task_vector.to(torch.float64))
            model_names.append(name)
        
        # Stack task vectors
        task_vectors = torch.stack(task_vectors, dim=0)  # [K, D]
        log.info(f"Task vectors shape: {task_vectors.shape}")
        
        # Determine global dimension and projection dimension
        global_dim = task_vectors.shape[-1]
        proj_dim = max(1, int(global_dim * self.proj_ratio))
        
        log.info(f"Global dimension: {global_dim}")
        log.info(f"Projection dimension: {proj_dim} ({self.proj_ratio:.1%} of global)")
        
        # Create Fastfood projection operator (using global seed key for consistency)
        seed_key = "global_analysis"
        fwd, _ = _fastfood_ops(
            global_dim=global_dim,
            proj_dim=proj_dim,
            seed_key=seed_key,
            device=self.device,
            use_G=self.use_G
        )
        
        # Project task vectors into subspace
        log.info("Projecting task vectors into Fastfood subspace")
        task_vectors_device = task_vectors.to(self.device)
        projected_task_vectors = []
        
        for i in range(task_vectors_device.shape[0]):
            projected = fwd(task_vectors_device[i])
            projected_task_vectors.append(projected.cpu())
        
        projected_task_vectors = torch.stack(projected_task_vectors, dim=0)
        log.info(f"Projected task vectors shape: {projected_task_vectors.shape}")
        
        # Initialize matrices for metrics in subspace
        num_models = len(model_names)
        cos_sim_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        l2_dist_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        sign_conflict_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        
        # Compute pairwise metrics in the subspace
        log.info("Computing pairwise metrics in subspace")
        for i in tqdm(range(num_models), desc="Computing similarities"):
            for j in range(i, num_models):
                vec_i = projected_task_vectors[i].to(torch.float64)
                vec_j = projected_task_vectors[j].to(torch.float64)
                
                # Cosine similarity
                if i == j:
                    # Diagonal entries should be exactly 1.0
                    cos_sim = 1.0
                else:
                    cos_sim = F.cosine_similarity(vec_i, vec_j, dim=0).item()
                    # Debug: Check for invalid cosine similarity values
                    if abs(cos_sim) > 1.0001:  # Allow small numerical tolerance
                        log.warning(f"Invalid cosine similarity {cos_sim:.6f} between models {i} and {j}")
                    
                cos_sim_matrix[i, j] = cos_sim
                cos_sim_matrix[j, i] = cos_sim
                
                # L2 distance
                l2_dist = torch.norm(vec_i - vec_j, p=2)
                l2_dist_matrix[i, j] = l2_dist
                l2_dist_matrix[j, i] = l2_dist
                
                # Sign conflicts
                signs_i = torch.sign(vec_i)
                signs_j = torch.sign(vec_j)
                non_zero_mask = (signs_i != 0) & (signs_j != 0)
                
                if non_zero_mask.sum() > 0:
                    conflicts = (signs_i[non_zero_mask] != signs_j[non_zero_mask])
                    conflict_ratio = conflicts.float().mean()
                else:
                    conflict_ratio = 0.0
                    
                sign_conflict_matrix[i, j] = conflict_ratio
                sign_conflict_matrix[j, i] = conflict_ratio

        # Convert to DataFrames
        cos_sim_df = pd.DataFrame(
            cos_sim_matrix.numpy(),
            index=model_names,
            columns=model_names,
        )
        
        l2_dist_df = pd.DataFrame(
            l2_dist_matrix.numpy(),
            index=model_names,
            columns=model_names,
        )
        
        sign_conflict_df = pd.DataFrame(
            sign_conflict_matrix.numpy(),
            index=model_names,
            columns=model_names,
        )

        # Print results
        log.info("Subspace Cosine Similarity Matrix:")
        log.info("\n" + str(cos_sim_df))
        log.info("\nSubspace L2 Distance Matrix:")
        log.info("\n" + str(l2_dist_df))
        log.info("\nSubspace Sign Conflict Matrix:")
        log.info("\n" + str(sign_conflict_df))
        
        # Save matrices
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            
            cos_sim_path = os.path.join(
                self.output_path, f"task_vector_subspace_cos_similarity_{self.method_name}.csv"
            )
            l2_dist_path = os.path.join(
                self.output_path, f"task_vector_subspace_l2_distance_{self.method_name}.csv"
            )
            sign_conflict_path = os.path.join(
                self.output_path, f"task_vector_subspace_sign_conflicts_{self.method_name}.csv"
            )
            
            cos_sim_df.to_csv(cos_sim_path)
            l2_dist_df.to_csv(l2_dist_path)
            sign_conflict_df.to_csv(sign_conflict_path)
            
            log.info(f"Saved subspace cosine similarity to: {cos_sim_path}")
            log.info(f"Saved subspace L2 distance to: {l2_dist_path}")
            log.info(f"Saved subspace sign conflicts to: {sign_conflict_path}")

        # Create visualization
        if self.plot_heatmap:
            self._plot_subspace_heatmap(cos_sim_df, l2_dist_df, sign_conflict_df)

        return pretrained_model

    def _plot_subspace_heatmap(
        self, 
        cos_sim_df: pd.DataFrame, 
        l2_dist_df: pd.DataFrame, 
        sign_conflict_df: pd.DataFrame
    ):
        """Generate and save three-panel heatmap for subspace analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        log.info("Creating subspace analysis heatmap")
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Helper function to get off-diagonal values for color scaling
        def get_off_diagonal_values(df):
            mask = ~np.eye(df.shape[0], dtype=bool)
            return df.values[mask]
        
        # Get off-diagonal values for color scale calculation
        cos_sim_off_diag = get_off_diagonal_values(cos_sim_df)
        l2_dist_off_diag = get_off_diagonal_values(l2_dist_df)
        sign_conflict_off_diag = get_off_diagonal_values(sign_conflict_df)
        
        # Plot 1: Cosine Similarity in Subspace
        cos_vmin, cos_vmax = cos_sim_off_diag.min(), cos_sim_off_diag.max()
        sns.heatmap(
            cos_sim_df,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            ax=axes[0],
            vmin=cos_vmin,
            vmax=cos_vmax,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        axes[0].set_title("Subspace Cosine Similarity\n(Fastfood Projection)", 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Tasks")
        axes[0].set_ylabel("Tasks")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=45)
        
        # Plot 2: L2 Distance in Subspace
        l2_vmin, l2_vmax = l2_dist_off_diag.min(), l2_dist_off_diag.max()
        sns.heatmap(
            l2_dist_df,
            annot=True,
            fmt=".3f",
            cmap="Reds",
            ax=axes[1],
            vmin=l2_vmin,
            vmax=l2_vmax,
            cbar_kws={'label': 'L2 Distance'}
        )
        axes[1].set_title("Subspace L2 Distance\n(Fastfood Projection)", 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Tasks")
        axes[1].set_ylabel("")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=45)
        
        # Plot 3: Sign Conflicts in Subspace
        sign_vmin, sign_vmax = sign_conflict_off_diag.min(), sign_conflict_off_diag.max()
        sns.heatmap(
            sign_conflict_df,
            annot=True,
            fmt=".3f",
            cmap="Oranges",
            ax=axes[2],
            vmin=sign_vmin,
            vmax=sign_vmax,
            cbar_kws={'label': 'Sign Conflict Ratio'}
        )
        axes[2].set_title("Subspace Sign Conflicts\n(Fastfood Projection)", 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel("Tasks")
        axes[2].set_ylabel("")
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].tick_params(axis='y', rotation=45)
        
        # Add overall title with projection info
        #fig.suptitle(f'Fastfood Subspace Task Vector Analysis (proj_ratio={self.proj_ratio:.1%})', fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for suptitle

        # Save the plot
        output_file = os.path.join(
            self.output_path, f"task_vector_subspace_analysis_{self.method_name}.pdf"
        )
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        
        log.info(f"Saved subspace analysis plot to: {output_file}")
