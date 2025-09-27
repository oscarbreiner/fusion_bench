import logging
import os
from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd
import torch
import torch.utils
from numpy.typing import NDArray
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
class TaskVectorCosSimilarity(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Computes and analyzes multiple metrics between task vectors of models in a model pool.

    This algorithm extracts task vectors from fine-tuned models by computing the difference
    between their parameters and a pretrained base model. It then calculates three pairwise
    metrics between all task vectors:
    
    1. Cosine Similarity: Angular similarity between task vectors
    2. L2 Distance: Euclidean distance between task vectors  
    3. Sign Conflicts: Ratio of parameters with conflicting signs

    The task vector for a model is defined as:
        task_vector = finetuned_model_params - pretrained_model_params

    Args:
        plot_heatmap (bool): Whether to generate and save a three-panel heatmap visualization
        trainable_only (bool, optional): If True, only consider trainable parameters
            when computing task vectors. Defaults to True.
        max_points_per_model (int, optional): Maximum number of parameters to sample
            per model for memory efficiency. If None, uses all parameters.
        output_path (str, optional): Directory to save outputs. If None, uses the
            fabric logger directory.
        method_name (str, optional): Name of the merging method for file naming.
            Prevents overwriting when analyzing multiple methods.

    Outputs:
        - task_vector_cos_similarity_{method_name}.csv: Pairwise cosine similarity matrix
        - task_vector_l2_distance_{method_name}.csv: Pairwise L2 distance matrix
        - task_vector_sign_conflicts_{method_name}.csv: Pairwise sign conflict matrix
        - task_vector_analysis_{method_name}.pdf: Three-panel heatmap (if plot_heatmap=True)

    Returns:
        The pretrained model from the model pool.

    Example:
        ```python
        >>> algorithm = TaskVectorCosSimilarity(
        ...     plot_heatmap=True,
        ...     trainable_only=True,
        ...     method_name="task_arithmetic",
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
        max_points_per_model: Optional[int] = None,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_path = output_path
        self.plot_heatmap = plot_heatmap
        self.trainable_only = trainable_only
        self.max_points_per_model = max_points_per_model
        self.method_name = method_name or "default"

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Execute the comprehensive task vector analysis.

        This method:
        1. Loads the pretrained base model from the model pool
        2. Computes task vectors for each fine-tuned model
        3. Calculates pairwise cosine similarities, L2 distances, and sign conflicts
        4. Saves all matrices as CSV files
        5. Optionally generates and saves a three-panel heatmap visualization

        Args:
            modelpool (BaseModelPool): Pool containing pretrained and fine-tuned models

        Returns:
            nn.Module: The pretrained model from the model pool
        """
        pretrained_model = modelpool.load_pretrained_model()

        task_vectors = []
        for name, finetuned_model in tqdm(
            modelpool.named_models(), total=len(modelpool)
        ):
            print(f"computing task vectors for {name}")
            task_vectors.append(
                self.get_task_vector(pretrained_model, finetuned_model).to(
                    torch.float64
                )
            )
        task_vectors = torch.stack(task_vectors, dim=0)

        # Initialize matrices for all three metrics
        num_models = len(modelpool)
        cos_sim_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        l2_dist_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        sign_conflict_matrix = torch.zeros(num_models, num_models, dtype=torch.float64)
        
        # Compute all pairwise metrics
        for i in range(num_models):
            for j in range(i, num_models):
                assert task_vectors[i].size() == task_vectors[j].size()
                
                # Cosine similarity
                if i == j:
                    # Diagonal entries should be exactly 1.0
                    cos_sim = 1.0
                else:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        task_vectors[i], task_vectors[j], dim=0
                    ).item()
                    
                cos_sim_matrix[i, j] = cos_sim
                cos_sim_matrix[j, i] = cos_sim
                
                # L2 distance
                l2_dist = torch.norm(task_vectors[i] - task_vectors[j], p=2)
                l2_dist_matrix[i, j] = l2_dist
                l2_dist_matrix[j, i] = l2_dist
                
                # Sign conflicts
                signs_i = torch.sign(task_vectors[i])
                signs_j = torch.sign(task_vectors[j])
                non_zero_mask = (signs_i != 0) & (signs_j != 0)
                if non_zero_mask.sum() > 0:
                    conflicts = (signs_i[non_zero_mask] != signs_j[non_zero_mask])
                    conflict_ratio = conflicts.float().mean()
                else:
                    conflict_ratio = 0.0
                sign_conflict_matrix[i, j] = conflict_ratio
                sign_conflict_matrix[j, i] = conflict_ratio

        # Convert matrices to pandas DataFrames
        cos_sim_df = pd.DataFrame(
            cos_sim_matrix.numpy(),
            index=modelpool.model_names,
            columns=modelpool.model_names,
        )
        
        l2_dist_df = pd.DataFrame(
            l2_dist_matrix.numpy(),
            index=modelpool.model_names,
            columns=modelpool.model_names,
        )
        
        sign_conflict_df = pd.DataFrame(
            sign_conflict_matrix.numpy(),
            index=modelpool.model_names,
            columns=modelpool.model_names,
        )

        print("Cosine Similarity Matrix:")
        print(cos_sim_df)
        print("\nL2 Distance Matrix:")
        print(l2_dist_df)
        print("\nSign Conflict Matrix:")
        print(sign_conflict_df)
        
        # Save all matrices to CSV files
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            cos_sim_df.to_csv(
                os.path.join(self.output_path, f"task_vector_cos_similarity_{self.method_name}.csv")
            )
            l2_dist_df.to_csv(
                os.path.join(self.output_path, f"task_vector_l2_distance_{self.method_name}.csv")
            )
            sign_conflict_df.to_csv(
                os.path.join(self.output_path, f"task_vector_sign_conflicts_{self.method_name}.csv")
            )

        if self.plot_heatmap:
            self._plot_three_panel_heatmap(cos_sim_df, l2_dist_df, sign_conflict_df)

        return pretrained_model

    def _plot_three_panel_heatmap(self, cos_sim_df: pd.DataFrame, l2_dist_df: pd.DataFrame, sign_conflict_df: pd.DataFrame):
        """
        Generate and save a three-panel heatmap visualization.

        Creates three side-by-side heatmaps showing cosine similarity, L2 distance, 
        and sign conflicts between task vectors. The diagonal identity values are 
        excluded from color scale calculation to improve visualization of off-diagonal differences.

        Args:
            cos_sim_df (pd.DataFrame): Cosine similarity matrix
            l2_dist_df (pd.DataFrame): L2 distance matrix  
            sign_conflict_df (pd.DataFrame): Sign conflict ratio matrix

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.colors as mcolors
        import numpy as np

        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Helper function to get off-diagonal values for color scale calculation
        def get_off_diagonal_values(df):
            mask = ~np.eye(df.shape[0], dtype=bool)
            return df.values[mask]
        
        # Get off-diagonal values for color scale calculation
        cos_sim_off_diag = get_off_diagonal_values(cos_sim_df)
        l2_dist_off_diag = get_off_diagonal_values(l2_dist_df) 
        sign_conflict_off_diag = get_off_diagonal_values(sign_conflict_df)
        
        # Plot 1: Cosine Similarity (more dissimilar = darker)
        cos_vmin, cos_vmax = cos_sim_off_diag.min(), cos_sim_off_diag.max()
        sns.heatmap(
            cos_sim_df,
            annot=True,
            fmt=".2f",
            cmap="Blues",  # Regular Blues colormap (lower values = lighter, higher values = darker)
            ax=axes[0],
            vmin=cos_vmin,
            vmax=cos_vmax,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        axes[0].set_title("Cosine Similarity\n(More dissimilar = darker)", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Tasks")
        axes[0].set_ylabel("Tasks")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=45)
        
        # Plot 2: L2 Distance
        l2_vmin, l2_vmax = l2_dist_off_diag.min(), l2_dist_off_diag.max()
        sns.heatmap(
            l2_dist_df,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            ax=axes[1],
            vmin=l2_vmin,
            vmax=l2_vmax,
            cbar_kws={'label': 'L2 Distance'}
        )
        axes[1].set_title("L2 Distance\n(Higher = more different)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Tasks")
        axes[1].set_ylabel("")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=45)
        
        # Plot 3: Sign Conflicts
        sign_vmin, sign_vmax = sign_conflict_off_diag.min(), sign_conflict_off_diag.max()
        sns.heatmap(
            sign_conflict_df,
            annot=True,
            fmt=".2f", 
            cmap="Oranges",
            ax=axes[2],
            vmin=sign_vmin,
            vmax=sign_vmax,
            cbar_kws={'label': 'Sign Conflict Ratio'}
        )
        axes[2].set_title("Sign Conflicts\n(Higher = more conflicts)", fontsize=14, fontweight='bold')
        axes[2].set_xlabel("Tasks")
        axes[2].set_ylabel("")
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].tick_params(axis='y', rotation=45)

        # Add overall title
        #fig.suptitle(f"Task Vector Analysis - {self.method_name}", fontsize=16, fontweight='bold')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for suptitle

        # Save the plot with method-specific filename
        output_file = os.path.join(self.output_path, f"task_vector_analysis_{self.method_name}.pdf")
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

    def get_task_vector(
        self, pretrained_model: nn.Module, finetuned_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute the task vector for a fine-tuned model.

        The task vector represents the parameter changes from pretraining to
        fine-tuning and is computed as:
            task_vector = finetuned_params - pretrained_params

        Args:
            pretrained_model (nn.Module): The base pretrained model
            finetuned_model (nn.Module): The fine-tuned model for a specific task

        Returns:
            torch.Tensor: Flattened task vector containing parameter differences.
                If max_points_per_model is set, the vector may be downsampled.

        Note:
            - Converts parameters to float64 for numerical precision
            - Supports optional downsampling for memory efficiency
            - Uses only trainable parameters if trainable_only=True
        """
        task_vector = state_dict_sub(
            self.get_state_dict(finetuned_model),
            self.get_state_dict(pretrained_model),
        )
        task_vector = state_dict_to_vector(task_vector)

        task_vector = task_vector.cpu().float().numpy()
        # downsample if necessary
        if (
            self.max_points_per_model is not None
            and self.max_points_per_model > 0
            and task_vector.shape[0] > self.max_points_per_model
        ):
            log.info(
                f"Downsampling task vectors to {self.max_points_per_model} points."
            )
            indices = np.random.choice(
                task_vector.shape[0], self.max_points_per_model, replace=False
            )
            task_vector = task_vector[indices].copy()

        task_vector = torch.from_numpy(task_vector)
        return task_vector

    def get_state_dict(self, model: nn.Module):
        """
        Extract the state dictionary from a model.

        Args:
            model (nn.Module): The model to extract parameters from

        Returns:
            Dict[str, torch.Tensor]: State dictionary containing model parameters.
                Returns only trainable parameters if trainable_only=True,
                otherwise returns all parameters.
        """
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()
