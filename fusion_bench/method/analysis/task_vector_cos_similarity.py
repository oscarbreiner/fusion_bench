import logging
import os
import math
import hashlib
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
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u, v = x[..., j], x[..., j + h]
                x[..., j], x[..., j + h] = u + v, u - v
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
        # Subspace analysis parameters
        proj_ratio: float = 0.75,
        use_G: bool = False,
        analyze_subspace: bool = True,
        device: str = "cuda",
        # FastFood-specific parameters for method name generation
        merge_func: Optional[str] = None,
        subspace_scope: Optional[str] = None,
        merge_where: Optional[str] = None,
        # Configurable metrics selection
        compute_cosine_similarity: bool = True,
        compute_l2_distance: bool = True,
        compute_sign_conflicts: bool = True,
        compute_fisher_divergence: bool = True,
        compute_spectral_angle: bool = True,
        compute_magnitude_ratio: bool = True,
        compute_kl_divergence: bool = True,
        compute_js_divergence: bool = True,
        compute_wasserstein_distance: bool = True,
        # Visualization options
        create_comprehensive_plots: bool = True,
        create_basic_plots: bool = True,
        plot_only_enabled_metrics: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_path = output_path
        self.plot_heatmap = plot_heatmap
        self.trainable_only = trainable_only
        self.max_points_per_model = max_points_per_model
        
        # Subspace analysis parameters
        self.proj_ratio = float(proj_ratio)
        self.use_G = bool(use_G)
        self.analyze_subspace = analyze_subspace
        self.device = torch.device(device)
        
        # FastFood-specific parameters for method name generation
        self.merge_func = merge_func
        self.subspace_scope = subspace_scope
        self.merge_where = merge_where
        
        # Configurable metrics selection
        self.compute_cosine_similarity = compute_cosine_similarity
        self.compute_l2_distance = compute_l2_distance
        self.compute_sign_conflicts = compute_sign_conflicts
        self.compute_fisher_divergence = compute_fisher_divergence
        self.compute_spectral_angle = compute_spectral_angle
        self.compute_magnitude_ratio = compute_magnitude_ratio
        self.compute_kl_divergence = compute_kl_divergence
        self.compute_js_divergence = compute_js_divergence
        self.compute_wasserstein_distance = compute_wasserstein_distance
        
        # Visualization options
        self.create_comprehensive_plots = create_comprehensive_plots
        self.create_basic_plots = create_basic_plots
        self.plot_only_enabled_metrics = plot_only_enabled_metrics
        
        # Generate unique method name including FastFood parameters if provided
        self.method_name = self._generate_method_name(method_name)

    def _generate_method_name(self, base_method_name: Optional[str] = None) -> str:
        """
        Generate a unique method name that includes FastFood parameters to prevent file overwrites.
        
        Args:
            base_method_name: Base method name to use. If None, uses "default"
            
        Returns:
            str: Unique method name incorporating FastFood parameters
        """
        if base_method_name is None:
            base_method_name = "default"
        
        # If FastFood parameters are provided, create a unique identifier
        if any([self.merge_func, self.subspace_scope, self.merge_where]) or self.proj_ratio != 0.75:
            parts = [base_method_name]
            
            # Add projection ratio if it's not default
            if self.proj_ratio != 0.75:
                parts.append(f"proj{self.proj_ratio}")
            
            # Add merge function if provided
            if self.merge_func:
                parts.append(self.merge_func)
                
            # Add subspace scope if provided and not default
            if self.subspace_scope and self.subspace_scope != 'global':
                parts.append(f"scope_{self.subspace_scope}")
                
            # Add merge location if provided and not default
            if self.merge_where and self.merge_where != 'subspace':
                parts.append(f"where_{self.merge_where}")
            
            return "_".join(parts)
        else:
            return base_method_name

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

        # Analyze in original space
        print(f"Analyzing task vectors in original space (method: {self.method_name})")
        self._analyze_and_save_matrices(task_vectors, modelpool, space_suffix="original")
        print(f"✓ Original space analysis completed")
        
        # Analyze in FastFood subspace if requested and parameters are available
        print(f"Subspace analysis check: analyze_subspace={self.analyze_subspace}, "
              f"has_proj_ratio={hasattr(self, 'proj_ratio')}, "
              f"proj_ratio={getattr(self, 'proj_ratio', None)}")
        
        if (self.analyze_subspace and 
            hasattr(self, 'proj_ratio') and 
            self.proj_ratio is not None and 
            self.proj_ratio > 0):
            
            print(f"✓ Starting FastFood subspace analysis with proj_ratio={self.proj_ratio}")
            
            try:
                # Project task vectors to FastFood subspace
                global_dim = task_vectors.shape[-1]
                proj_dim = max(1, int(global_dim * self.proj_ratio))
                
                # Create unique seed key for reproducible projections
                seed_key = f"task_vector_analysis_{self.method_name}_{global_dim}_{proj_dim}"
                
                # Get FastFood projection operators
                fwd, lift = _fastfood_ops(
                    global_dim=global_dim,
                    proj_dim=proj_dim,
                    seed_key=seed_key,
                    device=self.device,
                    use_G=self.use_G
                )
                
                # Project all task vectors to subspace
                task_vectors_subspace = []
                for i in range(task_vectors.shape[0]):
                    task_vec = task_vectors[i].to(self.device)
                    projected = fwd(task_vec)
                    task_vectors_subspace.append(projected.cpu().to(torch.float64))
                
                task_vectors_subspace = torch.stack(task_vectors_subspace, dim=0)
                
                # Analyze in subspace
                print(f"✓ Running subspace analysis...")
                self._analyze_and_save_matrices(task_vectors_subspace, modelpool, space_suffix="subspace")
                print(f"✓ Subspace analysis completed - _subspace files saved")
                
                # Optionally analyze lifted vectors (projected back to original space)
                print(f"✓ Running lifted space analysis (subspace -> original)")
                task_vectors_lifted = []
                for i in range(task_vectors_subspace.shape[0]):
                    subspace_vec = task_vectors_subspace[i].to(self.device)
                    lifted = lift(subspace_vec)
                    task_vectors_lifted.append(lifted.cpu().to(torch.float64))
                
                task_vectors_lifted = torch.stack(task_vectors_lifted, dim=0)
                
                # Analyze lifted vectors
                self._analyze_and_save_matrices(task_vectors_lifted, modelpool, space_suffix="lifted")
                print(f"✓ Lifted space analysis completed - _lifted files saved")
            
            except Exception as e:
                print(f"✗ Error during subspace analysis: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ Skipping subspace analysis - conditions not met")

        return pretrained_model

    def _analyze_and_save_matrices(self, task_vectors: torch.Tensor, modelpool: BaseModelPool, space_suffix: str = ""):
        """
        Analyze task vectors and save similarity matrices.
        
        Args:
            task_vectors: Task vectors to analyze [num_models, vector_dim]
            modelpool: Model pool for naming
            space_suffix: Suffix to add to filenames (e.g., "original", "subspace", "lifted")
        """
        # Initialize matrices based on enabled metrics
        num_models = len(modelpool)
        
        # Initialize basic metrics matrices (conditionally)
        cos_sim_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_cosine_similarity else None
        l2_dist_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_l2_distance else None
        sign_conflict_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_sign_conflicts else None
        
        # Initialize advanced metrics matrices (conditionally)
        fisher_divergence_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_fisher_divergence else None
        spectral_angle_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_spectral_angle else None
        magnitude_ratio_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_magnitude_ratio else None
        kl_divergence_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_kl_divergence else None
        js_divergence_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_js_divergence else None
        wasserstein_distance_matrix = torch.zeros(num_models, num_models, dtype=torch.float64) if self.compute_wasserstein_distance else None
        
        # Compute all pairwise metrics (only for enabled metrics)
        for i in range(num_models):
            for j in range(i, num_models):
                assert task_vectors[i].size() == task_vectors[j].size()
                
                # Cosine similarity (if enabled)
                if self.compute_cosine_similarity:
                    if i == j:
                        # Diagonal entries should be exactly 1.0
                        cos_sim = 1.0
                    else:
                        cos_sim = torch.nn.functional.cosine_similarity(
                            task_vectors[i], task_vectors[j], dim=0
                        ).item()
                        
                    cos_sim_matrix[i, j] = cos_sim
                    cos_sim_matrix[j, i] = cos_sim
                
                # L2 distance (if enabled)
                if self.compute_l2_distance:
                    l2_dist = torch.norm(task_vectors[i] - task_vectors[j], p=2)
                    l2_dist_matrix[i, j] = l2_dist
                    l2_dist_matrix[j, i] = l2_dist
                
                # Sign conflicts (if enabled)
                if self.compute_sign_conflicts:
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
                
                # Advanced metrics (only compute for off-diagonal elements and if enabled)
                if i != j:
                    # Fisher Information Divergence (if enabled)
                    if self.compute_fisher_divergence:
                        fisher_div = self._compute_fisher_divergence(task_vectors[i], task_vectors[j])
                        fisher_divergence_matrix[i, j] = fisher_div
                        fisher_divergence_matrix[j, i] = fisher_div
                    
                    # Spectral Angle Mapper (if enabled)
                    if self.compute_spectral_angle:
                        spectral_angle = self._compute_spectral_angle(task_vectors[i], task_vectors[j])
                        spectral_angle_matrix[i, j] = spectral_angle
                        spectral_angle_matrix[j, i] = spectral_angle
                    
                    # Magnitude Ratio (if enabled)
                    if self.compute_magnitude_ratio:
                        mag_ratio = self._compute_magnitude_ratio(task_vectors[i], task_vectors[j])
                        magnitude_ratio_matrix[i, j] = mag_ratio
                        magnitude_ratio_matrix[j, i] = 1.0 / mag_ratio if mag_ratio != 0 else float('inf')
                    
                    # KL Divergence (if enabled)
                    if self.compute_kl_divergence:
                        kl_div = self._compute_kl_divergence(task_vectors[i], task_vectors[j])
                        kl_divergence_matrix[i, j] = kl_div
                        kl_divergence_matrix[j, i] = self._compute_kl_divergence(task_vectors[j], task_vectors[i])
                    
                    # Jensen-Shannon Divergence (if enabled)
                    if self.compute_js_divergence:
                        js_div = self._compute_js_divergence(task_vectors[i], task_vectors[j])
                        js_divergence_matrix[i, j] = js_div
                        js_divergence_matrix[j, i] = js_div
                    
                    # Wasserstein Distance (if enabled)
                    if self.compute_wasserstein_distance:
                        wasserstein_dist = self._compute_wasserstein_distance(task_vectors[i], task_vectors[j])
                        wasserstein_distance_matrix[i, j] = wasserstein_dist
                        wasserstein_distance_matrix[j, i] = wasserstein_dist

        # Convert matrices to pandas DataFrames (only for enabled metrics)
        cos_sim_df = None
        l2_dist_df = None
        sign_conflict_df = None
        
        if self.compute_cosine_similarity:
            cos_sim_df = pd.DataFrame(
                cos_sim_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
            print(f"Cosine Similarity Matrix ({space_suffix}):")
            print(cos_sim_df)
        
        if self.compute_l2_distance:
            l2_dist_df = pd.DataFrame(
                l2_dist_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
            print(f"\nL2 Distance Matrix ({space_suffix}):")
            print(l2_dist_df)
        
        if self.compute_sign_conflicts:
            sign_conflict_df = pd.DataFrame(
                sign_conflict_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
            print(f"\nSign Conflict Matrix ({space_suffix}):")
            print(sign_conflict_df)
        
        # Generate filenames with space suffix
        if space_suffix:
            suffix = f"_{space_suffix}"
        else:
            suffix = ""
        
        # Save basic metrics to CSV files (only if computed and output path is provided)
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            
            if self.compute_cosine_similarity and cos_sim_df is not None:
                cos_sim_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_cos_similarity_{self.method_name}{suffix}.csv")
                )
            
            if self.compute_l2_distance and l2_dist_df is not None:
                l2_dist_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_l2_distance_{self.method_name}{suffix}.csv")
                )
            
            if self.compute_sign_conflicts and sign_conflict_df is not None:
                sign_conflict_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_sign_conflicts_{self.method_name}{suffix}.csv")
                )

        # Create advanced metrics DataFrames (only for enabled metrics)
        fisher_div_df = None
        spectral_angle_df = None
        magnitude_ratio_df = None
        kl_divergence_df = None
        js_divergence_df = None
        wasserstein_df = None
        
        if self.compute_fisher_divergence and fisher_divergence_matrix is not None:
            fisher_div_df = pd.DataFrame(
                fisher_divergence_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
            print(f"\nFisher Divergence Matrix ({space_suffix}):")
            print(fisher_div_df)
        
        if self.compute_spectral_angle and spectral_angle_matrix is not None:
            spectral_angle_df = pd.DataFrame(
                spectral_angle_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
            print(f"\nSpectral Angle Matrix ({space_suffix}):")
            print(spectral_angle_df)
        
        if self.compute_magnitude_ratio and magnitude_ratio_matrix is not None:
            magnitude_ratio_df = pd.DataFrame(
                magnitude_ratio_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
            print(f"\nMagnitude Ratio Matrix ({space_suffix}):")
            print(magnitude_ratio_df)
        
        if self.compute_kl_divergence and kl_divergence_matrix is not None:
            kl_divergence_df = pd.DataFrame(
                kl_divergence_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
        
        if self.compute_js_divergence and js_divergence_matrix is not None:
            js_divergence_df = pd.DataFrame(
                js_divergence_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
        
        if self.compute_wasserstein_distance and wasserstein_distance_matrix is not None:
            wasserstein_df = pd.DataFrame(
                wasserstein_distance_matrix.numpy(),
                index=modelpool.model_names,
                columns=modelpool.model_names,
            )
        
        # Save advanced metrics to CSV files (only if computed and output path provided)
        if self.output_path is not None:
            if self.compute_fisher_divergence and fisher_div_df is not None:
                fisher_div_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_fisher_divergence_{self.method_name}{suffix}.csv")
                )
            
            if self.compute_spectral_angle and spectral_angle_df is not None:
                spectral_angle_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_spectral_angle_{self.method_name}{suffix}.csv")
                )
            
            if self.compute_magnitude_ratio and magnitude_ratio_df is not None:
                magnitude_ratio_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_magnitude_ratio_{self.method_name}{suffix}.csv")
                )
            
            if self.compute_kl_divergence and kl_divergence_df is not None:
                kl_divergence_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_kl_divergence_{self.method_name}{suffix}.csv")
                )
            
            if self.compute_js_divergence and js_divergence_df is not None:
                js_divergence_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_js_divergence_{self.method_name}{suffix}.csv")
                )
            
            if self.compute_wasserstein_distance and wasserstein_df is not None:
                wasserstein_df.to_csv(
                    os.path.join(self.output_path, f"task_vector_wasserstein_distance_{self.method_name}{suffix}.csv")
                )

        if self.plot_heatmap:
            # Create visualizations based on configuration
            if self.create_comprehensive_plots:
                self._plot_comprehensive_heatmaps(cos_sim_df, l2_dist_df, sign_conflict_df, 
                                                fisher_div_df, spectral_angle_df, magnitude_ratio_df,
                                                js_divergence_df, wasserstein_df, space_suffix)
            
            if self.create_basic_plots:
                self._plot_basic_five_panel(cos_sim_df, l2_dist_df, sign_conflict_df, 
                                          fisher_div_df, js_divergence_df, space_suffix)

    def _compute_fisher_divergence(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute approximated Fisher Information Divergence.
        
        This measures how much the "information geometry" differs between two task vectors,
        which is particularly relevant for understanding optimization landscapes.
        """
        # Normalize vectors to unit length
        vec1_norm = vec1 / (torch.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (torch.norm(vec2) + 1e-8)
        
        # Approximate Fisher divergence using squared differences of gradients
        # This approximates the difference in curvature information
        diff = vec1_norm - vec2_norm
        fisher_approx = torch.sum(diff ** 2).item()
        
        return fisher_approx

    def _compute_spectral_angle(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute Spectral Angle Mapper (SAM) between two vectors.
        
        SAM measures the angle between vectors in high-dimensional space,
        which is invariant to magnitude and useful for understanding directional differences.
        """
        dot_product = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cos_angle = dot_product / (norm1 * norm2)
        # Clamp to avoid numerical issues
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(torch.abs(cos_angle)).item()
        
        return angle

    def _compute_magnitude_ratio(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute magnitude ratio between two vectors.
        
        This is crucial for understanding how FastFood subspace projection affects
        the relative scaling of different task vectors.
        """
        norm1 = torch.norm(vec1).item()
        norm2 = torch.norm(vec2).item()
        
        if norm2 == 0:
            return float('inf') if norm1 > 0 else 1.0
            
        return norm1 / norm2

    def _compute_kl_divergence(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute KL divergence between normalized absolute values of vectors.
        
        This measures information-theoretic distance, useful for understanding
        how parameter importance distributions differ between tasks.
        """
        # Convert to probability distributions using absolute values
        abs_vec1 = torch.abs(vec1) + 1e-8
        abs_vec2 = torch.abs(vec2) + 1e-8
        
        p = abs_vec1 / torch.sum(abs_vec1)
        q = abs_vec2 / torch.sum(abs_vec2)
        
        # KL(P||Q) = sum(p * log(p/q))
        kl_div = torch.sum(p * torch.log(p / q)).item()
        
        return kl_div if not torch.isnan(torch.tensor(kl_div)) else 0.0

    def _compute_js_divergence(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute Jensen-Shannon divergence (symmetric version of KL divergence).
        
        This provides a bounded, symmetric measure of distributional difference
        that's particularly useful for comparing parameter importance patterns.
        """
        # Convert to probability distributions
        abs_vec1 = torch.abs(vec1) + 1e-8
        abs_vec2 = torch.abs(vec2) + 1e-8
        
        p = abs_vec1 / torch.sum(abs_vec1)
        q = abs_vec2 / torch.sum(abs_vec2)
        m = 0.5 * (p + q)
        
        # JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        kl_pm = torch.sum(p * torch.log(p / m))
        kl_qm = torch.sum(q * torch.log(q / m))
        
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js_div.item() if not torch.isnan(js_div) else 0.0

    def _compute_wasserstein_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute 1-Wasserstein (Earth Mover's) distance between vectors.
        
        This measures the minimum cost to transform one distribution into another,
        providing insights into how "effort" is distributed differently across parameters.
        """
        # Sort absolute values to compute 1-Wasserstein distance efficiently
        abs_vec1 = torch.abs(vec1)
        abs_vec2 = torch.abs(vec2)
        
        # Normalize to probability distributions
        sum1 = torch.sum(abs_vec1)
        sum2 = torch.sum(abs_vec2)
        
        if sum1 == 0 or sum2 == 0:
            return 0.0
            
        p = abs_vec1 / sum1
        q = abs_vec2 / sum2
        
        # Sort the distributions
        p_sorted, _ = torch.sort(p)
        q_sorted, _ = torch.sort(q)
        
        # Compute cumulative distributions
        p_cumsum = torch.cumsum(p_sorted, dim=0)
        q_cumsum = torch.cumsum(q_sorted, dim=0)
        
        # 1-Wasserstein distance is the L1 distance between CDFs
        wasserstein_dist = torch.sum(torch.abs(p_cumsum - q_cumsum)).item()
        
        return wasserstein_dist

    def _plot_comprehensive_heatmaps(self, cos_sim_df: pd.DataFrame, l2_dist_df: pd.DataFrame, 
                                   sign_conflict_df: pd.DataFrame, fisher_div_df: pd.DataFrame,
                                   spectral_angle_df: pd.DataFrame, magnitude_ratio_df: pd.DataFrame,
                                   js_divergence_df: pd.DataFrame, wasserstein_df: pd.DataFrame, 
                                   space_suffix: str = ""):
        """
        Generate and save a three-panel heatmap visualization.

        Creates three side-by-side heatmaps showing cosine similarity, L2 distance, 
        and sign conflicts between task vectors. The diagonal identity values are 
        excluded from color scale calculation to improve visualization of off-diagonal differences.

        Args:
            cos_sim_df (pd.DataFrame): Cosine similarity matrix
            l2_dist_df (pd.DataFrame): L2 distance matrix  
            sign_conflict_df (pd.DataFrame): Sign conflict ratio matrix
            space_suffix (str): Suffix indicating the analysis space (e.g., "original", "subspace", "lifted")

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.colors as mcolors
        import numpy as np

        # Define plot configurations for each metric (only include enabled metrics)
        plot_configs = []
        
        if cos_sim_df is not None:
            plot_configs.append((cos_sim_df, "Cosine Similarity\n(Higher = more similar)", "Blues", "Cosine Similarity"))
        
        if l2_dist_df is not None:
            plot_configs.append((l2_dist_df, "L2 Distance\n(Higher = more different)", "Reds", "L2 Distance"))
        
        if sign_conflict_df is not None:
            plot_configs.append((sign_conflict_df, "Sign Conflicts\n(Higher = more conflicts)", "Oranges", "Sign Conflict Ratio"))
        
        if fisher_div_df is not None:
            plot_configs.append((fisher_div_df, "Fisher Divergence\n(Higher = more divergent)", "Purples", "Fisher Divergence"))
        
        if spectral_angle_df is not None:
            plot_configs.append((spectral_angle_df, "Spectral Angle\n(Higher = more angular difference)", "Greens", "Spectral Angle (rad)"))
        
        if magnitude_ratio_df is not None:
            plot_configs.append((magnitude_ratio_df, "Magnitude Ratio\n(Log scale, 1 = equal)", "coolwarm", "Log Magnitude Ratio"))
        
        if js_divergence_df is not None:
            plot_configs.append((js_divergence_df, "Jensen-Shannon Divergence\n(Higher = more different distributions)", "plasma", "JS Divergence"))
        
        if wasserstein_df is not None:
            plot_configs.append((wasserstein_df, "Wasserstein Distance\n(Higher = more transport cost)", "viridis", "Wasserstein Distance"))
        
        # Skip plotting if no metrics are enabled
        if not plot_configs:
            print(f"No metrics enabled for plotting in space: {space_suffix}")
            return

        # Calculate optimal subplot layout based on number of metrics
        num_metrics = len(plot_configs)
        if num_metrics <= 3:
            rows, cols = 1, num_metrics
            figsize = (6 * num_metrics, 6)
        elif num_metrics <= 6:
            rows, cols = 2, 3
            figsize = (18, 12)
        else:
            rows, cols = 2, 4
            figsize = (24, 12)
        
        # Create figure with dynamic layout
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_metrics == 1:
            axes = [axes]  # Make it a list for consistent indexing
        elif num_metrics > 1 and rows == 1:
            axes = axes  # Already a list
        else:
            axes = axes.flatten()
        
        # Helper function to get off-diagonal values for color scale calculation
        def get_off_diagonal_values(df):
            mask = ~np.eye(df.shape[0], dtype=bool)
            return df.values[mask]
        
        for idx, (df, title, cmap, label) in enumerate(plot_configs):
            ax = axes[idx]
            
            # Get off-diagonal values for color scaling (exclude diagonal for most metrics)
            if "Magnitude Ratio" in title:
                # For magnitude ratio, use log scale and center around 1
                plot_df = np.log10(df.replace([np.inf, -np.inf], np.nan).fillna(1))
                vmin, vmax = -2, 2  # Log scale from 0.01 to 100
                fmt = ".2f"
            else:
                plot_df = df
                off_diag_values = get_off_diagonal_values(df)
                if len(off_diag_values) > 0:
                    vmin, vmax = off_diag_values.min(), off_diag_values.max()
                else:
                    vmin, vmax = df.min().min(), df.max().max()
                fmt = ".3f" if vmax < 1 else ".2f"
            
            # Create heatmap
            sns.heatmap(
                plot_df,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': label},
                square=True,
                linewidths=0.5
            )
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel("Tasks" if idx >= (num_metrics - cols) else "")
            ax.set_ylabel("Tasks" if idx % cols == 0 else "")
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=45)
        
        # Hide any unused subplots
        total_subplots = rows * cols
        for idx in range(num_metrics, total_subplots):
            if idx < len(axes):
                axes[idx].set_visible(False)
        
        # Add overall title
        space_title = f" ({space_suffix.title()} Space)" if space_suffix else ""
        fig.suptitle(f"Comprehensive Task Vector Analysis - {self.method_name}{space_title}", 
                    fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)

        # Save the comprehensive plot
        if space_suffix:
            suffix = f"_{space_suffix}"
        else:
            suffix = ""
        output_file = os.path.join(self.output_path, f"task_vector_comprehensive_analysis_{self.method_name}{suffix}.pdf")
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        
        # Also create a simplified 3-panel plot for backward compatibility
        self._plot_basic_three_panel(cos_sim_df, l2_dist_df, sign_conflict_df, space_suffix)

    def _plot_basic_five_panel(self, cos_sim_df: pd.DataFrame, l2_dist_df: pd.DataFrame, 
                             sign_conflict_df: pd.DataFrame, fisher_div_df: pd.DataFrame,
                             js_divergence_df: pd.DataFrame, space_suffix: str = ""):
        """Create a 5-panel plot with the most important metrics for fast visualization."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        def get_off_diagonal_values(df):
            mask = ~np.eye(df.shape[0], dtype=bool)
            return df.values[mask]
        
        # Plot configurations (only include available DataFrames)
        plots = []
        if cos_sim_df is not None:
            plots.append((cos_sim_df, "Cosine Similarity\n(Higher = more similar)", "Blues", "Cosine Similarity"))
        if l2_dist_df is not None:
            plots.append((l2_dist_df, "L2 Distance\n(Higher = more different)", "Reds", "L2 Distance"))
        if sign_conflict_df is not None:
            plots.append((sign_conflict_df, "Sign Conflicts\n(Higher = more conflicts)", "Oranges", "Sign Conflict Ratio"))
        if fisher_div_df is not None:
            plots.append((fisher_div_df, "Fisher Divergence\n(Higher = more divergent)", "Purples", "Fisher Divergence"))
        if js_divergence_df is not None:
            plots.append((js_divergence_df, "JS Divergence\n(Higher = more different)", "plasma", "JS Divergence"))
        
        # Skip plotting if no basic metrics are enabled
        if not plots:
            print(f"No basic metrics enabled for plotting in space: {space_suffix}")
            return
        
        # Adjust figure layout based on number of available plots
        fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 6))
        if len(plots) == 1:
            axes = [axes]
        
        for idx, (df, title, cmap, label) in enumerate(plots):
            off_diag = get_off_diagonal_values(df)
            vmin, vmax = off_diag.min(), off_diag.max()
            
            sns.heatmap(df, annot=True, fmt=".3f", cmap=cmap, ax=axes[idx],
                       vmin=vmin, vmax=vmax, cbar_kws={'label': label})
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Tasks")
            if idx == 0:
                axes[idx].set_ylabel("Tasks")
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].tick_params(axis='y', rotation=45)
        
        plt.tight_layout()
        
        # Save basic plot
        if space_suffix:
            suffix = f"_{space_suffix}"
        else:
            suffix = ""
        output_file = os.path.join(self.output_path, f"task_vector_analysis_{self.method_name}{suffix}.pdf")
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

    def _plot_basic_three_panel(self, cos_sim_df: pd.DataFrame, l2_dist_df: pd.DataFrame, 
                              sign_conflict_df: pd.DataFrame, space_suffix: str = ""):
        """Backward compatibility wrapper for 3-panel plotting."""
        self._plot_basic_five_panel(cos_sim_df, l2_dist_df, sign_conflict_df, None, None, space_suffix)

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
