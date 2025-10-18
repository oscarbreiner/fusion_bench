"""
Intrinsic Dimension and Task Diversity Analysis

This module provides layer-wise analysis of task vectors focusing on:
1. Intrinsic Dimension: Effective rank of each layer's task vector
2. Task Diversity: Cross-task covariance rank at each layer

Key Features:
- Per-task effective rank computation at each layer
- Cross-task covariance analysis
- Visualization with error bars showing variance
- Support for all model architectures

The effective rank is computed using the entropy of normalized singular values:
    effective_rank = exp(entropy(normalized_singular_values))

This provides a smooth measure of the number of principal components needed
to capture the information in a matrix.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
class IntrinsicDimensionAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Intrinsic Dimension and Task Diversity Analysis of Task Vectors.
    
    Fixes vs. original:
      - Effective rank uses σ^2 (variance) spectrum, not σ.
      - Avoids arbitrary square-ish reshape: per-tensor SVD in native 2D (Linear, Conv).
      - Skips 1D params (bias, LayerNorm γ/β) for rank (can be added back if desired).
      - Diversity computed from centered task matrix SVD (max rank = min(T-1, D)).
      - Float64 SVD for stability + relative tolerance.
      - Adds relative metrics to CSV (no API change).
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        trainable_only: bool = False,
        output_path: Optional[str] = None,
        method_name: Optional[str] = None,
        device: str = "cuda",
        min_singular_value: float = 1e-6,   # kept for backward compatibility (not used as hard cutoff anymore)
        save_detailed_results: bool = True,
        create_plots: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trainable_only = trainable_only
        self._output_path = output_path
        self.method_name = method_name or "intrinsic_dim"
        self.device = torch.device(device)
        self.min_singular_value = min_singular_value
        self.save_detailed_results = save_detailed_results
        self.create_plots = create_plots

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    # ---------------------------
    # Helpers: naming & grouping
    # ---------------------------
    def _get_model_state_dict(self, model: nn.Module) -> StateDictType:
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()

    def _extract_layer_number(self, param_name: str) -> int:
        # CLIP-like
        match = re.search(r'\.resblocks\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        # other Transformer impls
        match = re.search(r'\.layers\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        match = re.search(r'\.layer\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        match = re.search(r'\.blocks?\.(\d+)\.', param_name)
        if match:
            return int(match.group(1))
        return -1

    def _group_params_by_layer(self, state_dict: StateDictType) -> Dict[int, Dict[str, torch.Tensor]]:
        layers = {}
        for param_name, param_tensor in state_dict.items():
            layer_idx = self._extract_layer_number(param_name)
            layers.setdefault(layer_idx, {})[param_name] = param_tensor
        return layers

    # ---------------------------
    # Linear algebra helpers
    # ---------------------------
    @staticmethod
    def _tensor_to_2d(W: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Map parameter tensor to a 2D matrix suitable for SVD:
          - Linear: (out, in)
          - Conv:   (out, in * k * k)
          - Others (1D like bias/LN): return None (skip)
        """
        if W.ndim == 2:
            return W
        if W.ndim == 4:
            out, in_c, k1, k2 = W.shape
            return W.reshape(out, in_c * k1 * k2)
        return None  # skip 1D (bias/LN) and other exotic shapes by default

    @staticmethod
    def _effective_rank_from_svals(S: torch.Tensor, eps: float = 1e-12) -> float:
        """
        Entropy-based effective rank from singular values, using variance spectrum λ_i ∝ σ_i^2.
        r_eff = exp( -∑ p_i log p_i ), where p_i = λ_i / ∑ λ_i.
        """
        if S.numel() == 0:
            return 0.0
        S = S.to(torch.float64)
        v = S ** 2
        vsum = torch.sum(v)
        if vsum <= 0:
            return 0.0
        p = (v / vsum).clamp_min(eps)
        H = -torch.sum(p * torch.log(p))
        return float(torch.exp(H))

    @staticmethod
    def _participation_ratio_from_svals(S: torch.Tensor, eps: float = 1e-12) -> float:
        """
        Participation Ratio (soft rank alternative):
        PR = (∑ λ_i)^2 / ∑ λ_i^2, with λ_i = σ_i^2.
        """
        if S.numel() == 0:
            return 0.0
        S = S.to(torch.float64)
        v = S ** 2
        num = torch.sum(v) ** 2
        den = torch.sum(v ** 2) + eps
        return float(num / den)

    # ---------------------------
    # Intrinsic dimension (per layer, per task)
    # ---------------------------
    def _layer_effective_rank_and_meta(self, layer_params: Dict[str, torch.Tensor]) -> Tuple[float, float, int]:
        """
        Compute parameter-count weighted effective rank over all 2D tensors in a layer.
        Returns:
          (weighted_r_eff, weighted_r_max, n_tensors_used)
        where r_max for a 2D matrix is min(out, in*), aggregated weighted by numel.
        """
        ranks, rmaxes, weights = [], [], []
        used = 0

        for name, W in layer_params.items():
            M = self._tensor_to_2d(W)
            if M is None:
                continue  # skip 1D (bias/LN) and non 2D
            M64 = M.to(torch.float64, non_blocking=True).detach().cpu()
            # numerically stable SVD (vals only)
            try:
                S = torch.linalg.svdvals(M64)
            except Exception as e:
                log.warning(f"SVD failed for param {name}: {e}")
                continue

            # relative tolerance: drop tiny singular values numerically
            if S.numel() == 0:
                continue
            smax = float(S.max())
            if smax <= 0:
                continue
            rel_tol = 1e-12 * max(M64.shape)  # similar to LAPACK-style
            S = S[S > rel_tol * smax]
            if S.numel() == 0:
                continue

            r_eff = self._effective_rank_from_svals(S)
            # max rank is min(m, n)
            r_max = min(M64.shape[0], M64.shape[1])

            ranks.append(r_eff)
            rmaxes.append(float(r_max))
            w = int(M64.numel())
            weights.append(w)
            used += 1

        if used == 0:
            return 0.0, 0.0, 0

        weighted_r_eff = float(np.average(ranks, weights=weights))
        weighted_r_max = float(np.average(rmaxes, weights=weights))
        return weighted_r_eff, weighted_r_max, used

    def _compute_intrinsic_dimension_per_layer(
        self, 
        task_vectors_by_layer: Dict[str, Dict[int, Dict[str, torch.Tensor]]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute per-task effective rank at each layer and aggregate across tasks.
        Uses variance-spectrum entropy rank per 2D param, weighted by numel; skips 1D params.
        """
        all_layers = set()
        for task_layers in task_vectors_by_layer.values():
            all_layers.update(task_layers.keys())
        layer_indices = sorted([l for l in all_layers if l >= 0])

        results = {}
        for layer_idx in tqdm(layer_indices, desc="Computing intrinsic dimensions"):
            per_task_ranks = []
            per_task_rmax = []

            for task_name, task_layers in task_vectors_by_layer.items():
                if layer_idx not in task_layers:
                    continue

                r_eff_w, r_max_w, n_used = self._layer_effective_rank_and_meta(task_layers[layer_idx])
                if n_used == 0:
                    continue
                per_task_ranks.append(r_eff_w)
                per_task_rmax.append(r_max_w)

            if per_task_ranks:
                ranks_arr = np.asarray(per_task_ranks, dtype=float)
                rmax_arr  = np.asarray(per_task_rmax,  dtype=float)
                # relative effective rank per task then averaged
                rel_per_task = np.divide(ranks_arr, np.maximum(rmax_arr, 1e-12))
                results[layer_idx] = {
                    'per_task_effective_ranks': per_task_ranks,
                    'mean_effective_rank': float(np.mean(ranks_arr)),
                    'std_effective_rank': float(np.std(ranks_arr)),
                    'min_effective_rank': float(np.min(ranks_arr)),
                    'max_effective_rank': float(np.max(ranks_arr)),
                    'mean_relative_effective_rank': float(np.mean(rel_per_task)),
                    'std_relative_effective_rank': float(np.std(rel_per_task)),
                    'num_tasks': int(len(per_task_ranks))
                }

        return results

    # ---------------------------
    # Task diversity (across tasks, per layer)
    # ---------------------------
    def _compute_task_diversity_per_layer(
        self, 
        task_vectors_by_layer: Dict[str, Dict[int, Dict[str, torch.Tensor]]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        For each layer:
          - Stack centered task vectors X ∈ R^{T×D} (skipping 1D params)
          - SVD on X (not Gram reshape)
          - Effective rank from σ^2 spectrum (equiv. covariance eigenvalues)
          - Max possible rank = min(T-1, D)
        """
        all_layers = set()
        for task_layers in task_vectors_by_layer.values():
            all_layers.update(task_layers.keys())
        layer_indices = sorted([l for l in all_layers if l >= 0])

        results = {}

        for layer_idx in tqdm(layer_indices, desc="Computing task diversity"):
            task_vecs = []
            task_names = []

            for task_name, task_layers in task_vectors_by_layer.items():
                if layer_idx not in task_layers:
                    continue
                # flatten only 2D tensors (skip 1D)
                flat_list = []
                for pname, P in task_layers[layer_idx].items():
                    M = self._tensor_to_2d(P)
                    if M is None:
                        continue
                    flat_list.append(M.reshape(-1))
                if not flat_list:
                    continue
                v = torch.cat(flat_list)  # D
                task_vecs.append(v)
                task_names.append(task_name)

            T = len(task_vecs)
            if T < 2:
                continue

            X = torch.stack(task_vecs).to(torch.float64)  # T x D
            # center across tasks
            X = X - X.mean(dim=0, keepdim=True)

            # SVD on centered X: singular values relate to covariance eigs
            try:
                S = torch.linalg.svdvals(X.cpu())
            except Exception as e:
                log.warning(f"SVD failed for diversity at layer {layer_idx}: {e}")
                continue

            # numerical trimming
            if S.numel() == 0:
                continue
            smax = float(S.max())
            if smax <= 0:
                continue
            rel_tol = 1e-12 * max(X.shape)
            S = S[S > rel_tol * smax]
            if S.numel() == 0:
                continue

            cov_eff_rank = self._effective_rank_from_svals(S)
            pr_rank      = self._participation_ratio_from_svals(S)

            D_layer = X.shape[1]
            rmax = min(T - 1, D_layer)  # after centering
            rel_cov_rank = float(cov_eff_rank / max(rmax, 1e-12))
            rel_pr_rank  = float(pr_rank      / max(rmax, 1e-12))

            results[layer_idx] = {
                'covariance_effective_rank': float(cov_eff_rank),
                'covariance_pr_rank': float(pr_rank),
                'relative_covariance_effective_rank': rel_cov_rank,
                'relative_covariance_pr_rank': rel_pr_rank,
                'num_tasks': int(T),
                'num_parameters': int(D_layer),
                'max_possible_rank': int(rmax),
                'task_names': task_names
            }

        return results

    # ---------------------------
    # Viz + save (unchanged API; adds columns)
    # ---------------------------
    def _create_visualization_plots(
        self, 
        intrinsic_dim_results: Dict[int, Dict[str, Any]],
        task_diversity_results: Dict[int, Dict[str, Any]]
    ):
        if not intrinsic_dim_results and not task_diversity_results:
            log.warning("No results to plot")
            return
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Intrinsic Dimension
        if intrinsic_dim_results:
            layer_indices = sorted(intrinsic_dim_results.keys())
            mean_ranks = [intrinsic_dim_results[idx]['mean_effective_rank'] for idx in layer_indices]
            std_ranks = [intrinsic_dim_results[idx]['std_effective_rank'] for idx in layer_indices]
            axes[0].errorbar(layer_indices, mean_ranks, yerr=std_ranks, 
                             marker='o', linestyle='-', linewidth=2, markersize=6,
                             color='steelblue', ecolor='lightsteelblue', 
                             capsize=5, capthick=2, label='Mean ± Std')
            axes[0].set_title('Intrinsic Dimension: Per-Task Effective Rank Across Layers', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Layer Index', fontsize=12)
            axes[0].set_ylabel('Effective Rank', fontsize=12)
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].legend(fontsize=10)
            axes[0].fill_between(layer_indices, 
                                 np.array(mean_ranks) - np.array(std_ranks),
                                 np.array(mean_ranks) + np.array(std_ranks),
                                 alpha=0.2, color='steelblue')
        
        # Plot 2: Task Diversity
        if task_diversity_results:
            layer_indices_div = sorted(task_diversity_results.keys())
            covariance_ranks = [task_diversity_results[idx]['covariance_effective_rank'] 
                                for idx in layer_indices_div]
            axes[1].plot(layer_indices_div, covariance_ranks, 
                         marker='s', linestyle='-', linewidth=2, markersize=6,
                         color='coral', label='Covariance Effective Rank')
            # Correct max possible rank line (min(T-1, D))
            max_line = [task_diversity_results[idx]['max_possible_rank'] for idx in layer_indices_div]
            axes[1].plot(layer_indices_div, max_line, linestyle='--', linewidth=1.5, alpha=0.7, label='Max Rank (min(T-1, D))')
            axes[1].set_title('Task Diversity: Cross-Task Covariance Effective Rank Across Layers', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Layer Index', fontsize=12)
            axes[1].set_ylabel('Covariance Effective Rank', fontsize=12)
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].legend(fontsize=10)
        
        fig.suptitle(f'Intrinsic Dimension and Task Diversity Analysis - {self.method_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.3)
        plot_path = os.path.join(self.output_path, f'intrinsic_dimension_analysis_{self.method_name}.pdf')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        log.info(f"Saved visualization plots to {plot_path}")
        
        # ---- Relative effective-rank plot (new PDF) ----
        if intrinsic_dim_results:
            rel_path = os.path.join(
                self.output_path,
                f'intrinsic_dimension_relative_analysis_{self.method_name}.pdf'
            )

            fig2, ax2 = plt.subplots(figsize=(14, 5))
            layer_indices = sorted(intrinsic_dim_results.keys())
            rel_means = [
                intrinsic_dim_results[i].get('mean_relative_effective_rank', np.nan)
                for i in layer_indices
            ]
            rel_stds = [
                intrinsic_dim_results[i].get('std_relative_effective_rank', np.nan)
                for i in layer_indices
            ]

            ax2.errorbar(
                layer_indices, rel_means, yerr=rel_stds,
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color='darkgreen', ecolor='mediumseagreen',
                capsize=5, capthick=2, label='Mean ± Std'
            )
            ax2.fill_between(
                layer_indices,
                np.array(rel_means) - np.array(rel_stds),
                np.array(rel_means) + np.array(rel_stds),
                alpha=0.2, color='mediumseagreen'
            )
            ax2.set_title(
                'Relative Intrinsic Dimension (Effective Rank / Full Rank)',
                fontsize=14, fontweight='bold'
            )
            ax2.set_xlabel('Layer Index', fontsize=12)
            ax2.set_ylabel('Relative Effective Rank', fontsize=12)
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=10)
            fig2.tight_layout()
            plt.savefig(rel_path, bbox_inches='tight', dpi=300)
            plt.close(fig2)
            log.info(f"Saved relative-rank visualization to {rel_path}")

    def _save_results(
        self, 
        intrinsic_dim_results: Dict[int, Dict[str, Any]],
        task_diversity_results: Dict[int, Dict[str, Any]]
    ):
        # Intrinsic results (+ relative columns, backward compatible)
        if intrinsic_dim_results:
            intrinsic_data = []
            for layer_idx in sorted(intrinsic_dim_results.keys()):
                res = intrinsic_dim_results[layer_idx]
                row = {
                    'layer_index': layer_idx,
                    'mean_effective_rank': res['mean_effective_rank'],
                    'std_effective_rank': res['std_effective_rank'],
                    'min_effective_rank': res['min_effective_rank'],
                    'max_effective_rank': res['max_effective_rank'],
                    'num_tasks': res['num_tasks']
                }
                # New relative metrics:
                if 'mean_relative_effective_rank' in res:
                    row['mean_relative_effective_rank'] = res['mean_relative_effective_rank']
                    row['std_relative_effective_rank'] = res['std_relative_effective_rank']
                intrinsic_data.append(row)
            intrinsic_df = pd.DataFrame(intrinsic_data)
            intrinsic_path = os.path.join(self.output_path, f'intrinsic_dimension_results_{self.method_name}.csv')
            intrinsic_df.to_csv(intrinsic_path, index=False)
            log.info(f"Saved intrinsic dimension results to {intrinsic_path}")
        
        # Diversity results (+ relative columns)
        if task_diversity_results:
            diversity_data = []
            for layer_idx in sorted(task_diversity_results.keys()):
                res = task_diversity_results[layer_idx]
                diversity_data.append({
                    'layer_index': layer_idx,
                    'covariance_effective_rank': res['covariance_effective_rank'],
                    'covariance_pr_rank': res['covariance_pr_rank'],
                    'relative_covariance_effective_rank': res['relative_covariance_effective_rank'],
                    'relative_covariance_pr_rank': res['relative_covariance_pr_rank'],
                    'num_tasks': res['num_tasks'],
                    'num_parameters': res['num_parameters'],
                    'max_possible_rank': res['max_possible_rank']
                })
            diversity_df = pd.DataFrame(diversity_data)
            diversity_path = os.path.join(self.output_path, f'task_diversity_results_{self.method_name}.csv')
            diversity_df.to_csv(diversity_path, index=False)
            log.info(f"Saved task diversity results to {diversity_path}")
        
        # Summary (kept structure; mentions new relative metrics if available)
        summary_path = os.path.join(self.output_path, f'analysis_summary_{self.method_name}.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Intrinsic Dimension and Task Diversity Analysis\n")
            f.write(f"Method: {self.method_name}\n")
            f.write("=" * 80 + "\n\n")
            
            if intrinsic_dim_results:
                f.write("INTRINSIC DIMENSION ANALYSIS (Per-Task Effective Rank)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Number of layers analyzed: {len(intrinsic_dim_results)}\n")
                all_means = [res['mean_effective_rank'] for res in intrinsic_dim_results.values()]
                f.write(f"Overall mean effective rank: {np.mean(all_means):.4f}\n")
                f.write(f"Overall std effective rank: {np.std(all_means):.4f}\n")
                f.write(f"Min mean effective rank: {np.min(all_means):.4f}\n")
                f.write(f"Max mean effective rank: {np.max(all_means):.4f}\n\n")

                # If relative metrics present:
                if 'mean_relative_effective_rank' in next(iter(intrinsic_dim_results.values())):
                    rel_means = [res['mean_relative_effective_rank'] for res in intrinsic_dim_results.values()]
                    f.write(f"Overall mean relative effective rank: {np.mean(rel_means):.4f}\n\n")

                f.write("Per-layer summary:\n")
                for layer_idx in sorted(intrinsic_dim_results.keys()):
                    res = intrinsic_dim_results[layer_idx]
                    base = (f"  Layer {layer_idx:3d}: mean={res['mean_effective_rank']:8.4f}, "
                            f"std={res['std_effective_rank']:8.4f}, "
                            f"min={res['min_effective_rank']:8.4f}, "
                            f"max={res['max_effective_rank']:8.4f}")
                    if 'mean_relative_effective_rank' in res:
                        base += (f", rel_mean={res['mean_relative_effective_rank']:7.4f}, "
                                 f"rel_std={res['std_relative_effective_rank']:7.4f}")
                    f.write(base + "\n")
                f.write("\n")
            
            if task_diversity_results:
                f.write("TASK DIVERSITY ANALYSIS (Cross-Task Covariance Rank)\n")
                f.write("-" * 80 + "\n")
                f.write(f"Number of layers analyzed: {len(task_diversity_results)}\n")
                all_cov = [res['covariance_effective_rank'] for res in task_diversity_results.values()]
                f.write(f"Overall mean covariance effective rank: {np.mean(all_cov):.4f}\n")
                f.write(f"Overall std covariance effective rank: {np.std(all_cov):.4f}\n")
                f.write(f"Min covariance effective rank: {np.min(all_cov):.4f}\n")
                f.write(f"Max covariance effective rank: {np.max(all_cov):.4f}\n\n")

                f.write("Per-layer summary:\n")
                for layer_idx in sorted(task_diversity_results.keys()):
                    res = task_diversity_results[layer_idx]
                    f.write(
                        f"  Layer {layer_idx:3d}: cov_rank={res['covariance_effective_rank']:8.4f}, "
                        f"PR={res['covariance_pr_rank']:8.4f}, "
                        f"rel_cov={res['relative_covariance_effective_rank']:7.4f}, "
                        f"rel_pr={res['relative_covariance_pr_rank']:7.4f}, "
                        f"T={res['num_tasks']}, D={res['num_parameters']}, "
                        f"rmax={res['max_possible_rank']}\n"
                    )

        log.info(f"Saved analysis summary to {summary_path}")

    # ---------------------------
    # Task vector construction
    # ---------------------------
    def _compute_task_vector_by_layer(
        self, 
        pretrained_model: nn.Module, 
        finetuned_model: nn.Module
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        pretrained_state = self._get_model_state_dict(pretrained_model)
        finetuned_state = self._get_model_state_dict(finetuned_model)
        task_vector_state = state_dict_sub(finetuned_state, pretrained_state)
        layer_groups = self._group_params_by_layer(task_vector_state)
        return layer_groups

    # ---------------------------
    # Orchestrator
    # ---------------------------
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        log.info("Starting Intrinsic Dimension and Task Diversity Analysis")
        log.info(f"Method: {self.method_name}")
        log.info(f"Trainable parameters only: {self.trainable_only}")
        
        os.makedirs(self.output_path, exist_ok=True)
        
        pretrained_model = modelpool.load_pretrained_model()
        
        task_vectors_by_layer = {}
        log.info("Computing task vectors by layer for each model...")
        for name, finetuned_model in tqdm(modelpool.named_models(), desc="Computing task vectors"):
            task_vectors_by_layer[name] = self._compute_task_vector_by_layer(
                pretrained_model, finetuned_model
            )
        log.info(f"Computed task vectors for {len(task_vectors_by_layer)} models")
        
        log.info("Computing intrinsic dimension analysis...")
        intrinsic_dim_results = self._compute_intrinsic_dimension_per_layer(task_vectors_by_layer)
        
        log.info("Computing task diversity analysis...")
        task_diversity_results = self._compute_task_diversity_per_layer(task_vectors_by_layer)
        
        if self.save_detailed_results:
            log.info("Saving detailed results...")
            self._save_results(intrinsic_dim_results, task_diversity_results)
        
        if self.create_plots:
            log.info("Creating visualization plots...")
            self._create_visualization_plots(intrinsic_dim_results, task_diversity_results)
        
        log.info("Intrinsic Dimension and Task Diversity Analysis complete!")
        return pretrained_model


# Configuration mapping for Hydra
IntrinsicDimensionAnalysis._config_mapping = IntrinsicDimensionAnalysis._config_mapping