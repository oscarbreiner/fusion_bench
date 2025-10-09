"""
Subspace Sign Conflict Analysis

This module analyzes how sign conflicts and distribution characteristics change
when task vectors are projected into FastFood/SRHT subspaces.

Key metrics:
- Sign conflict rate (original vs subspace vs lifted)
- Sign agreement strength (how strongly tasks agree on directions)
- Jensen-Shannon Divergence (distribution similarity)
- Magnitude preservation (signal strength in subspace)
- Reconstruction error (subspace fidelity)

Usage:
    from fusion_bench.method.analysis.subspace_sign_conflict_analysis import SubspaceSignConflictAnalysis
    
    analyzer = SubspaceSignConflictAnalysis(
        proj_ratios=[0.1, 0.25, 0.5, 0.75, 0.95],
        use_G=True,
        output_path="./sign_conflict_analysis"
    )
    analyzer.run(modelpool)
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from torch import Tensor, nn

from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType

# Import FastFood utilities
try:
    from fusion_bench.method.fastfood_merging.fastfood_utils import (
        create_fastfood_ops,
        layer_key,
    )
except ImportError:
    from fusion_bench.method.fastfood_merging.fastfood_merging import (
        _fastfood_ops as create_fastfood_ops,
        _layer_key as layer_key,
    )

log = logging.getLogger(__name__)


class SubspaceSignConflictAnalysis(SimpleProfilerMixin, BaseAlgorithm):
    """
    Analyze sign conflicts and distribution changes in FastFood subspaces.
    
    This analysis helps understand why TIES merging underperforms in subspace
    by measuring:
    1. Sign conflict rates before and after projection
    2. Distribution similarity (JSD) between original and subspace
    3. Signal preservation and reconstruction quality
    """

    def __init__(
        self,
        proj_ratios: List[float] = [0.1, 0.25, 0.5, 0.75, 0.95],
        use_G: bool = True,
        subspace_scope: str = "global",
        device: str = "cuda",
        output_path: str = "./sign_conflict_analysis",
        trainable_only: bool = True,
        sample_size: int = None,  # If set, randomly sample this many tensors
        **kwargs: Any,
    ):
        """
        Initialize the analysis.
        
        Args:
            proj_ratios: List of projection ratios to analyze
            use_G: Whether to use Gaussian scaling in FastFood
            subspace_scope: "global", "layer", or "per_tensor"
            device: Compute device
            output_path: Directory to save analysis results
            trainable_only: Only analyze trainable parameters
            sample_size: If set, randomly sample this many tensors (for speed)
        """
        super().__init__(**kwargs)
        self.proj_ratios = proj_ratios
        self.use_G = use_G
        self.subspace_scope = subspace_scope
        self.device = torch.device(device)
        self.output_path = Path(output_path)
        self.trainable_only = trainable_only
        self.sample_size = sample_size
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Initialized SubspaceSignConflictAnalysis")
        log.info(f"  Projection ratios: {proj_ratios}")
        log.info(f"  Subspace scope: {subspace_scope}")
        log.info(f"  Output: {output_path}")

    def compute_sign_conflicts(self, task_vectors: Tensor) -> Dict[str, float]:
        """
        Compute sign conflict metrics for a set of task vectors.
        
        Args:
            task_vectors: [K, D] tensor where K is number of tasks
            
        Returns:
            Dict with conflict metrics:
            - conflict_rate: Fraction of dimensions with sign conflicts
            - agreement_strength: Average strength of sign agreement
            - majority_margin: How much the majority exceeds minority
        """
        K = task_vectors.shape[0]
        if K < 2:
            return {
                "conflict_rate": 0.0,
                "agreement_strength": 1.0,
                "majority_margin": 1.0,
            }
        
        # Get signs: {-1, 0, +1}
        signs = torch.sign(task_vectors)
        
        # Count positive, negative, zero
        pos_count = (signs > 0).sum(dim=0).float()
        neg_count = (signs < 0).sum(dim=0).float()
        zero_count = (signs == 0).sum(dim=0).float()
        
        # Conflict: positions where both positive and negative exist
        has_positive = pos_count > 0
        has_negative = neg_count > 0
        conflicts = has_positive & has_negative
        
        # Only consider non-zero positions for conflict rate
        non_zero_positions = (pos_count + neg_count) > 0
        conflict_rate = conflicts.sum().item() / max(non_zero_positions.sum().item(), 1)
        
        # Agreement strength: for non-zero positions, how unified is the sign?
        # Range [0, 1] where 1 = perfect agreement, 0 = perfect split
        max_count = torch.maximum(pos_count, neg_count)
        total_count = pos_count + neg_count
        agreement = torch.where(
            total_count > 0,
            max_count / total_count,
            torch.ones_like(max_count)
        )
        agreement_strength = agreement[non_zero_positions].mean().item()
        
        # Majority margin: (majority - minority) / total
        min_count = torch.minimum(pos_count, neg_count)
        majority_margin = torch.where(
            total_count > 0,
            (max_count - min_count) / total_count,
            torch.ones_like(max_count)
        )
        majority_margin = majority_margin[non_zero_positions].mean().item()
        
        return {
            "conflict_rate": conflict_rate,
            "agreement_strength": agreement_strength,
            "majority_margin": majority_margin,
            "num_conflicts": conflicts.sum().item(),
            "num_positions": non_zero_positions.sum().item(),
        }

    def compute_jsd(self, task_vectors: Tensor) -> float:
        """
        Compute average pairwise Jensen-Shannon Divergence between task vectors.
        
        JSD measures similarity between probability distributions.
        Range: [0, 1] where 0 = identical, 1 = completely different.
        
        Args:
            task_vectors: [K, D] tensor
            
        Returns:
            Average pairwise JSD
        """
        K = task_vectors.shape[0]
        if K < 2:
            return 0.0
        
        # Convert to probability distributions (use abs values, normalize)
        abs_vectors = torch.abs(task_vectors)
        
        # Add small constant to avoid zeros
        abs_vectors = abs_vectors + 1e-10
        
        # Normalize to probability distributions
        prob_vectors = abs_vectors / abs_vectors.sum(dim=1, keepdim=True)
        
        # Convert to numpy for scipy
        prob_vectors_np = prob_vectors.cpu().numpy()
        
        # Compute pairwise JSD
        jsd_values = []
        for i in range(K):
            for j in range(i + 1, K):
                jsd = jensenshannon(prob_vectors_np[i], prob_vectors_np[j])
                jsd_values.append(jsd)
        
        return float(np.mean(jsd_values)) if jsd_values else 0.0

    def compute_magnitude_stats(self, task_vectors: Tensor) -> Dict[str, float]:
        """
        Compute magnitude statistics for task vectors.
        
        Args:
            task_vectors: [K, D] tensor
            
        Returns:
            Dict with magnitude metrics
        """
        norms = torch.norm(task_vectors, dim=1)
        abs_mean = torch.abs(task_vectors).mean()
        sparsity = (task_vectors == 0).float().mean()
        
        return {
            "mean_norm": norms.mean().item(),
            "std_norm": norms.std().item(),
            "mean_abs_value": abs_mean.item(),
            "sparsity": sparsity.item(),
        }

    def analyze_tensor(
        self,
        name: str,
        base_tensor: Tensor,
        task_tensors: List[Tensor],
        proj_ratio: float,
        seed_key: str,
        global_D: int = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single tensor across original space, subspace, and lifted space.
        
        Args:
            name: Parameter name
            base_tensor: Pretrained model tensor
            task_tensors: List of task-specific tensors
            proj_ratio: Projection ratio for this analysis
            seed_key: Seed for FastFood operator
            global_D: Global dimension if using global scope
            
        Returns:
            Dict with analysis results
        """
        # Compute task vectors (deltas)
        task_vectors_list = [t - base_tensor for t in task_tensors]
        
        # Stack into [K, D] format
        orig_shape = base_tensor.shape
        d_last = base_tensor.shape[-1]
        rows = base_tensor.numel() // d_last
        
        base_flat = base_tensor.view(rows, d_last).float()
        task_vecs_flat = torch.stack([
            tv.view(rows, d_last).float() for tv in task_vectors_list
        ], dim=0)  # [K, rows, d_last]
        
        # For simplicity, analyze the first row (or mean across rows)
        # Use mean to get representative behavior
        task_vecs_orig = task_vecs_flat.mean(dim=1)  # [K, d_last]
        
        # Original space metrics
        orig_conflicts = self.compute_sign_conflicts(task_vecs_orig)
        orig_jsd = self.compute_jsd(task_vecs_orig)
        orig_mag = self.compute_magnitude_stats(task_vecs_orig)
        
        # Create FastFood operators
        cur_D = global_D if global_D is not None else d_last
        proj_dim = max(1, int(cur_D * proj_ratio))
        
        fwd, lift = create_fastfood_ops(
            global_dim=cur_D,
            proj_dim=proj_dim,
            seed_key=seed_key,
            device=self.device,
            use_G=self.use_G,
        )
        
        # Project to subspace
        task_vecs_orig_dev = task_vecs_orig.to(self.device)
        
        # Handle global dimension padding
        if global_D is not None and d_last < cur_D:
            padded = torch.zeros((task_vecs_orig_dev.shape[0], cur_D), 
                                dtype=torch.float32, device=self.device)
            padded[:, :d_last] = task_vecs_orig_dev
            task_vecs_proj = torch.stack([fwd(padded[k]) for k in range(len(task_tensors))], dim=0)
        else:
            task_vecs_proj = torch.stack([fwd(task_vecs_orig_dev[k]) for k in range(len(task_tensors))], dim=0)
        
        # Subspace metrics
        subspace_conflicts = self.compute_sign_conflicts(task_vecs_proj)
        subspace_jsd = self.compute_jsd(task_vecs_proj)
        subspace_mag = self.compute_magnitude_stats(task_vecs_proj)
        
        # Lift back to original space
        task_vecs_lifted = torch.stack([lift(task_vecs_proj[k]) for k in range(len(task_tensors))], dim=0)
        task_vecs_lifted = task_vecs_lifted[:, :d_last].cpu()
        
        # Lifted space metrics
        lifted_conflicts = self.compute_sign_conflicts(task_vecs_lifted)
        lifted_jsd = self.compute_jsd(task_vecs_lifted)
        lifted_mag = self.compute_magnitude_stats(task_vecs_lifted)
        
        # Reconstruction error
        reconstruction_error = torch.norm(task_vecs_lifted - task_vecs_orig, dim=1).mean().item()
        relative_error = reconstruction_error / (torch.norm(task_vecs_orig, dim=1).mean().item() + 1e-10)
        
        return {
            "name": name,
            "shape": list(orig_shape),
            "num_params": base_tensor.numel(),
            "proj_ratio": proj_ratio,
            "proj_dim": proj_dim,
            "orig_dim": d_last,
            # Original space
            "orig_conflict_rate": orig_conflicts["conflict_rate"],
            "orig_agreement_strength": orig_conflicts["agreement_strength"],
            "orig_majority_margin": orig_conflicts["majority_margin"],
            "orig_jsd": orig_jsd,
            "orig_mean_norm": orig_mag["mean_norm"],
            "orig_sparsity": orig_mag["sparsity"],
            # Subspace
            "subspace_conflict_rate": subspace_conflicts["conflict_rate"],
            "subspace_agreement_strength": subspace_conflicts["agreement_strength"],
            "subspace_majority_margin": subspace_conflicts["majority_margin"],
            "subspace_jsd": subspace_jsd,
            "subspace_mean_norm": subspace_mag["mean_norm"],
            "subspace_sparsity": subspace_mag["sparsity"],
            # Lifted
            "lifted_conflict_rate": lifted_conflicts["conflict_rate"],
            "lifted_agreement_strength": lifted_conflicts["agreement_strength"],
            "lifted_majority_margin": lifted_conflicts["majority_margin"],
            "lifted_jsd": lifted_jsd,
            "lifted_mean_norm": lifted_mag["mean_norm"],
            "lifted_sparsity": lifted_mag["sparsity"],
            # Reconstruction
            "reconstruction_error": reconstruction_error,
            "relative_reconstruction_error": relative_error,
        }

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the subspace sign conflict analysis.
        
        Args:
            modelpool: Model pool containing pretrained and task-specific models
            
        Returns:
            Dict with comprehensive analysis results
        """
        log.info("=" * 80)
        log.info("Starting Subspace Sign Conflict Analysis")
        log.info("=" * 80)
        
        modelpool = to_modelpool(modelpool)
        
        # Load models
        with self.profile("loading models"):
            base_model = modelpool.load_model("_pretrained_")
            donor_names = list(modelpool.model_names)
            
            if len(donor_names) < 2:
                raise ValueError(f"Need at least 2 donor models, got {len(donor_names)}")
            
            log.info(f"Loaded base model and {len(donor_names)} task models")
            
            donors_sd: List[StateDictType] = [
                modelpool.load_model(n).state_dict(keep_vars=True)
                for n in donor_names
            ]
            base_sd: StateDictType = base_model.state_dict(keep_vars=True)
        
        # Filter eligible tensors
        keys_all = list(base_sd.keys())
        keys_float = [
            k for k in keys_all
            if (k in donors_sd[0])
            and torch.is_floating_point(base_sd[k])
            and base_sd[k].ndim >= 1
            and all((k in d) and torch.is_floating_point(d[k]) and d[k].shape == base_sd[k].shape for d in donors_sd)
        ]
        
        if self.trainable_only:
            # Filter out frozen parameters (heuristic: exclude embeddings and layer norms)
            keys_float = [k for k in keys_float if not any(x in k.lower() for x in ['embed', 'ln', 'layernorm'])]
        
        log.info(f"Analyzing {len(keys_float)} eligible tensors")
        
        # Sample if requested
        if self.sample_size and self.sample_size < len(keys_float):
            import random
            random.seed(42)
            keys_float = random.sample(keys_float, self.sample_size)
            log.info(f"Sampled {len(keys_float)} tensors for analysis")
        
        # Determine global dimension if needed
        global_D = None
        if self.subspace_scope == "global":
            global_D = sum(base_sd[k].numel() for k in keys_float)
            log.info(f"Global dimension: {global_D:,} parameters")
        
        # Analyze each tensor at each projection ratio
        all_results = []
        
        with self.profile("analyzing tensors"):
            for proj_ratio in self.proj_ratios:
                log.info(f"\nAnalyzing projection ratio: {proj_ratio}")
                
                for idx, name in enumerate(keys_float):
                    if (idx + 1) % 50 == 0:
                        log.info(f"  Progress: {idx + 1}/{len(keys_float)} tensors")
                    
                    # Get seed key based on scope
                    if self.subspace_scope == "global":
                        seed_key = "global"
                    elif self.subspace_scope == "layer":
                        seed_key = layer_key(name)
                    else:  # per_tensor
                        seed_key = name
                    
                    # Analyze this tensor
                    try:
                        result = self.analyze_tensor(
                            name=name,
                            base_tensor=base_sd[name],
                            task_tensors=[d[name] for d in donors_sd],
                            proj_ratio=proj_ratio,
                            seed_key=seed_key,
                            global_D=global_D,
                        )
                        all_results.append(result)
                    except Exception as e:
                        log.warning(f"Failed to analyze {name}: {e}")
                        continue
        
        # Aggregate results
        log.info("\nAggregating results...")
        aggregated = self.aggregate_results(all_results)
        
        # Save results
        self.save_results(all_results, aggregated)
        
        # Create visualizations
        self.create_visualizations(all_results, aggregated)
        
        log.info("=" * 80)
        log.info("Analysis complete!")
        log.info(f"Results saved to: {self.output_path}")
        log.info("=" * 80)
        
        self.print_profile_summary()
        
        return aggregated

    def aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all tensors and projection ratios."""
        import pandas as pd
        
        df = pd.DataFrame(all_results)
        
        # Group by projection ratio
        grouped = df.groupby('proj_ratio')
        
        aggregated = {}
        for proj_ratio, group in grouped:
            aggregated[proj_ratio] = {
                # Conflict rates
                "orig_conflict_rate_mean": group['orig_conflict_rate'].mean(),
                "orig_conflict_rate_std": group['orig_conflict_rate'].std(),
                "subspace_conflict_rate_mean": group['subspace_conflict_rate'].mean(),
                "subspace_conflict_rate_std": group['subspace_conflict_rate'].std(),
                "lifted_conflict_rate_mean": group['lifted_conflict_rate'].mean(),
                "lifted_conflict_rate_std": group['lifted_conflict_rate'].std(),
                # Change in conflict rate
                "conflict_rate_increase": (
                    group['subspace_conflict_rate'].mean() - group['orig_conflict_rate'].mean()
                ),
                # JSD
                "orig_jsd_mean": group['orig_jsd'].mean(),
                "subspace_jsd_mean": group['subspace_jsd'].mean(),
                "lifted_jsd_mean": group['lifted_jsd'].mean(),
                "jsd_increase": group['subspace_jsd'].mean() - group['orig_jsd'].mean(),
                # Agreement strength
                "orig_agreement_mean": group['orig_agreement_strength'].mean(),
                "subspace_agreement_mean": group['subspace_agreement_strength'].mean(),
                "lifted_agreement_mean": group['lifted_agreement_strength'].mean(),
                # Reconstruction error
                "reconstruction_error_mean": group['reconstruction_error'].mean(),
                "relative_reconstruction_error_mean": group['relative_reconstruction_error'].mean(),
                # Sample size
                "num_tensors": len(group),
            }
        
        return aggregated

    def save_results(self, all_results: List[Dict[str, Any]], aggregated: Dict[str, Any]):
        """Save results to CSV and JSON files."""
        import pandas as pd
        import json
        
        # Save detailed results
        df = pd.DataFrame(all_results)
        csv_path = self.output_path / "sign_conflict_detailed_results.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Saved detailed results to {csv_path}")
        
        # Save aggregated results
        json_path = self.output_path / "sign_conflict_aggregated_results.json"
        with open(json_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        log.info(f"Saved aggregated results to {json_path}")
        
        # Save human-readable summary
        summary_path = self.output_path / "sign_conflict_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SUBSPACE SIGN CONFLICT ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for proj_ratio in sorted(aggregated.keys()):
                stats = aggregated[proj_ratio]
                f.write(f"\nProjection Ratio: {proj_ratio}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Conflict Rate:\n")
                f.write(f"    Original:  {stats['orig_conflict_rate_mean']:.4f} ± {stats['orig_conflict_rate_std']:.4f}\n")
                f.write(f"    Subspace:  {stats['subspace_conflict_rate_mean']:.4f} ± {stats['subspace_conflict_rate_std']:.4f}\n")
                f.write(f"    Lifted:    {stats['lifted_conflict_rate_mean']:.4f} ± {stats['lifted_conflict_rate_std']:.4f}\n")
                f.write(f"    Increase:  {stats['conflict_rate_increase']:.4f} ({stats['conflict_rate_increase']/max(stats['orig_conflict_rate_mean'], 0.01)*100:.1f}%)\n")
                f.write(f"\n  Jensen-Shannon Divergence:\n")
                f.write(f"    Original:  {stats['orig_jsd_mean']:.4f}\n")
                f.write(f"    Subspace:  {stats['subspace_jsd_mean']:.4f}\n")
                f.write(f"    Increase:  {stats['jsd_increase']:.4f}\n")
                f.write(f"\n  Agreement Strength:\n")
                f.write(f"    Original:  {stats['orig_agreement_mean']:.4f}\n")
                f.write(f"    Subspace:  {stats['subspace_agreement_mean']:.4f}\n")
                f.write(f"\n  Reconstruction Error:\n")
                f.write(f"    Absolute:  {stats['reconstruction_error_mean']:.6f}\n")
                f.write(f"    Relative:  {stats['relative_reconstruction_error_mean']:.4f}\n")
                f.write(f"\n  Tensors Analyzed: {stats['num_tensors']}\n")
        
        log.info(f"Saved summary to {summary_path}")

    def create_visualizations(self, all_results: List[Dict[str, Any]], aggregated: Dict[str, Any]):
        """Create comprehensive visualization plots."""
        import pandas as pd
        
        df = pd.DataFrame(all_results)
        
        # Set up the plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Subspace Sign Conflict Analysis', fontsize=16, fontweight='bold')
        
        proj_ratios = sorted(df['proj_ratio'].unique())
        
        # 1. Conflict Rate Comparison
        ax = axes[0, 0]
        conflict_data = []
        for pr in proj_ratios:
            subset = df[df['proj_ratio'] == pr]
            conflict_data.append({
                'Projection Ratio': pr,
                'Original': subset['orig_conflict_rate'].mean(),
                'Subspace': subset['subspace_conflict_rate'].mean(),
                'Lifted': subset['lifted_conflict_rate'].mean(),
            })
        conflict_df = pd.DataFrame(conflict_data)
        x = np.arange(len(proj_ratios))
        width = 0.25
        ax.bar(x - width, conflict_df['Original'], width, label='Original', alpha=0.8)
        ax.bar(x, conflict_df['Subspace'], width, label='Subspace', alpha=0.8)
        ax.bar(x + width, conflict_df['Lifted'], width, label='Lifted', alpha=0.8)
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Conflict Rate')
        ax.set_title('Sign Conflict Rate by Space')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{pr:.2f}' for pr in proj_ratios])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Conflict Rate Increase
        ax = axes[0, 1]
        increase_data = [aggregated[pr]['conflict_rate_increase'] for pr in proj_ratios]
        ax.plot(proj_ratios, increase_data, marker='o', linewidth=2, markersize=8)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Conflict Rate Increase')
        ax.set_title('Subspace Conflict Rate Increase vs Original')
        ax.grid(True, alpha=0.3)
        
        # 3. JSD Comparison
        ax = axes[0, 2]
        jsd_data = []
        for pr in proj_ratios:
            subset = df[df['proj_ratio'] == pr]
            jsd_data.append({
                'Projection Ratio': pr,
                'Original': subset['orig_jsd'].mean(),
                'Subspace': subset['subspace_jsd'].mean(),
                'Lifted': subset['lifted_jsd'].mean(),
            })
        jsd_df = pd.DataFrame(jsd_data)
        ax.plot(proj_ratios, jsd_df['Original'], marker='o', label='Original', linewidth=2)
        ax.plot(proj_ratios, jsd_df['Subspace'], marker='s', label='Subspace', linewidth=2)
        ax.plot(proj_ratios, jsd_df['Lifted'], marker='^', label='Lifted', linewidth=2)
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Jensen-Shannon Divergence')
        ax.set_title('Distribution Divergence (JSD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Agreement Strength
        ax = axes[1, 0]
        agreement_data = []
        for pr in proj_ratios:
            subset = df[df['proj_ratio'] == pr]
            agreement_data.append({
                'Projection Ratio': pr,
                'Original': subset['orig_agreement_strength'].mean(),
                'Subspace': subset['subspace_agreement_strength'].mean(),
                'Lifted': subset['lifted_agreement_strength'].mean(),
            })
        agreement_df = pd.DataFrame(agreement_data)
        ax.plot(proj_ratios, agreement_df['Original'], marker='o', label='Original', linewidth=2)
        ax.plot(proj_ratios, agreement_df['Subspace'], marker='s', label='Subspace', linewidth=2)
        ax.plot(proj_ratios, agreement_df['Lifted'], marker='^', label='Lifted', linewidth=2)
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Agreement Strength')
        ax.set_title('Sign Agreement Strength (1=perfect, 0=split)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Reconstruction Error
        ax = axes[1, 1]
        recon_error = [aggregated[pr]['relative_reconstruction_error_mean'] for pr in proj_ratios]
        ax.plot(proj_ratios, recon_error, marker='o', color='purple', linewidth=2, markersize=8)
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Relative Reconstruction Error')
        ax.set_title('Subspace Reconstruction Error')
        ax.grid(True, alpha=0.3)
        
        # 6. Conflict Rate vs JSD (scatter)
        ax = axes[1, 2]
        for pr in proj_ratios:
            subset = df[df['proj_ratio'] == pr]
            ax.scatter(subset['subspace_conflict_rate'], subset['subspace_jsd'], 
                      alpha=0.5, label=f'PR={pr:.2f}', s=20)
        ax.set_xlabel('Subspace Conflict Rate')
        ax.set_ylabel('Subspace JSD')
        ax.set_title('Conflict Rate vs Distribution Divergence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 7. Magnitude Preservation
        ax = axes[2, 0]
        for pr in proj_ratios:
            subset = df[df['proj_ratio'] == pr]
            ratio = subset['subspace_mean_norm'] / (subset['orig_mean_norm'] + 1e-10)
            ax.scatter([pr] * len(ratio), ratio, alpha=0.3, s=10)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect preservation')
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Subspace Norm / Original Norm')
        ax.set_title('Magnitude Preservation in Subspace')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. Sparsity Changes
        ax = axes[2, 1]
        sparsity_data = []
        for pr in proj_ratios:
            subset = df[df['proj_ratio'] == pr]
            sparsity_data.append({
                'Projection Ratio': pr,
                'Original': subset['orig_sparsity'].mean(),
                'Subspace': subset['subspace_sparsity'].mean(),
            })
        sparsity_df = pd.DataFrame(sparsity_data)
        ax.plot(proj_ratios, sparsity_df['Original'], marker='o', label='Original', linewidth=2)
        ax.plot(proj_ratios, sparsity_df['Subspace'], marker='s', label='Subspace', linewidth=2)
        ax.set_xlabel('Projection Ratio')
        ax.set_ylabel('Sparsity (fraction of zeros)')
        ax.set_title('Sparsity in Original vs Subspace')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 9. Summary heatmap
        ax = axes[2, 2]
        heatmap_data = []
        for pr in proj_ratios:
            stats = aggregated[pr]
            heatmap_data.append([
                stats['orig_conflict_rate_mean'],
                stats['subspace_conflict_rate_mean'],
                stats['conflict_rate_increase'],
                stats['subspace_jsd_mean'],
                stats['relative_reconstruction_error_mean'],
            ])
        
        im = ax.imshow(np.array(heatmap_data).T, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(np.arange(len(proj_ratios)))
        ax.set_xticklabels([f'{pr:.2f}' for pr in proj_ratios], fontsize=8)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(['Orig Conflict', 'Sub Conflict', 'Conflict Δ', 'JSD', 'Recon Err'], fontsize=8)
        ax.set_xlabel('Projection Ratio')
        ax.set_title('Metrics Heatmap')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_path / "sign_conflict_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        log.info(f"Saved visualization to {fig_path}")
        
        plt.close()
