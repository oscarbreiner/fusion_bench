#!/usr/bin/env python3
"""
Comparative Analysis Script: Original vs Intrinsic Space

This script runs both analysis methods and creates a comparison report showing
how similarities change when projecting to intrinsic dimension.

Usage:
    python compare_original_vs_intrinsic.py --modelpool clip-vit-base-patch32_TA8 --output ./comparison_output
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add fusion_bench to path
fusion_bench_path = Path(__file__).parent.parent
sys.path.insert(0, str(fusion_bench_path))

from fusion_bench.method.analysis import (
    LinearWeightsSimilarityOriginal,
    LinearWeightsSimilarityIntrinsic
)


def run_comparative_analysis(modelpool, output_path, proj_ratios=[0.50, 0.75, 0.90]):
    """
    Run comparative analysis across original and multiple intrinsic dimensions.
    
    Args:
        modelpool: Model pool to analyze
        output_path: Directory to save results
        proj_ratios: List of projection ratios to test
    """
    os.makedirs(output_path, exist_ok=True)
    
    print("\n" + "="*80)
    print("Comparative Analysis: Original vs Intrinsic Space")
    print("="*80)
    
    # 1. Run original space analysis
    print("\n[1/N] Running Original Space Analysis...")
    original_analyzer = LinearWeightsSimilarityOriginal(
        plot_heatmap=True,
        trainable_only=True,
        method_name="original",
        output_path=os.path.join(output_path, "original"),
        device="cuda"
    )
    original_analyzer.run(modelpool)
    
    # Load original space results
    original_cos_sim = pd.read_csv(
        os.path.join(output_path, "original", "linear_weights_cosine_similarity_original.csv"),
        index_col=0
    )
    
    # 2. Run intrinsic space analysis for each projection ratio
    intrinsic_results = {}
    for i, proj_ratio in enumerate(proj_ratios, start=2):
        print(f"\n[{i}/{len(proj_ratios)+1}] Running Intrinsic Analysis (proj_ratio={proj_ratio:.2%})...")
        
        intrinsic_analyzer = LinearWeightsSimilarityIntrinsic(
            plot_heatmap=True,
            trainable_only=True,
            method_name=f"intrinsic_proj{proj_ratio}",
            proj_ratio=proj_ratio,
            use_G=False,
            output_path=os.path.join(output_path, f"intrinsic_proj{proj_ratio}"),
            device="cuda"
        )
        intrinsic_analyzer.run(modelpool)
        
        # Load intrinsic space results
        intrinsic_cos_sim = pd.read_csv(
            os.path.join(output_path, f"intrinsic_proj{proj_ratio}", 
                        f"linear_weights_intrinsic_cosine_similarity_intrinsic_proj{proj_ratio}.csv"),
            index_col=0
        )
        intrinsic_results[proj_ratio] = intrinsic_cos_sim
    
    # 3. Create comparison visualizations
    print("\n[Final] Creating comparison visualizations...")
    create_comparison_plots(original_cos_sim, intrinsic_results, output_path)
    
    # 4. Generate comparison report
    generate_comparison_report(original_cos_sim, intrinsic_results, output_path)
    
    print("\n" + "="*80)
    print("✓ Comparative Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print("\nGenerated files:")
    print("  - original/: Original space analysis results")
    for proj_ratio in proj_ratios:
        print(f"  - intrinsic_proj{proj_ratio}/: Intrinsic space analysis (proj={proj_ratio:.2%})")
    print("  - comparison_plots.pdf: Side-by-side comparison")
    print("  - comparison_report.txt: Numerical comparison summary")


def create_comparison_plots(original_df, intrinsic_dfs, output_path):
    """Create side-by-side comparison plots."""
    num_plots = 1 + len(intrinsic_dfs)
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot original space
    sns.heatmap(
        original_df, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=-1, vmax=1, ax=axes[0],
        cbar_kws={'label': 'Cosine Similarity'}
    )
    axes[0].set_title("Original Space\n(Full Dimension)", fontsize=12, fontweight='bold')
    
    # Plot intrinsic spaces
    for idx, (proj_ratio, intrinsic_df) in enumerate(intrinsic_dfs.items(), start=1):
        sns.heatmap(
            intrinsic_df, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=-1, vmax=1, ax=axes[idx],
            cbar_kws={'label': 'Cosine Similarity'}
        )
        axes[idx].set_title(
            f"Intrinsic Space\n(Projection={proj_ratio:.2%})",
            fontsize=12, fontweight='bold'
        )
    
    plt.tight_layout()
    output_file = os.path.join(output_path, "comparison_plots.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot to {output_file}")
    plt.close()


def generate_comparison_report(original_df, intrinsic_dfs, output_path):
    """Generate a text report comparing metrics."""
    report_path = os.path.join(output_path, "comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Comparative Analysis Report: Original vs Intrinsic Space\n")
        f.write("="*80 + "\n\n")
        
        # Extract off-diagonal elements (pairwise similarities)
        def get_off_diagonal(df):
            mask = ~np.eye(df.shape[0], dtype=bool)
            return df.values[mask]
        
        original_values = get_off_diagonal(original_df)
        
        f.write("COSINE SIMILARITY STATISTICS\n")
        f.write("-"*80 + "\n\n")
        
        # Original space stats
        f.write("Original Space (Full Dimension):\n")
        f.write(f"  Mean:   {original_values.mean():.4f}\n")
        f.write(f"  Std:    {original_values.std():.4f}\n")
        f.write(f"  Min:    {original_values.min():.4f}\n")
        f.write(f"  Max:    {original_values.max():.4f}\n")
        f.write(f"  Median: {np.median(original_values):.4f}\n\n")
        
        # Intrinsic space stats
        for proj_ratio, intrinsic_df in intrinsic_dfs.items():
            intrinsic_values = get_off_diagonal(intrinsic_df)
            
            f.write(f"Intrinsic Space (Projection={proj_ratio:.2%}):\n")
            f.write(f"  Mean:   {intrinsic_values.mean():.4f}\n")
            f.write(f"  Std:    {intrinsic_values.std():.4f}\n")
            f.write(f"  Min:    {intrinsic_values.min():.4f}\n")
            f.write(f"  Max:    {intrinsic_values.max():.4f}\n")
            f.write(f"  Median: {np.median(intrinsic_values):.4f}\n")
            
            # Compute difference from original
            mean_diff = intrinsic_values.mean() - original_values.mean()
            std_diff = intrinsic_values.std() - original_values.std()
            
            f.write(f"\n  Δ Mean vs Original: {mean_diff:+.4f}\n")
            f.write(f"  Δ Std vs Original:  {std_diff:+.4f}\n")
            f.write("\n")
        
        # Pairwise comparison
        f.write("\n" + "="*80 + "\n")
        f.write("PAIRWISE COMPARISON (Element-wise Differences)\n")
        f.write("="*80 + "\n\n")
        
        for proj_ratio, intrinsic_df in intrinsic_dfs.items():
            diff = intrinsic_df.values - original_df.values
            diff_off_diag = diff[~np.eye(diff.shape[0], dtype=bool)]
            
            f.write(f"Intrinsic (proj={proj_ratio:.2%}) - Original:\n")
            f.write(f"  Mean absolute difference: {np.abs(diff_off_diag).mean():.4f}\n")
            f.write(f"  Max absolute difference:  {np.abs(diff_off_diag).max():.4f}\n")
            f.write(f"  RMS difference:           {np.sqrt((diff_off_diag**2).mean()):.4f}\n")
            f.write("\n")
        
        # Interpretation guide
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("Higher mean cosine similarity in intrinsic space suggests:\n")
        f.write("  → Projection reduces noise and emphasizes principal components\n")
        f.write("  → Task vectors share common low-rank structure\n\n")
        f.write("Lower std in intrinsic space suggests:\n")
        f.write("  → More uniform similarity across task pairs\n")
        f.write("  → Less variability in task relationships\n\n")
        f.write("Small differences suggest:\n")
        f.write("  → Task diversity is well-captured by intrinsic dimension\n")
        f.write("  → Projection preserves similarity structure\n\n")
        f.write("Large differences suggest:\n")
        f.write("  → Projection is too aggressive (increase proj_ratio)\n")
        f.write("  → Important information may be lost\n")
    
    print(f"  Saved comparison report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comparative analysis between original and intrinsic spaces"
    )
    parser.add_argument(
        "--modelpool",
        type=str,
        required=True,
        help="Model pool identifier (e.g., 'clip-vit-base-patch32_TA8')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./comparison_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--proj-ratios",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 0.90],
        help="Projection ratios to test (default: 0.50 0.75 0.90)"
    )
    
    args = parser.parse_args()
    
    # Import modelpool loading utilities
    from fusion_bench.utils import instantiate_modelpool
    
    print(f"\nLoading model pool: {args.modelpool}")
    modelpool = instantiate_modelpool(args.modelpool)
    
    # Run comparative analysis
    run_comparative_analysis(
        modelpool=modelpool,
        output_path=args.output,
        proj_ratios=args.proj_ratios
    )


if __name__ == "__main__":
    main()
