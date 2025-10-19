#!/usr/bin/env python3
"""
Test script for Linear Weights Similarity Analysis methods.

This script demonstrates how to use both analysis methods:
1. LinearWeightsSimilarityOriginal: Analyzes similarities in original parameter space
2. LinearWeightsSimilarityIntrinsic: Analyzes similarities after Fastfood projection

Both methods compute:
- Cosine Similarity
- L2 Distance  
- Sign Conflicts
- Jaccard Similarity

But only for 2D linear weight tensors (excluding biases, layer norms, embeddings, etc.)
"""

import os
import sys
import torch
from pathlib import Path

# Add fusion_bench to path
fusion_bench_path = Path(__file__).parent.parent
sys.path.insert(0, str(fusion_bench_path))

from fusion_bench.method.analysis.linear_weights_similarity_original import LinearWeightsSimilarityOriginal
from fusion_bench.method.analysis.linear_weights_similarity_intrinsic import LinearWeightsSimilarityIntrinsic
from fusion_bench.modelpool import DictModelPool


def create_dummy_model_pool():
    """
    Create a dummy model pool for testing.
    
    Returns a model pool with:
    - 1 pretrained model
    - 3 fine-tuned models (with different task vectors)
    """
    import torch.nn as nn
    
    # Simple test model with 2D and 1D parameters
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=100, hidden_dim=50, output_dim=10):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x
    
    # Create pretrained model
    pretrained = SimpleModel()
    
    # Create fine-tuned models with different modifications
    models = {}
    
    # Model 1: Small positive shift
    model1 = SimpleModel()
    model1.load_state_dict(pretrained.state_dict())
    with torch.no_grad():
        model1.linear1.weight.add_(torch.randn_like(model1.linear1.weight) * 0.1)
        model1.linear2.weight.add_(torch.randn_like(model1.linear2.weight) * 0.1)
    models["task_a"] = model1
    
    # Model 2: Large positive shift in different direction
    model2 = SimpleModel()
    model2.load_state_dict(pretrained.state_dict())
    with torch.no_grad():
        model2.linear1.weight.add_(torch.randn_like(model2.linear1.weight) * 0.3)
        model2.linear2.weight.add_(torch.randn_like(model2.linear2.weight) * 0.3)
    models["task_b"] = model2
    
    # Model 3: Negative shift (conflicts with model 1 and 2)
    model3 = SimpleModel()
    model3.load_state_dict(pretrained.state_dict())
    with torch.no_grad():
        # Add negative shift to create sign conflicts
        model3.linear1.weight.add_(-torch.randn_like(model3.linear1.weight) * 0.2)
        model3.linear2.weight.add_(-torch.randn_like(model3.linear2.weight) * 0.2)
    models["task_c"] = model3
    
    # Create model pool
    modelpool = DictModelPool(
        models=models,
        pretrained_model=pretrained
    )
    
    return modelpool


def test_original_space_analysis():
    """Test Linear Weights Similarity Analysis in Original Space."""
    print("\n" + "="*80)
    print("TEST 1: Linear Weights Similarity Analysis (Original Space)")
    print("="*80)
    
    # Create dummy model pool
    modelpool = create_dummy_model_pool()
    
    # Create analyzer
    analyzer = LinearWeightsSimilarityOriginal(
        plot_heatmap=True,
        trainable_only=True,
        method_name="test_original",
        jaccard_threshold=0.01,
        output_path="./test_output_original",
        device="cpu"  # Use CPU for testing
    )
    
    # Run analysis
    result = analyzer.run(modelpool)
    
    print("\n✓ Original space analysis completed successfully!")
    print("  Check ./test_output_original/ for results:")
    print("    - linear_weights_cosine_similarity_test_original.csv")
    print("    - linear_weights_l2_distance_test_original.csv")
    print("    - linear_weights_sign_conflicts_test_original.csv")
    print("    - linear_weights_jaccard_similarity_test_original.csv")
    print("    - linear_weights_analysis_test_original.pdf")
    
    return result


def test_intrinsic_space_analysis():
    """Test Linear Weights Similarity Analysis in Intrinsic Dimension Space."""
    print("\n" + "="*80)
    print("TEST 2: Linear Weights Similarity Analysis (Intrinsic Dimension)")
    print("="*80)
    
    # Create dummy model pool
    modelpool = create_dummy_model_pool()
    
    # Create analyzer with Fastfood projection
    analyzer = LinearWeightsSimilarityIntrinsic(
        plot_heatmap=True,
        trainable_only=True,
        method_name="test_intrinsic_proj0.50",
        jaccard_threshold=0.01,
        proj_ratio=0.50,  # Project to 50% of original dimension
        use_G=False,
        output_path="./test_output_intrinsic",
        device="cpu"  # Use CPU for testing
    )
    
    # Run analysis
    result = analyzer.run(modelpool)
    
    print("\n✓ Intrinsic dimension analysis completed successfully!")
    print("  Check ./test_output_intrinsic/ for results:")
    print("    - linear_weights_intrinsic_cosine_similarity_test_intrinsic_proj0.50.csv")
    print("    - linear_weights_intrinsic_l2_distance_test_intrinsic_proj0.50.csv")
    print("    - linear_weights_intrinsic_sign_conflicts_test_intrinsic_proj0.50.csv")
    print("    - linear_weights_intrinsic_jaccard_similarity_test_intrinsic_proj0.50.csv")
    print("    - linear_weights_intrinsic_analysis_test_intrinsic_proj0.50.pdf")
    
    return result


def test_multiple_projection_ratios():
    """Test multiple projection ratios to compare dimensionality effects."""
    print("\n" + "="*80)
    print("TEST 3: Multiple Projection Ratios Comparison")
    print("="*80)
    
    # Create dummy model pool
    modelpool = create_dummy_model_pool()
    
    projection_ratios = [0.25, 0.50, 0.75, 0.90]
    
    for proj_ratio in projection_ratios:
        print(f"\n--- Testing projection ratio: {proj_ratio:.2%} ---")
        
        analyzer = LinearWeightsSimilarityIntrinsic(
            plot_heatmap=True,
            trainable_only=True,
            method_name=None,  # Auto-generate based on proj_ratio
            jaccard_threshold=0.01,
            proj_ratio=proj_ratio,
            use_G=False,
            output_path=f"./test_output_proj{proj_ratio}",
            device="cpu"
        )
        
        analyzer.run(modelpool)
        print(f"  ✓ Completed analysis for proj_ratio={proj_ratio:.2%}")
    
    print("\n✓ All projection ratios tested successfully!")
    print("  Compare results across different dimensionalities:")
    for proj_ratio in projection_ratios:
        print(f"    - ./test_output_proj{proj_ratio}/")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Linear Weights Similarity Analysis - Test Suite")
    print("="*80)
    print("\nThis test suite demonstrates both analysis methods:")
    print("  1. Original Space: Analyze task vectors in full parameter space")
    print("  2. Intrinsic Dimension: Analyze task vectors after Fastfood projection")
    print("\nBoth methods only consider 2D linear weight tensors.")
    
    try:
        # Test 1: Original space analysis
        test_original_space_analysis()
        
        # Test 2: Intrinsic dimension analysis
        test_intrinsic_space_analysis()
        
        # Test 3: Multiple projection ratios
        test_multiple_projection_ratios()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review the generated CSV files and PDF visualizations")
        print("  2. Compare metrics between original and intrinsic spaces")
        print("  3. Experiment with different projection ratios and use_G settings")
        print("  4. Apply to real model pools using Hydra configs")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
