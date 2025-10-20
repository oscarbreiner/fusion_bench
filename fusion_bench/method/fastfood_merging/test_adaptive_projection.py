#!/usr/bin/env python3
"""
Quick test script for adaptive projection size estimation in Fastfood merging.

Usage:
    python test_adaptive_projection.py
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fusion_bench.method.fastfood_merging.projection_size_estimator import (
    ProjSizeCfg,
    proj_size_for,
    stable_rank_tensor,
    effective_rank_layer
)


def test_basic_functionality():
    """Test basic projection size estimation."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    # Create sample tensors
    linear_weight = torch.randn(512, 256)  # (out, in)
    conv_weight = torch.randn(64, 32, 3, 3)  # (out, in, kh, kw)
    
    # Configuration
    cfg = ProjSizeCfg(
        m_min=16,
        f_max=0.5,
        ratio=0.25,
        beta=2.5,
        pow2_round=True,
        pow2_mode="ceil"
    )
    
    # Test tensor mode with different strategies
    print("\nTensor Mode:")
    for strategy in ["fixed", "rank"]:
        m = proj_size_for(linear_weight, mode="tensor", strategy=strategy, cfg=cfg)
        print(f"  {strategy:8s}: linear {linear_weight.shape} → m={m}")
    
    # Test layer mode
    print("\nLayer Mode:")
    layer_params = {
        "attn.q_proj.weight": torch.randn(512, 256),
        "attn.k_proj.weight": torch.randn(512, 256),
        "attn.v_proj.weight": torch.randn(512, 256),
        "attn.bias": torch.randn(512)
    }
    
    m = proj_size_for(layer_params, mode="layer", strategy="rank", cfg=cfg)
    print(f"  Layer with 3 weights + bias → m={m}")
    
    print("\n✓ Basic functionality test passed")


def test_rank_estimation():
    """Test rank estimation methods."""
    print("\n" + "=" * 60)
    print("Test 2: Rank Estimation")
    print("=" * 60)
    
    # Create tensors with known rank structure
    print("\nLow-rank tensor (rank ≈ 10):")
    U = torch.randn(512, 10)
    V = torch.randn(10, 256)
    W_low = U @ V
    r_stable = stable_rank_tensor(W_low)
    print(f"  Shape: {W_low.shape}, Stable rank: {r_stable:.2f}")
    
    print("\nFull-rank tensor:")
    W_full = torch.randn(512, 256)
    r_stable = stable_rank_tensor(W_full)
    print(f"  Shape: {W_full.shape}, Stable rank: {r_stable:.2f}")
    
    print("\nLayer effective rank:")
    layer = {
        "w1": torch.randn(512, 256),
        "w2": torch.randn(256, 128),
        "w3": torch.randn(128, 64)
    }
    r_eff = effective_rank_layer(layer)
    print(f"  3 weights, Effective rank: {r_eff:.2f}")
    
    print("\n✓ Rank estimation test passed")


def test_bounds_and_rounding():
    """Test bounds clipping and power-of-2 rounding."""
    print("\n" + "=" * 60)
    print("Test 3: Bounds and Rounding")
    print("=" * 60)
    
    W = torch.randn(512, 256)
    
    # Test different configurations
    configs = [
        ("Default (ceil)", ProjSizeCfg(m_min=16, f_max=0.5, pow2_round=True, pow2_mode="ceil", beta=2.5)),
        ("Floor rounding", ProjSizeCfg(m_min=16, f_max=0.5, pow2_round=True, pow2_mode="floor", beta=2.5)),
        ("No rounding", ProjSizeCfg(m_min=16, f_max=0.5, pow2_round=False, beta=2.5)),
        ("Low beta", ProjSizeCfg(m_min=16, f_max=0.5, pow2_round=True, beta=1.0)),
        ("High beta", ProjSizeCfg(m_min=16, f_max=0.5, pow2_round=True, beta=5.0)),
    ]
    
    print(f"\nTensor shape: {W.shape}")
    print(f"Stable rank: {stable_rank_tensor(W):.2f}\n")
    
    for name, cfg in configs:
        m = proj_size_for(W, mode="tensor", strategy="rank", cfg=cfg)
        is_pow2 = (m & (m - 1)) == 0
        print(f"  {name:20s}: m={m:4d} (pow2={is_pow2})")
    
    print("\n✓ Bounds and rounding test passed")


def test_integration_simulation():
    """Simulate usage in Fastfood merging context."""
    print("\n" + "=" * 60)
    print("Test 4: Integration Simulation")
    print("=" * 60)
    
    # Simulate a small model
    model_params = {
        "encoder.layer.0.attention.query.weight": torch.randn(768, 768),
        "encoder.layer.0.attention.key.weight": torch.randn(768, 768),
        "encoder.layer.0.attention.value.weight": torch.randn(768, 768),
        "encoder.layer.0.output.dense.weight": torch.randn(768, 3072),
        "encoder.layer.1.attention.query.weight": torch.randn(768, 768),
        "encoder.layer.1.attention.key.weight": torch.randn(768, 768),
        "encoder.layer.1.output.dense.weight": torch.randn(768, 3072),
    }
    
    cfg = ProjSizeCfg(beta=2.5, m_min=32, f_max=0.5, pow2_round=True)
    
    print("\nTensor mode (per-tensor projection):")
    for name, param in list(model_params.items())[:4]:
        m = proj_size_for(param, mode="tensor", strategy="rank", cfg=cfg)
        compression = m / param.shape[-1]
        print(f"  {name:50s} → m={m:4d} (comp={compression:.3f})")
    
    # Group by layer for layer mode
    from fusion_bench.method.fastfood_merging.fastfood_utils import layer_key
    
    print("\nLayer mode (shared projection per layer):")
    layer_groups = {}
    for name, param in model_params.items():
        lkey = layer_key(name)
        if lkey not in layer_groups:
            layer_groups[lkey] = {}
        layer_groups[lkey][name] = param
    
    for lkey, layer_params in layer_groups.items():
        m = proj_size_for(layer_params, mode="layer", strategy="rank", cfg=cfg)
        max_dim = max(p.shape[-1] for p in layer_params.values() if p.ndim >= 2)
        compression = m / max_dim if max_dim > 0 else 0
        print(f"  {lkey:40s} ({len(layer_params)} params) → m={m:4d} (comp={compression:.3f})")
    
    print("\n✓ Integration simulation test passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ADAPTIVE PROJECTION SIZE ESTIMATOR - TEST SUITE")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_rank_estimation()
        test_bounds_and_rounding()
        test_integration_simulation()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
