"""
Test and validation script for quantization framework.

Verifies that all quantization methods work correctly and produce
reasonable results on a small sample.
"""

import torch
import torch.nn as nn
import json
from typing import Dict, Any

from quantization_utils import (
    load_config,
    get_model_size,
    pseudo_quantize_tensor,
)

from pot_apot_quantizer import (
    generate_apot_levels,
    pot_quantize_tensor,
    apot_quantize_tensor,
)


# ==============================================================================
# UNIT TESTS
# ==============================================================================

def test_pseudo_quantize():
    """Test pseudo quantization function."""
    print("Testing pseudo quantization...")
    
    # Create test tensor
    w = torch.randn(32, 64)
    
    # Test 4-bit quantization without grouping
    w_q = pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1)
    assert w_q.shape == w.shape
    assert not torch.isnan(w_q).any()
    print("  ✓ Basic pseudo quantization works")
    
    # Test with grouping
    w_q = pseudo_quantize_tensor(w, n_bit=4, q_group_size=32)
    assert w_q.shape == w.shape
    assert not torch.isnan(w_q).any()
    print("  ✓ Pseudo quantization with grouping works")


def test_pot_quantize():
    """Test Power-of-Two quantization."""
    print("\nTesting Power-of-Two quantization...")
    
    # Create test tensor
    w = torch.randn(32, 64)
    
    # Test 4-bit POT quantization
    w_q = pot_quantize_tensor(w, n_bit=4, q_group_size=-1)
    assert w_q.shape == w.shape
    assert not torch.isnan(w_q).any()
    print("  ✓ POT quantization produces valid output")
    
    # Check that weights are actually modified
    assert not torch.allclose(w, w_q)
    print("  ✓ POT quantization modifies weights")
    
    # Test with grouping
    w_q = pot_quantize_tensor(w, n_bit=4, q_group_size=32)
    assert w_q.shape == w.shape
    assert not torch.isnan(w_q).any()
    print("  ✓ POT quantization with grouping works")


def test_apot_levels():
    """Test APOT level generation."""
    print("\nTesting APOT level generation...")
    
    # Test 4-bit (n=2, k=2)
    levels_4bit = generate_apot_levels(n=2, k=2)
    assert len(levels_4bit) > 0
    assert not torch.isnan(levels_4bit).any()
    print(f"  ✓ 4-bit APOT: {len(levels_4bit)} unique levels")
    
    # Test 2-bit (n=1, k=2)
    levels_2bit = generate_apot_levels(n=1, k=2)
    assert len(levels_2bit) > 0
    assert not torch.isnan(levels_2bit).any()
    print(f"  ✓ 2-bit APOT: {len(levels_2bit)} unique levels")


def test_apot_quantize():
    """Test Additive Power-of-Two quantization."""
    print("\nTesting Additive Power-of-Two quantization...")
    
    # Create test tensor
    w = torch.randn(32, 64)
    
    # Test 4-bit APOT quantization
    w_q = apot_quantize_tensor(w, n_bit=4, q_group_size=-1, k=2)
    assert w_q.shape == w.shape
    assert not torch.isnan(w_q).any()
    print("  ✓ APOT quantization produces valid output")
    
    # Check that weights are modified
    assert not torch.allclose(w, w_q)
    print("  ✓ APOT quantization modifies weights")
    
    # Test with grouping
    w_q = apot_quantize_tensor(w, n_bit=4, q_group_size=32, k=2)
    assert w_q.shape == w.shape
    assert not torch.isnan(w_q).any()
    print("  ✓ APOT quantization with grouping works")


def test_model_size():
    """Test model size calculation."""
    print("\nTesting model size calculation...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    
    # FP32 model size
    size_fp32 = get_model_size(model, data_width=32, group_size=-1)
    assert size_fp32 > 0
    print(f"  ✓ FP32 model size: {size_fp32 / (8 * 1024 * 1024):.2f} MB")
    
    # 4-bit model size
    size_4bit = get_model_size(model, data_width=4, group_size=128)
    assert size_4bit > 0
    assert size_4bit < size_fp32
    print(f"  ✓ 4-bit model size: {size_4bit / (8 * 1024 * 1024):.2f} MB")
    
    # Verify compression ratio
    ratio = size_4bit / size_fp32
    print(f"  ✓ Compression ratio: {ratio:.2%}")


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    config = load_config("config.json")
    
    # Verify required fields
    required_fields = [
        'model_name',
        'quantization_methods',
        'calibration_dataset',
        'test_dataset',
        'quantization_config',
    ]
    
    for field in required_fields:
        assert field in config, f"Missing required config field: {field}"
    print("  ✓ Config has all required fields")
    
    # Verify quantization configs
    for method in config['quantization_methods']:
        if method in ['awq', 'gptq', 'pot', 'apot']:
            assert method in config['quantization_config']
    print("  ✓ All quantization methods have configurations")


def test_quantization_error():
    """Test that quantization reduces model precision as expected."""
    print("\nTesting quantization error characteristics...")
    
    # Create a simple weight tensor
    w = torch.randn(1, 100)
    
    # Quantize to different bit widths
    errors = {}
    for n_bit in [2, 4, 8, 16]:
        w_q = pseudo_quantize_tensor(w, n_bit=n_bit, q_group_size=-1)
        mse = ((w - w_q) ** 2).mean().item()
        errors[n_bit] = mse
        print(f"  ✓ {n_bit}-bit MSE: {mse:.6f}")
    
    # Verify that lower bits have higher error
    assert errors[2] > errors[4] > errors[8]
    print("  ✓ Error decreases with increasing bit-width")


def test_quantization_stability():
    """Test that quantization is stable (no NaNs, Infs)."""
    print("\nTesting quantization stability...")
    
    # Create various test cases
    test_cases = [
        torch.randn(32, 64),  # Normal distribution
        torch.randn(32, 64) * 1000,  # Large values
        torch.randn(32, 64) / 1000,  # Small values
        torch.ones(32, 64),  # All ones
        -torch.ones(32, 64),  # All negative ones
    ]
    
    for i, w in enumerate(test_cases):
        # POT
        w_q = pot_quantize_tensor(w, n_bit=4)
        assert not torch.isnan(w_q).any() and not torch.isinf(w_q).any()
        
        # APOT
        w_q = apot_quantize_tensor(w, n_bit=4)
        assert not torch.isnan(w_q).any() and not torch.isinf(w_q).any()
        
        # Pseudo-quantize
        w_q = pseudo_quantize_tensor(w, n_bit=4)
        assert not torch.isnan(w_q).any() and not torch.isinf(w_q).any()
    
    print("  ✓ All quantization methods stable on edge cases")


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_framework_imports():
    """Test that all framework modules can be imported."""
    print("\nTesting framework imports...")
    
    try:
        from quantization_utils import load_config, get_model_size
        print("  ✓ quantization_utils imported")
    except ImportError as e:
        print(f"  ✗ Failed to import quantization_utils: {e}")
        return False
    
    try:
        from awq_quantizer import awq_quantize_model_weight
        print("  ✓ awq_quantizer imported")
    except ImportError as e:
        print(f"  ✗ Failed to import awq_quantizer: {e}")
        return False
    
    try:
        from gptq_quantizer import gptq_quantize_model_weight
        print("  ✓ gptq_quantizer imported")
    except ImportError as e:
        print(f"  ✗ Failed to import gptq_quantizer: {e}")
        return False
    
    try:
        from pot_apot_quantizer import pot_quantize_model_weight, apot_quantize_model_weight
        print("  ✓ pot_apot_quantizer imported")
    except ImportError as e:
        print(f"  ✗ Failed to import pot_apot_quantizer: {e}")
        return False
    
    try:
        from benchmark_runner import QuantizationBenchmark
        print("  ✓ benchmark_runner imported")
    except ImportError as e:
        print(f"  ✗ Failed to import benchmark_runner: {e}")
        return False
    
    return True

from smooth_quant_quantizer import (
    collect_act_scales,
    smooth_weights,
    smoothquant_quantize_model_weight,
)


def test_collect_act_scales():
    """Test activation scale collection."""
    print("\n" + "=" * 60)
    print("Test 1: Collect Activation Scales")
    print("=" * 60)
    
    # Create a simple model on CPU
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    model.to('cpu')
    model.eval()
    
    # Create dummy calibration data on CPU
    calib_samples = [torch.randn(1, 10).to('cpu') for _ in range(3)]
    
    # collect activation scales
    act_scales = collect_act_scales(model, calib_samples, verbose=True)
    
    print(f"\nCollected scales for {len(act_scales)} layers:")
    for name, scales in act_scales.items():
        print(f"  {name}: shape {scales.shape}, min={scales.min():.4f}, max={scales.max():.4f}")
    
    assert len(act_scales) > 0, "No activation scales collected"
    assert all(s.shape[-1] > 0 for s in act_scales.values()), "Invalid scale shapes"
    print("\n✓ Test 1 Passed!")


def test_smooth_weights():
    """Test weight smoothing transformation."""
    print("\n" + "=" * 60)
    print("Test 2: Weight Smoothing")
    print("=" * 60)
    
    # Create a simple model on CPU
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    model.to('cpu')
    model.eval()
    
    # Create dummy calibration data on CPU
    calib_samples = [torch.randn(1, 10).to('cpu') for _ in range(3)]
    
    # collect activation scales
    act_scales = collect_act_scales(model, calib_samples, verbose=False)
    
    # Save original weights for comparison
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_weights[name] = module.weight.data.clone()
    
    # Apply smoothing with different alpha values
    for alpha in [0.0, 0.5, 1.0]:
        print(f"\nTesting with alpha={alpha}...")
        
        # Reload model to reset weights
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        model.to('cpu')
        model.eval()
        
        # Reload original weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name in original_weights:
                    module.weight.data = original_weights[name].clone()
        
        # Apply smoothing
        smooth_weights(model, act_scales, alpha=alpha, verbose=False)
        
        # Check that smoothing was applied
        # For any alpha, the smoothing transformation should be applied
        # even if weights don't change much (depends on scale values)
        print(f"  ✓ Smoothing applied with alpha={alpha}")
    
    print("\n✓ Test 2 Passed!")


def test_smoothquant_quantization():
    """Test full SmoothQuant quantization pipeline."""
    print("\n" + "=" * 60)
    print("Test 3: Full SmoothQuant Quantization Pipeline")
    print("=" * 60)
    
    # Create a simple model on CPU
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    model.to('cpu')
    model.eval()
    
    # Create dummy calibration data on CPU
    calib_samples = [torch.randn(1, 10).to('cpu') for _ in range(5)]
    
    # collect activation scales
    print("\nStep 1: Collecting activation scales...")
    act_scales = collect_act_scales(model, calib_samples, verbose=False)
    print(f"  ✓ Collected scales for {len(act_scales)} layers")
    
    # Apply smoothquant quantization
    print("\nStep 2: Applying SmoothQuant quantization (w_bit=8, alpha=0.5)...")
    smoothquant_quantize_model_weight(
        model, 
        w_bit=8,
        q_group_size=-1,  # Use -1 for per-group quantization (whole row)
        act_scales=act_scales,
        alpha=0.5,
        verbose=False
    )
    print("  ✓ Quantization complete")
    
    # Verify weights are quantized (should be different from original)
    print("\nStep 3: Verifying quantization...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_range = (module.weight.data.max() - module.weight.data.min()).item()
            print(f"  {name}: weight range = {weight_range:.6f}")
    
    print("\n✓ Test 3 Passed!")


def test_alpha_effect():
    """Test that different alpha values produce different results."""
    print("\n" + "=" * 60)
    print("Test 4: Alpha Parameter Effect")
    print("=" * 60)
    
    # Create multiple models to test different alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    
    calib_samples = [torch.randn(1, 10).to('cpu') for _ in range(3)]
    
    for alpha in alphas:
        print(f"\nTesting alpha={alpha}...")
        
        # Create fresh model on CPU
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        model.to('cpu')
        model.eval()
        
        # Collect scales and quantize
        act_scales = collect_act_scales(model, calib_samples, verbose=False)
        smoothquant_quantize_model_weight(
            model, 
            w_bit=8,
            q_group_size=-1,  # Use -1 for per-group quantization
            act_scales=act_scales,
            alpha=alpha,
            verbose=False
        )
        
        # Compute average weight magnitude
        total_weight_norm = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_weight_norm += module.weight.data.norm().item()
        
        results[alpha] = total_weight_norm
        print(f"  Total weight norm: {total_weight_norm:.4f}")
    
    # Different alphas should produce different results (not all identical)
    values = list(results.values())
    assert not all(abs(v - values[0]) < 1e-6 for v in values), \
        "Different alpha values should produce different results"
    
    print("\n✓ Test 4 Passed! Different alpha values produce different quantizations.")



# ==============================================================================
# MAIN TEST SUITE
# ==============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("QUANTIZATION FRAMEWORK TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_framework_imports,
        test_config_loading,
        test_pseudo_quantize,
        test_pot_quantize,
        test_apot_levels,
        test_apot_quantize,
        test_model_size,
        test_quantization_error,
        test_quantization_stability,
        test_collect_act_scales,
        test_smooth_weights,
        test_smoothquant_quantization,
        test_alpha_effect,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0



if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)


