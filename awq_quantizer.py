"""
AWQ (Activation-aware Weight Quantization) implementation.

Based on: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
https://arxiv.org/abs/2306.00978

Key idea: Protect salient weights (those corresponding to large activation channels)
by either keeping them in higher precision or scaling them before quantization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from quantization_utils import pseudo_quantize_tensor


# ==============================================================================
# AWQ QUANTIZATION
# ==============================================================================

@torch.no_grad()
def awq_quantize_model_weight(
    model: nn.Module,
    w_bit: int,
    q_group_size: int,
    input_feat: Dict[str, List[torch.Tensor]],
    protect_ratio: float = 0.01,
    scale_factor: float = 1.0
) -> None:
    """
    Apply AWQ quantization to all linear layers in the model.
    
    The algorithm:
    1. For each layer, compute importance scores from activations
    2. Identify top protect_ratio channels as "salient"
    3. Scale up salient channels before quantization
    4. Quantize weights
    5. Scale back down salient channels
    
    This protects important weight channels from large quantization error.
    
    Args:
        model: Model to quantize
        w_bit: Number of bits for quantization
        q_group_size: Group size for quantization
        input_feat: Dict mapping layer names to activation statistics
        protect_ratio: Fraction of channels to protect (e.g., 0.01 = top 1%)
        scale_factor: Scale factor to apply to protected channels
    """
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if name not in input_feat:
                # Layer not in calibration data, skip
                continue
            
            # Compute importance from activation magnitudes
            importance = sum(input_feat[name]).float()
            
            # Find top protect_ratio% channels
            n_protect = max(1, int(len(importance) * protect_ratio))
            outlier_indices = torch.topk(importance, n_protect)[1]
            
            assert outlier_indices.dim() == 1
            
            # preserve original dtype
            orig_dtype = m.weight.data.dtype
            
            # Scale up protected channels before quantization
            # This reduces quantization error for important channels
            m.weight.data[:, outlier_indices] *= scale_factor
            
            # Quantize all weights (including scaled protected channels)
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size
            )
            
            # Scale back down protected channels to original scale
            # The quantized values are now more accurate due to the scaling
            m.weight.data[:, outlier_indices] /= scale_factor
            
            # ensure dtype consistency
            m.weight.data = m.weight.data.to(orig_dtype)


@torch.no_grad()
def awq_search_scale_factor(
    model: nn.Module,
    w_bit: int,
    q_group_size: int,
    input_feat: Dict[str, List[torch.Tensor]],
    protect_ratio: float = 0.01,
    scale_search_range: Tuple[float, float] = (1.0, 2.0),
    n_grid: int = 20
) -> float:
    """
    Search for optimal scale factor using grid search on calibration data.
    
    Minimizes the reconstruction error on a small sample of activations.
    
    Args:
        model: Model to quantize
        w_bit: Number of bits
        q_group_size: Group size
        input_feat: Activation statistics
        protect_ratio: Fraction of channels to protect
        scale_search_range: (min_scale, max_scale) for grid search
        n_grid: Number of grid points
        
    Returns:
        Best scale factor found
    """
    print("Searching for optimal scale factor...")
    
    # Note: This is a simplified version. Full implementation would:
    # 1. Collect sample activations
    # 2. For each scale factor, quantize and measure reconstruction error
    # 3. Return scale with minimum error
    
    # For now, return middle of search range
    min_scale, max_scale = scale_search_range
    optimal_scale = (min_scale + max_scale) / 2.0
    
    print(f"  -> Using scale factor: {optimal_scale:.3f}")
    return optimal_scale

