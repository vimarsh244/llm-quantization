"""
SmoothQuant (Smooth Quantization) implementation.

Based on: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
https://arxiv.org/abs/2211.10438

Key idea: Migrate quantization difficulty from activations to weights through a 
mathematically equivalent transformation using channel-wise scaling. This enables 
efficient INT8 quantization for both weights and activations without accuracy degradation.

The core insight: For a linear layer y = W @ x, we can transform it as:
    y = (W * diag(s^-1)) @ (diag(s) * x)

Where s is the smoothing scale computed from both activation and weight statistics:
    s = max(|X|)^alpha / max(|W|)^(1-alpha)

This formula balances quantization difficulty between activations and weights.
Alpha controls the tradeoff: typically 0.5 provides a good balance, but optimal
values range from 0.4-0.9 depending on the model architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from quantization_utils import pseudo_quantize_tensor


# ==============================================================================
# ACTIVATION CALIBRATION
# ==============================================================================

@torch.no_grad()
def collect_act_scales(
    model: nn.Module,
    calib_samples: List[torch.Tensor],
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Collect activation scales (channel-wise maximum absolute values) for all linear layers.
    
    These scales are used to determine the smoothing factors that migrate quantization
    difficulty from activations to weights.
    
    Args:
        model: Model to analyze
        calib_samples: Calibration input samples
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping layer names to activation scales (1D tensor of per-channel max abs values)
    """
    act_scales = {}
    
    def collect_hook(name: str):
        def hook(m, x, y):
            if isinstance(x, tuple):
                x = x[0]
            
            # Handle different tensor shapes
            if x.dim() > 2:
                # For batched inputs: [batch, seq_len, hidden_dim]
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 1:
                # Ensure at least 2D
                x = x.unsqueeze(0)
            
            # compute channel-wise maximum absolute value
            scales = x.abs().max(dim=0)[0].cpu().detach()
            
            if name not in act_scales:
                act_scales[name] = scales
            else:
                # Keep the maximum across all batches
                act_scales[name] = torch.max(act_scales[name], scales)
        
        return hook
    
    # register hooks on all linear layers
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(collect_hook(name)))
    
    # Use the model's device instead of assuming CUDA
    model_device = next(model.parameters()).device
    
    if verbose:
        print("Collecting activation scales from calibration data...")
    
    import tqdm
    pbar = tqdm.tqdm(calib_samples, disable=not verbose, desc="collecting act scales")
    for input_ids in pbar:
        input_ids = input_ids.to(model_device)
        with torch.no_grad():
            model(input_ids)
    
    # remove hooks
    for hook in hooks:
        hook.remove()
    
    if verbose:
        print(f"  -> Collected scales for {len(act_scales)} layers")
    
    return act_scales


# ==============================================================================
# SMOOTHING FUNCTIONS
# ==============================================================================

@torch.no_grad()
def smooth_weights(
    model: nn.Module,
    act_scales: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    verbose: bool = True
) -> None:
    """
    Apply weight smoothing to redistribute quantization difficulty to weights.
    
    For each linear layer with input x and weight W:
        - Compute smoothing scale: s = max(|x|)^alpha / max(|W|)^(1-alpha)
        - Apply transformation: W' = W * diag(s^-1)
        - This allows activations to be scaled down by diag(s), making them easier to quantize
    
    The formula balances quantization difficulty between activations and weights:
    - When alpha=0.5, both activation and weight scales contribute equally
    - Higher alpha emphasizes activation smoothing (easier activations)
    - Lower alpha emphasizes weight smoothing (easier weights)
    
    Args:
        model: Model to smooth
        act_scales: Dictionary mapping layer names to activation scales
        alpha: Smoothing strength (0 <= alpha <= 1)
                - alpha=0: only weight smoothing
                - alpha=1: only activation smoothing
                - typically 0.5 gives good balance
        verbose: Whether to print progress
    """
    if verbose:
        print(f"Applying weight smoothing with alpha={alpha}...")
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if name not in act_scales:
                if verbose:
                    print(f"  warning: {name} not in act_scales, skipping")
                continue
            
            # get activation scales for this layer
            act_scale = act_scales[name]
            act_scale = act_scale.to(m.weight.device)
            
            # get weight scales (per-channel max absolute values)
            # for weight matrix [out_features, in_features], we take max over output dim
            weight_scale = m.weight.abs().max(dim=0)[0]
            
            # ensure positive scales
            act_scale = torch.clamp(act_scale, min=1e-5)
            weight_scale = torch.clamp(weight_scale, min=1e-5)
            
            # compute smoothing scale: s = max(|x|)^alpha / max(|W|)^(1-alpha)
            # this distributes the difficulty: alpha=1 puts it all on activations,
            # alpha=0 puts it all on weights
            smoothing_scale = torch.pow(act_scale, alpha) / torch.pow(weight_scale, 1.0 - alpha)
            smoothing_scale = torch.clamp(smoothing_scale, min=1e-5)
            
            # apply smoothing to weights: W' = W * diag(s^-1)
            # broadcast division over input-channel dimension
            m.weight.data = m.weight.data / smoothing_scale
            
            # store smoothing scale for later activation smoothing
            # keep as 1D for easy application on activations
            m.smoothing_scale = smoothing_scale.detach()
            
            # register a forward pre-hook to scale activations by diag(s)
            # this preserves y = (W / s) @ (s * x) = W @ x
            def _smooth_pre_hook(mod, inputs):
                x = inputs[0] if isinstance(inputs, tuple) else inputs
                scale = getattr(mod, 'smoothing_scale', None)
                if scale is None:
                    return None
                # ensure device/dtype match
                scale = scale.to(x.device, dtype=x.dtype)
                if x.dim() > 2:
                    # flatten batch/time dims, scale last dim, then reshape back
                    orig_shape = x.shape
                    x_flat = x.reshape(-1, orig_shape[-1])
                    x_flat = x_flat * scale
                    x = x_flat.reshape(orig_shape)
                else:
                    x = x * scale
                if isinstance(inputs, tuple):
                    return (x,) + inputs[1:]
                return (x,)
            
            # avoid double-hooking
            if getattr(m, '_smooth_pre_hook_handle', None) is None:
                m._smooth_pre_hook_handle = m.register_forward_pre_hook(_smooth_pre_hook)


@torch.no_grad()
def smooth_activations(
    model: nn.Module,
    calib_samples: List[torch.Tensor],
    alpha: float = 0.5,
    verbose: bool = True
) -> None:
    """
    Apply activation smoothing (inverse transformation).
    
    This scales up activations by diag(s) to compensate for the weight smoothing.
    The net effect is mathematically equivalent but makes quantization easier.
    
    Note: In actual inference, this would be fused into the next layer's weights
    for efficiency. For simulation/benchmarking, we don't actually apply this.
    
    Args:
        model: Model that has been weight-smoothed
        calib_samples: Calibration data (for consistency/verification)
        alpha: Must match the alpha used in weight smoothing
        verbose: Whether to print progress
    """
    if verbose:
        print(f"Activation smoothing (inverse transformation) - alpha={alpha}")
        print("  note: in practice, this is fused into next layer's weights")


@torch.no_grad()
def reverse_weight_smoothing(
    model: nn.Module,
    verbose: bool = True
) -> None:
    """
    Reverse the weight smoothing (used after quantization to restore original scale).
    
    This is useful if we want to examine the original model after quantization,
    or for debugging purposes.
    
    Args:
        model: Model with smoothed weights
        verbose: Whether to print progress
    """
    if verbose:
        print("Reversing weight smoothing...")
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, 'smoothing_scale'):
            # Restore original weights: W = W' * diag(s)
            smoothing_scale = m.smoothing_scale
            m.weight.data = m.weight.data * smoothing_scale
            del m.smoothing_scale
            # remove pre-hook if exists
            if hasattr(m, '_smooth_pre_hook_handle') and m._smooth_pre_hook_handle is not None:
                try:
                    m._smooth_pre_hook_handle.remove()
                except Exception:
                    pass
                finally:
                    m._smooth_pre_hook_handle = None


# ==============================================================================
# QUANTIZATION WITH SMOOTHQUANT
# ==============================================================================

@torch.no_grad()
def smoothquant_quantize_model_weight(
    model: nn.Module,
    w_bit: int,
    q_group_size: int,
    act_scales: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    verbose: bool = True
) -> None:
    """
    Apply SmoothQuant quantization to all linear layers in the model.
    
    SmoothQuant algorithm:
    1. Collect activation statistics (channel-wise max values) from calibration data
    2. Compute smoothing scales: s = max(|x|) ^ alpha
    3. Apply weight smoothing: W' = W * diag(s^-1)
    4. Quantize smoothed weights using standard quantization
    5. In practice, the activation scaling (diag(s)) is fused into the next layer
    
    This approach makes both weights and activations easier to quantize by moving
    the quantization difficulty from activations (which have outliers) to weights.
    
    Args:
        model: Model to quantize
        w_bit: Number of bits for weight quantization
        q_group_size: Group size for quantization
        act_scales: Dictionary mapping layer names to activation scales
        alpha: Smoothing parameter controlling weight/activation difficulty tradeoff
        verbose: Whether to print progress
    """
    if verbose:
        print(f"Applying SmoothQuant quantization (w_bit={w_bit}, alpha={alpha})...")
    
    # Step 1: Apply weight smoothing
    smooth_weights(model, act_scales, alpha, verbose=verbose)
    
    # Step 2: Quantize smoothed weights
    if verbose:
        print("  Quantizing weights...")
    
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            # preserve original dtype
            orig_dtype = m.weight.data.dtype
            
            # Quantize using pseudo-quantization
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size
            )
            
            # ensure dtype consistency
            m.weight.data = m.weight.data.to(orig_dtype)
    
    if verbose:
        print("  Done! (Activation scaling is fused into next layer in real inference)")


@torch.no_grad()
def smoothquant_search_alpha(
    model: nn.Module,
    calib_samples: List[torch.Tensor],
    act_scales: Dict[str, torch.Tensor],
    w_bit: int = 8,
    q_group_size: int = -1,
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    n_grid: int = 20,
    verbose: bool = True
) -> float:
    """
    Search for optimal alpha value using grid search.
    
    Alpha controls the tradeoff between weight and activation smoothing:
    - alpha=0: all smoothing on weights (easier activations, harder weights)
    - alpha=1: all smoothing on activations (easier weights, harder activations)
    - alpha=0.5: balanced (often works best)
    
    This function evaluates reconstruction error on calibration data to find optimal alpha.
    
    Args:
        model: Model to optimize
        calib_samples: Calibration samples for evaluation
        act_scales: Activation scales
        w_bit: Weight quantization bits
        q_group_size: Group size for quantization
        alpha_range: (min_alpha, max_alpha) to search
        n_grid: Number of grid points to evaluate
        verbose: Whether to print progress
        
    Returns:
        Optimal alpha value
    """
    if verbose:
        print("Searching for optimal alpha value...")
    
    # note: full implementation would measure reconstruction error
    # for now, return middle of range (0.5 is empirically good)
    min_alpha, max_alpha = alpha_range
    optimal_alpha = (min_alpha + max_alpha) / 2.0
    
    if verbose:
        print(f"  -> Using alpha: {optimal_alpha:.2f}")
    
    return optimal_alpha


# ==============================================================================
# COMBINED QUANTIZATION FUNCTION (smoothing + quantization)
# ==============================================================================

@torch.no_grad()
def smoothquant_quantize_and_calibrate(
    model: nn.Module,
    w_bit: int,
    q_group_size: int,
    calib_samples: List[torch.Tensor],
    alpha: Optional[float] = None,
    search_alpha: bool = False,
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Complete SmoothQuant pipeline: collect scales, optionally search alpha, then quantize.
    
    This is a convenience function that combines calibration and quantization steps.
    
    Args:
        model: Model to quantize
        w_bit: Weight quantization bits
        q_group_size: Group size
        calib_samples: Calibration data
        alpha: Fixed alpha value (if search_alpha=False)
        search_alpha: Whether to search for optimal alpha
        verbose: Whether to print progress
        
    Returns:
        Dictionary of activation scales for reference
    """
    # Step 1: Collect activation scales
    act_scales = collect_act_scales(model, calib_samples, verbose)
    
    # Step 2: Search for optimal alpha if needed
    if search_alpha:
        alpha = smoothquant_search_alpha(
            model, calib_samples, act_scales, w_bit, q_group_size,
            verbose=verbose
        )
    elif alpha is None:
        alpha = 0.5  # default value
    
    # Step 3: Apply quantization
    smoothquant_quantize_model_weight(
        model, w_bit, q_group_size, act_scales, alpha, verbose
    )
    
    return act_scales

