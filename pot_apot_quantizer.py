"""
Power-of-Two (POT) and Additive Power-of-Two (APOT) quantization implementations.

Based on papers:
- POT-PTQ: "Power-of-Two Quantization for Low-Precision Neural Networks"
- APOT: "Additive Power-of-Two Quantization"
- POT-PTQ: "PoTPTQ: A Two-step Power-of-Two Post-training for LLMs" 

Key ideas:
- POT: Restrict scales to powers of two (2^k) for efficient shift operations
- APOT: Represent weights as sums of power-of-two terms for more flexibility
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import itertools


# ==============================================================================
# POWER OF TWO QUANTIZATION
# ==============================================================================

@torch.no_grad()
def pot_quantize_tensor(
    w: torch.Tensor,
    n_bit: int = 4,
    q_group_size: int = -1
) -> torch.Tensor:
    """
    Power-of-Two quantization of weight tensor.
    
    Represents weights as: w_q = scale * sign(w) * 2^E
    
    Where:
    - scale is a power of two
    - E is the exponent (clamped to valid range)
    - sign(w) preserves the sign
    
    Args:
        w: Weight tensor to quantize
        n_bit: Number of bits for exponent
        q_group_size: Group size for per-group quantization
        
    Returns:
        Quantized weight tensor (pseudo-quantized, same dtype as input)
    """
    org_w_shape = w.shape
    org_dtype = w.dtype
    
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    
    assert w.dim() == 2
    
    # number of magnitude levels (sign handled separately)
    n_levels = 2 ** (n_bit - 1)
    E_max_idx = n_levels - 1
    
    # Step 1: Initialize base scale (power of two)
    max_val = w.abs().amax(dim=1, keepdim=True)
    # guard against all-zeros rows to avoid -inf from log2
    max_val_safe = torch.clamp(max_val, min=1e-12)
    # anchor exponent range to data: e_min = floor(log2(max_val)) - (n_levels - 1)
    e_max = torch.floor(torch.log2(max_val_safe))
    e_min = e_max - E_max_idx
    # base scale is a power of two; avoid underflow to zero in finite-precision
    base_two = torch.tensor(2.0, dtype=w.dtype, device=w.device)
    s_0 = torch.pow(base_two, e_min.to(w.dtype))
    s_0 = torch.clamp(s_0, min=torch.finfo(w.dtype).tiny)
    
    # Step 2: Grid search for optimal scale multiplier
    # Try scales: s_0 * b where b in [0.01, 2.01]
    B = torch.arange(0.01, 2.01, 0.01, device=w.device)
    best_error = torch.full((w.size(0), 1), float('inf'), device=w.device)
    best_scale = s_0.clone()
    
    for b in B:
        # Candidate scale: s_0 * b (still power-of-two based)
        s_b = s_0 * b
        s_b = torch.clamp(s_b, min=torch.finfo(w.dtype).tiny)
        
        # Compute exponent: E = round(log2(|w| / s_b))
        # Clamp to valid range [0, E_max_idx]
        with torch.no_grad():
            ratio = torch.clamp(w.abs() / s_b, min=1e-10)
            E = torch.clamp(torch.round(torch.log2(ratio)), 0, E_max_idx)
        
        # Reconstruct: w_q = s_b * sign(w) * 2^E
        w_q = s_b * torch.sign(w) * torch.pow(2.0, E)
        
        # Compute quantization error (MSE)
        error = ((w - w_q) ** 2).sum(dim=1, keepdim=True)
        
        # Update best scale
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b, best_scale)
    
    # Step 3: Final quantization with best scale
    with torch.no_grad():
        best_scale = torch.clamp(best_scale, min=torch.finfo(w.dtype).tiny)
        ratio = torch.clamp(w.abs() / best_scale, min=1e-10)
        E = torch.clamp(torch.round(torch.log2(ratio)), 0, E_max_idx)
    
    w_quantized = best_scale * torch.sign(w) * torch.pow(2.0, E)
    
    assert torch.isnan(w_quantized).sum() == 0
    w_quantized = w_quantized.reshape(org_w_shape)
    
    # preserve original dtype
    w_quantized = w_quantized.to(org_dtype)
    
    return w_quantized


@torch.no_grad()
def pot_quantize_model_weight(
    model: nn.Module,
    w_bit: int,
    q_group_size: int
) -> None:
    """Apply POT quantization to all linear layers in model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = pot_quantize_tensor(
                module.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size
            )


# ==============================================================================
# ADDITIVE POWER OF TWO QUANTIZATION
# ==============================================================================

def generate_apot_levels(
    n: int,
    k: int,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate all possible APOT quantization levels.
    
    APOT formula (from paper):
    Q_a(α, kn) = γ × {Σ p_i} where p_i ∈ {0, 1/2^i, 1/2^(i+n), 1/2^(i+2n), ..., 1/2^(i+(2^k-2)n)}
    
    Each term can take 2^k different values (including 0).
    Total number of unique levels ≤ (2^k)^n = 2^(kn) = 2^n_bit
    
    Args:
        n: Number of additive terms
        k: Base bit-width (bits per term)
        device: Device to create tensor on
        
    Returns:
        Tensor of all possible quantization levels (unsigned, sorted)
    """
    num_choices_per_term = 2 ** k
    
    # Generate possible values for each additive term
    all_p_values = []
    for i in range(n):
        # Each term i can be: 0, or 2^(-i), 2^(-(i+n)), 2^(-(i+2n)), ..., 2^(-(i+(2^k-2)n))
        p_i_values = [0.0]  # First choice is always 0
        
        for j in range(1, num_choices_per_term):
            exponent = i + (j - 1) * n
            p_i_values.append(2.0 ** (-exponent))
        
        all_p_values.append(p_i_values)
    
    # Generate all combinations (Cartesian product)
    all_combinations = list(itertools.product(*all_p_values))
    
    # Sum each combination to get final levels
    levels = torch.tensor(
        [sum(combo) for combo in all_combinations],
        dtype=torch.float32,
        device=device
    )
    
    # Remove duplicates and sort
    levels = torch.unique(levels)
    levels = torch.sort(levels)[0]
    
    return levels


@torch.no_grad()
def apot_quantize_tensor(
    w: torch.Tensor,
    n_bit: int = 4,
    q_group_size: int = -1,
    k: int = 2
) -> torch.Tensor:
    """
    Additive Power-of-Two quantization of weight tensor.
    
    Represents weights as: w_q = scale * sum(sign_i * 2^E_i) for i=1 to n
    
    Where n = n_bit // k, allowing more flexible quantization than POT.
    
    Args:
        w: Weight tensor to quantize
        n_bit: Total number of bits for quantization
        q_group_size: Group size for per-group quantization
        k: Base bit-width (bits per additive term)
        
    Returns:
        Quantized weight tensor
    """
    org_w_shape = w.shape
    org_dtype = w.dtype
    
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    
    assert w.dim() == 2
    
    # calculate number of additive terms
    n = max(1, n_bit // k)
    
    # generate APOT quantization levels
    levels = generate_apot_levels(n, k, device=w.device)
    
    # normalize levels: divide by max so max level = 1.0
    max_level = levels.max()
    if max_level > 0:
        levels = levels / max_level
    
    # create symmetric levels: {-levels, 0, +levels}
    positive_levels = levels[levels > 0]
    full_levels = torch.cat([
        -positive_levels.flip(0),
        torch.tensor([0.0], device=w.device),
        positive_levels
    ])
    
    # memory optimization: if too many levels, this will cause OOM
    num_levels = full_levels.numel()
    if num_levels > 32:
        # sample a subset of levels to reduce memory
        indices = torch.linspace(0, num_levels-1, 32, dtype=torch.long, device=w.device)
        full_levels = full_levels[indices]
    
    # Step 1: Initialize base scale
    max_val = w.abs().amax(dim=1, keepdim=True)
    s_0 = torch.clamp(max_val, min=1e-5)
    
    # calculate total elements for memory optimization decisions
    total_elements = w.size(0) * w.size(1)
    
    # step 2: grid search for optimal scale
    # use coarser grid for large tensors to save memory
    if total_elements > 500000:
        grid_step = 0.1  # coarser grid for large layers
    else:
        grid_step = 0.05
    B = torch.arange(0.01, 2.01, grid_step, device=w.device)
    best_error = torch.full((w.size(0), 1), float('inf'), device=w.device)
    best_scale = s_0.clone()
    
    # process in chunks to avoid memory issues
    # use smaller chunks for larger weight matrices to avoid OOM
    if total_elements > 1000000:  # very large layers (>1M elements)
        chunk_size = 16
    elif total_elements > 500000:  # large layers
        chunk_size = 32
    elif total_elements > 100000:
        chunk_size = 64
    elif total_elements > 50000:
        chunk_size = 128
    else:
        chunk_size = 256
    n_cols = w.size(1)
    
    for b_idx, b in enumerate(B):
        s_b = s_0 * b
        
        # normalize weights to unit scale
        w_normalized = w / s_b
        
        # process columns in chunks to avoid OOM (ye mene CUDA out of memory ka handling ke liye hai)
        w_q_normalized = torch.zeros_like(w_normalized)
        for col_start in range(0, n_cols, chunk_size):
            col_end = min(col_start + chunk_size, n_cols)
            w_chunk = w_normalized[:, col_start:col_end]
            
            # find closest quantization level for each weight in chunk
            # broadcasting: w_chunk is [rows, chunk_cols], full_levels is [num_levels]
            distances = torch.abs(
                w_chunk.unsqueeze(-1) - full_levels.view(1, 1, -1)
            )
            closest_idx = torch.argmin(distances, dim=-1)
            w_q_normalized[:, col_start:col_end] = full_levels[closest_idx]
            
            # clear intermediate tensors
            del distances, closest_idx, w_chunk
        
        # reconstruct with scale
        w_q = s_b * w_q_normalized
        
        # compute quantization error
        error = ((w - w_q) ** 2).sum(dim=1, keepdim=True)
        
        # update best scale
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b, best_scale)
        
        # clear intermediate tensors
        del w_q, error, w_normalized, w_q_normalized, mask
        
        # periodically clear cuda cache during grid search
        if torch.cuda.is_available() and (b_idx % 10 == 0):
            torch.cuda.empty_cache()
    
    # step 3: final quantization with best scale
    with torch.no_grad():
        w_normalized = w / best_scale
        
        # process in chunks to avoid memory issues
        w_q_normalized = torch.zeros_like(w_normalized)
        for col_start in range(0, n_cols, chunk_size):
            col_end = min(col_start + chunk_size, n_cols)
            w_chunk = w_normalized[:, col_start:col_end]
            
            distances = torch.abs(
                w_chunk.unsqueeze(-1) - full_levels.view(1, 1, -1)
            )
            closest_idx = torch.argmin(distances, dim=-1)
            w_q_normalized[:, col_start:col_end] = full_levels[closest_idx]
            
            # clear intermediate tensors
            del distances, closest_idx, w_chunk
        
        w_quantized = best_scale * w_q_normalized
        
        # clear intermediate tensors
        del w_normalized, w_q_normalized, best_scale, best_error
    
    assert torch.isnan(w_quantized).sum() == 0
    w_quantized = w_quantized.reshape(org_w_shape)
    
    # preserve original dtype
    w_quantized = w_quantized.to(org_dtype)
    
    return w_quantized


@torch.no_grad()
def apot_quantize_model_weight(
    model: nn.Module,
    w_bit: int,
    q_group_size: int,
    k: int = 2
) -> None:
    """Apply APOT quantization to all linear layers in model."""
    import gc
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = apot_quantize_tensor(
                module.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size,
                k=k
            )
            # clear cuda cache after each layer to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

