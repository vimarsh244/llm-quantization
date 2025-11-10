"""
Common utilities for quantization benchmarking framework
Includes: config loading, dataset management, model utilities, and evaluation metrics
"""

import json
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import gc


# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


# ==============================================================================
# SIZE UNITS
# ==============================================================================

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: Optional[str] = None,
    use_fast_tokenizer: bool = False
) -> Tuple[nn.Module, Any]:
    """
    Load model and tokenizer from Hugging Face.
    
    Args:
        model_name: Model identifier (e.g., 'facebook/opt-1.3b')
        device_map: Device mapping strategy
        torch_dtype: Data type (e.g., 'float16', 'float32', or None)
        use_fast_tokenizer: Whether to use fast tokenizer
        
    Returns:
        Tuple of (model, tokenizer)
    """
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        None: None
    }
    
    dtype_arg = dtype_map.get(torch_dtype, None)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast_tokenizer
    )
    
    model_kwargs = {'device_map': device_map}
    if dtype_arg is not None:
        model_kwargs['dtype'] = dtype_arg
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    return model, tokenizer


def unload_model(model: Optional[nn.Module] = None) -> None:
    """Clean up model from GPU memory."""
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


# ==============================================================================
# DATASET MANAGEMENT
# ==============================================================================

def get_calibration_dataset(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    n_samples: int = 256,
    block_size: int = 512
) -> List[Tensor]:
    """
    Load and prepare calibration dataset.
    
    Args:
        tokenizer: Model tokenizer
        dataset_name: Dataset identifier (e.g., 'wikitext')
        dataset_config: Dataset config (e.g., 'wikitext-2-raw-v1'), or None if no config
        split: Dataset split (e.g., 'validation', 'train')
        n_samples: Number of samples to use
        block_size: Sequence block size
        
    Returns:
        List of tokenized tensors
    """
    config_str = dataset_config if dataset_config is not None else "default"
    print(f"Loading {dataset_name} dataset ({config_str}, {split} split)...")
    if dataset_config is None:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset = dataset.shuffle(seed=42)
    
    samples = []
    n_run = 0
    
    for data in dataset:
        line = data["text"]
        line = line.strip()
        if not line:
            continue
            
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
            
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
            
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    
    # Concatenate and split by block size
    if not samples:
        raise ValueError("No valid samples found in dataset")
        
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f"  -> Split into {n_split} blocks of size {block_size}")
    
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]


def get_test_dataset(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    split: str,
    n_samples: int = 40,
    block_size: int = 2048
) -> Tensor:
    """
    Load and prepare test dataset for perplexity evaluation.
    
    Args:
        tokenizer: Model tokenizer
        dataset_name: Dataset identifier
        dataset_config: Dataset config
        split: Dataset split
        n_samples: Number of samples to evaluate
        block_size: Sequence block size
        
    Returns:
        Concatenated tensor of test samples
    """
    print(f"Loading {dataset_name} test dataset ({dataset_config}, {split} split)...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # Concatenate all texts
    text_data = "\n\n".join(dataset['text'])
    testenc = tokenizer(text_data, return_tensors='pt')
    
    print(f"  -> Test dataset shape: {testenc.input_ids.shape}")
    return testenc.input_ids


# ==============================================================================
# ACTIVATION COLLECTION
# ==============================================================================

def get_calib_feat(
    model: nn.Module,
    tokenizer: Any,
    calib_samples: List[Tensor],
    verbose: bool = True
) -> Dict[str, List[Tensor]]:
    """
    Collect activation statistics for all linear layers.
    
    This is used by AWQ to identify important weight channels.
    
    Args:
        model: Model to analyze
        tokenizer: Model tokenizer
        calib_samples: Calibration data samples
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping layer names to lists of activation tensors
    """
    input_dict = {}
    
    def stat_input_max_hook(m: nn.Module, x: Tuple, y: Tensor, name: str):
        """Hook to capture input activations to linear layers."""
        if isinstance(x, tuple):
            x = x[0]
        # Average magnitude of activations across batch and sequence dims
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]
    
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    lambda m, x, y, n=name: stat_input_max_hook(m, x, y, n)
                )
            )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print("Collecting activation scales from calibration data...")
    
    pbar = tqdm.tqdm(calib_samples, disable=not verbose, desc="calibration")
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        with torch.no_grad():
            model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return input_dict


# ==============================================================================
# PERPLEXITY EVALUATION
# ==============================================================================

def evaluate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    test_dataset: Tensor,
    n_samples: int = 40,
    block_size: int = 2048,
    verbose: bool = True
) -> float:
    """
    Calculate perplexity of model on test dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Model tokenizer (unused but included for consistency)
        test_dataset: Test input IDs tensor
        n_samples: Number of blocks to evaluate
        block_size: Sequence length per block
        verbose: Whether to print progress
        
    Returns:
        Perplexity score (lower is better)
    """
    test_dataset = test_dataset.to(model.device)
    model = model.eval()
    
    nlls = []
    
    pbar = tqdm.tqdm(
        range(n_samples),
        disable=not verbose,
        desc="evaluating perplexity"
    )
    
    for i in pbar:
        batch = test_dataset[:, (i * block_size):((i + 1) * block_size)]
        batch = batch.to(model.device)
        
        with torch.no_grad():
            lm_logits = model(batch).logits
        
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_dataset[:, (i * block_size):((i + 1) * block_size)][:, 1:]
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        neg_log_likelihood = loss.float() * block_size
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * block_size))
    return ppl.item()


# ==============================================================================
# MODEL SIZE CALCULATION
# ==============================================================================

def get_model_size(
    model: nn.Module,
    data_width: int = 16,
    group_size: int = -1,
    use_zero_point: bool = True
) -> int:
    """
    Calculate model size in bits.
    
    Args:
        model: Model to measure
        data_width: Bits per weight
        group_size: Group size for quantization (adds overhead for scales/zeros)
        use_zero_point: Whether zero points are used (POT/APOT don't use zero points)
        
    Returns:
        Model size in bits
    """
    if group_size != -1:
        # Add overhead for scale per group (16 bits for FP16 scale)
        scale_overhead = 16 / group_size
        # Add zero-point overhead only if used (4 bits for zero point)
        zero_overhead = (4 / group_size) if use_zero_point else 0
        data_width += scale_overhead + zero_overhead
    
    num_elements = sum(p.numel() for p in model.parameters())
    return num_elements * data_width


# ==============================================================================
# QUANTIZATION UTILITIES
# ==============================================================================

def pseudo_quantize_tensor(
    w: Tensor,
    n_bit: int = 4,
    q_group_size: int = -1
) -> Tensor:
    """
    Uniform quantization and dequantization (simulated).
    
    Maps values to n_bit integer range, then dequantizes back to float.
    
    Args:
        w: Weight tensor to quantize
        n_bit: Number of bits for quantization
        q_group_size: Group size for per-group quantization
        
    Returns:
        Pseudo-quantized tensor (same shape as input)
    """
    org_w_shape = w.shape
    org_dtype = w.dtype
    
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    
    assert w.dim() == 2
    
    # Calculate min and max per group
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    
    # Calculate scale and zero-point
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0
    
    # Quantize: map to [0, 2^n_bit - 1]
    w_q = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    
    # Dequantize: map back to float
    w = (w_q - zeros) * scales
    
    assert torch.isnan(w).sum() == 0
    w = w.reshape(org_w_shape)
    
    # preserve original dtype
    w = w.to(org_dtype)
    
    return w


def get_linear_layers(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Get all linear layers from model."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))
    return layers

