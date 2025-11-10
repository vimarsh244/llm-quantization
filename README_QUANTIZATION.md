# Quantization Benchmarking Framework

>> THIS README WAS GENERATED USING LLM.

A comprehensive framework for benchmarking different LLM quantization methods without external dependencies for core algorithms. This framework allows you to compare **AWQ**, **GPTQ**, **Power-of-Two (POT)**, and **Additive Power-of-Two (APOT)** quantization methods on any Hugging Face model.

## Features

- **Multiple Quantization Methods**: AWQ, GPTQ, POT, APOT
- **Flexible Configuration**: JSON-based configuration for easy experimentation
- **Multiple Datasets**: Support for any dataset from Hugging Face (wikitext, pile, etc.)
- **Comprehensive Evaluation**: Perplexity calculation, model size measurement, and comparative analysis
- **No External Dependencies**: Core quantization algorithms implemented from scratch
- **Modular Design**: Easy to extend with new quantization methods

## Architecture

```
├── config.json                  # Configuration file for benchmarks
├── quantization_utils.py        # Common utilities (config, datasets, evaluation)
├── awq_quantizer.py             # AWQ implementation
├── gptq_quantizer.py            # GPTQ implementation  
├── pot_apot_quantizer.py        # POT and APOT implementations
├── benchmark_runner.py          # Main benchmark orchestrator
├── test_quantization.py         # Test suite
└── README_QUANTIZATION.md       # This file
```

## Quantization Methods

### 1. AWQ (Activation-aware Weight Quantization)
- **Paper**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- **Key Idea**: Protects salient weights (those corresponding to large activation channels)
- **Method**: Identifies top 1% important channels and scales them to reduce quantization error
- **Advantages**: Good balance between compression and accuracy

### 2. GPTQ (Generative Pre-trained Transformer Quantization)
- **Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- **Key Idea**: Uses approximate second-order Hessian information
- **Method**: Quantizes weights iteratively using Hessian to order quantization
- **Advantages**: Highest quality quantization, uses calibration data effectively

### 3. POT (Power-of-Two Quantization)
- **Key Idea**: Restrict scales to powers of two (2^k)
- **Method**: Grid search for optimal power-of-two scale
- **Advantages**: Enables efficient shift-based operations in hardware

### 4. APOT (Additive Power-of-Two Quantization)
- **Key Idea**: Represent weights as sums of power-of-two terms
- **Method**: More flexible than POT, allows additive combinations
- **Advantages**: Better accuracy than POT while maintaining hardware efficiency

## Quick Start

### 1. Installation

```bash
pip install torch transformers datasets tqdm
```

### 2. Configure Your Benchmark

Edit `config.json`:

```json
{
  "model_name": "facebook/opt-1.3b",
  "quantization_methods": ["raw", "awq", "gptq", "pot", "apot"],
  "calibration_dataset": "wikitext",
  "calibration_dataset_config": "wikitext-2-raw-v1",
  "test_dataset": "wikitext",
  "test_dataset_config": "wikitext-2-raw-v1",
  "quantization_config": {
    "awq": {
      "w_bit": 3,
      "q_group_size": 128,
      "protect_ratio": 0.01,
      "scale_factor": 2.0
    },
    "gptq": {
      "w_bit": 4,
      "q_group_size": 128
    },
    "pot": {
      "w_bit": 4,
      "q_group_size": 128
    },
    "apot": {
      "w_bit": 4,
      "q_group_size": 128,
      "k": 2
    }
  }
}
```

### 3. Run Benchmarks

```bash
python benchmark_runner.py config.json
```

This will:
1. Load your model and tokenizer
2. Load calibration and test datasets
3. Benchmark raw model
4. Apply each quantization method
5. Evaluate perplexity
6. Save results to `benchmark_results.json`

### 4. View Results

Results are printed to console and saved to JSON:

```
================================================================================
BENCHMARK SUMMARY
================================================================================

Model: facebook/opt-1.3b
Calibration: wikitext
Test Dataset: wikitext
Timestamp: 2024-01-15 10:30:45

────────────────────────────────────────────────────────────────────────────────
Method       | Perplexity | Size (MB)  | Runtime (s)
────────────────────────────────────────────────────────────────────────────────
raw          |      15.87 |   4978.00 |       0.00
awq          |      16.45 |    620.00 |      45.23
gptq         |      16.12 |    620.00 |      120.45
pot          |      18.23 |    620.00 |      32.10
apot         |      17.34 |    620.00 |      38.50
────────────────────────────────────────────────────────────────────────────────

Improvements vs Raw Model:
  awq       : PPL  +3.66% | Size -87.54%
  gptq      : PPL  +1.58% | Size -87.54%
  pot       : PPL +15.02% | Size -87.54%
  apot      : PPL  +9.27% | Size -87.54%
```

## Configuration Reference

### Top-level Parameters

- **model_name** (str): Hugging Face model identifier
- **quantization_methods** (list): Methods to benchmark (raw, awq, gptq, pot, apot)
- **calibration_dataset** (str): Dataset for calibration (wikitext, pile, etc.)
- **calibration_dataset_config** (str): Dataset configuration
- **calibration_split** (str): Train/validation split for calibration
- **test_dataset** (str): Dataset for evaluation
- **test_dataset_config** (str): Test dataset configuration
- **test_split** (str): Test split
- **n_calibration_samples** (int): Number of calibration samples
- **n_test_samples** (int): Number of test blocks for perplexity
- **device_map** (str): Device mapping strategy (auto, cpu, cuda)
- **torch_dtype** (str): Model dtype (float32, float16, bfloat16)

### Quantization Method Configs

#### AWQ Configuration
- **w_bit** (int): Number of bits for weights (default: 3)
- **q_group_size** (int): Group size for quantization (default: 128)
- **protect_ratio** (float): Fraction of channels to protect (default: 0.01 = 1%)
- **scale_factor** (float): Scale factor for protected channels (default: 2.0)

#### GPTQ Configuration
- **w_bit** (int): Number of bits
- **q_group_size** (int): Group size
- **perp_damp** (float): Hessian damping factor (default: 0.01)
- **blocksize** (int): Block size for processing (default: 128)
- **nsamples** (int): Number of calibration samples to use (default: 128)
- **actorder** (bool): Whether to order by activation magnitude (default: false)

#### POT Configuration
- **w_bit** (int): Number of bits
- **q_group_size** (int): Group size
- **grid_search_range** (list): [min, max] for scale search
- **grid_step** (float): Grid search step size

#### APOT Configuration
- **w_bit** (int): Number of bits
- **q_group_size** (int): Group size
- **k** (int): Base bit-width per additive term (default: 2)

## Module Reference

### quantization_utils.py

Core utilities:
- `load_config(path)`: Load JSON configuration
- `load_model_and_tokenizer(name)`: Load HF model/tokenizer
- `get_calibration_dataset(...)`: Load and prepare calibration data
- `get_test_dataset(...)`: Load test data for evaluation
- `get_calib_feat(model, tokenizer, samples)`: Collect activation statistics
- `evaluate_perplexity(model, tokenizer, dataset)`: Calculate perplexity
- `get_model_size(model, data_width, group_size)`: Calculate model size

### awq_quantizer.py

AWQ implementation:
- `awq_quantize_model_weight(...)`: Main AWQ quantization function
- Identifies salient channels using activation magnitudes
- Protects them by scaling before/after quantization

### gptq_quantizer.py

GPTQ implementation:
- `gptq_quantize_model_weight(...)`: Main GPTQ quantization
- Uses approximate Hessian for weight ordering
- Per-layer quantization with error compensation

### pot_apot_quantizer.py

Power-of-Two quantization:
- `pot_quantize_tensor(...)`: POT quantization
- `apot_quantize_tensor(...)`: APOT quantization
- `generate_apot_levels(...)`: Generate APOT quantization levels

## Examples

### Example 1: Compare Methods on OPT-1.3B

```bash
# Edit config.json to use:
# - model: facebook/opt-1.3b
# - methods: [raw, awq, gptq, pot, apot]

python benchmark_runner.py config.json
```

### Example 2: Test on Smaller Model

```json
{
  "model_name": "facebook/opt-350m",
  "quantization_methods": ["raw", "awq", "pot"],
  "n_calibration_samples": 128,
  "n_test_samples": 20
}
```

### Example 3: Different Dataset

```json
{
  "calibration_dataset": "pile",
  "calibration_dataset_config": "all",
  "test_dataset": "pile",
  "test_dataset_config": "all"
}
```

## Testing

Run the test suite to verify all quantization methods:

```bash
python test_quantization.py
```

Output:
```
================================================================================
QUANTIZATION FRAMEWORK TEST SUITE
================================================================================

Testing framework imports...
  ✓ quantization_utils imported
  ✓ awq_quantizer imported
  ✓ gptq_quantizer imported
  ✓ pot_apot_quantizer imported
  ✓ benchmark_runner imported

Testing configuration loading...
  ✓ Config has all required fields
  ...

================================================================================
TEST RESULTS: 9 passed, 0 failed
================================================================================
```

## Advanced Usage

### Using Different Bit-Widths

Modify quantization_config to test multiple bit-widths:

```json
{
  "awq": {
    "w_bit": 2,
    "q_group_size": 128
  }
}
```

### Controlling Calibration Data

Adjust calibration parameters:

```json
{
  "n_calibration_samples": 512,
  "calibration_block_size": 1024
}
```

### Running Subset of Methods

In config.json:

```json
{
  "quantization_methods": ["raw", "awq", "apot"]
}
```

## Performance Tips

1. **Reduce calibration samples** for faster iteration:
   ```json
   {"n_calibration_samples": 64}
   ```

2. **Use smaller model** for testing:
   ```json
   {"model_name": "facebook/opt-350m"}
   ```

3. **Reduce test samples**:
   ```json
   {"n_test_samples": 10}
   ```

4. **Use GPU**: Ensure CUDA is available for faster evaluation

## Troubleshooting

### Out of Memory
- Reduce model size
- Reduce n_calibration_samples
- Reduce test_block_size

### Slow Evaluation
- Use fewer calibration samples
- Use smaller model
- Reduce grid search resolution in POT/APOT config

### NaN/Inf in Results
- Check quantization config values
- Try different scale factors
- Verify input data quality

## References

1. **AWQ**: https://arxiv.org/abs/2306.00978
2. **GPTQ**: https://arxiv.org/abs/2210.17323
3. **POT**: Power-of-Two Post-Training Quantization
4. **APOT**: Additive Power-of-Two Quantization

## License

This framework is provided as-is for research and educational purposes.

## Contributing

To add a new quantization method:

1. Create new file: `your_quantizer.py`
2. Implement: `your_quantize_model_weight(model, ...)`
3. Add to `benchmark_runner.py` benchmark methods
4. Add configuration to `config.json`
5. Add tests to `test_quantization.py`

