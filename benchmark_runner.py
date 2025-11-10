"""
Main benchmarking framework for quantization methods.

Orchestrates: config loading, model loading, calibration, quantization,
and evaluation of multiple quantization methods.
"""

import json
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import sys
import gc
import time

from quantization_utils import (
    load_config,
    load_model_and_tokenizer,
    unload_model,
    get_calibration_dataset,
    get_test_dataset,
    get_calib_feat,
    evaluate_perplexity,
    get_model_size,
    MiB,
    GiB,
)

from awq_quantizer import awq_quantize_model_weight
from gptq_quantizer import gptq_quantize_model_weight
from pot_apot_quantizer import (
    pot_quantize_model_weight,
    apot_quantize_model_weight,
)
from smooth_quant_quantizer import (
    collect_act_scales,
    smoothquant_quantize_model_weight,
)


# ==============================================================================
# BENCHMARK RESULT COLLECTION
# ==============================================================================

class BenchmarkResult:
    """Container for quantization benchmark results."""
    
    def __init__(self, method_name: str, config: Dict[str, Any]):
        self.method_name = method_name
        self.config = config
        self.perplexity: Optional[float] = None
        self.model_size_bits: Optional[int] = None
        self.model_size_mb: Optional[float] = None
        self.bits_per_byte: Optional[float] = None
        self.error: Optional[str] = None
        self.runtime_seconds: Optional[float] = None
    
    def is_success(self) -> bool:
        return self.error is None and self.perplexity is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method_name,
            'perplexity': self.perplexity,
            'model_size_mb': self.model_size_mb,
            'model_size_bits': self.model_size_bits,
            'bits_per_byte': self.bits_per_byte,
            'runtime_seconds': self.runtime_seconds,
            'error': self.error,
            'config': self.config,
        }
    
    def __str__(self) -> str:
        if self.is_success():
            bits_str = f"{self.bits_per_byte:.2f}" if self.bits_per_byte is not None else "N/A"
            return (
                f"{self.method_name:12s} | PPL: {self.perplexity:8.2f} | "
                f"Size: {self.model_size_mb:8.2f} MB | "
                f"Bits/Byte: {bits_str:>5} | "
                f"Time: {self.runtime_seconds or 0:6.2f}s"
            )
        else:
            return f"{self.method_name:12s} | ERROR: {self.error}"


# ==============================================================================
# BENCHMARK ORCHESTRATOR
# ==============================================================================

class QuantizationBenchmark:
    """Main benchmark orchestrator."""
    
    def __init__(self, config_path: str, verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        self.config = load_config(config_path)
        self.results: Dict[str, BenchmarkResult] = {}
        
        # Load model and tokenizer once
        self.model = None
        self.tokenizer = None
        self.test_dataset = None
        self.calib_samples = None
        self.input_feat = None
    
    def log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(msg)
    
    def cleanup_memory(self):
        """Clean up CUDA memory and run garbage collection."""
        if self.model is not None:
            # move model to cpu first to free gpu memory
            try:
                self.model.cpu()
            except:
                pass
        
        # clear cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # run garbage collection
        gc.collect()
    
    def setup(self):
        """Setup: load model, tokenizer, and datasets."""
        self.log("\n" + "=" * 80)
        self.log("BENCHMARK SETUP")
        self.log("=" * 80)
        
        # Load model and tokenizer
        self.log(f"\nLoading model: {self.config['model_name']}...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config['model_name'],
            device_map=self.config.get('device_map', 'auto'),
            torch_dtype=self.config.get('torch_dtype', None),
            use_fast_tokenizer=self.config.get('use_fast_tokenizer', False),
        )
        
        # Load test dataset
        self.log("\nLoading test dataset...")
        self.test_dataset = get_test_dataset(
            self.tokenizer,
            self.config['test_dataset'],
            self.config['test_dataset_config'],
            self.config['test_split'],
            n_samples=self.config.get('n_test_samples', 40),
            block_size=self.config.get('test_block_size', 2048),
        )
        
        # Load calibration dataset
        self.log("\nLoading calibration dataset...")
        self.calib_samples = get_calibration_dataset(
            self.tokenizer,
            self.config['calibration_dataset'],
            self.config['calibration_dataset_config'],
            self.config['calibration_split'],
            n_samples=self.config.get('n_calibration_samples', 256),
            block_size=self.config.get('calibration_block_size', 512),
        )
        
        self.log("Setup complete!")
    
    def _get_original_model_size_bytes(self) -> int:
        """Calculate original model size in bytes (assuming FP32/FP16)."""
        total_bytes = 0
        for param in self.model.parameters():
            # get dtype size in bytes
            if param.dtype == torch.float32:
                dtype_bytes = 4
            elif param.dtype == torch.float16:
                dtype_bytes = 2
            elif param.dtype == torch.bfloat16:
                dtype_bytes = 2
            else:
                dtype_bytes = 4  # default to 4 bytes
            total_bytes += param.numel() * dtype_bytes
        return total_bytes
    
    def _prepare_activations(self, force_refresh: bool = False):
        """Prepare activation statistics for AWQ/GPTQ methods."""
        if self.input_feat is not None and not force_refresh:
            # already collected
            return
        
        # clear old features to save memory
        if self.input_feat is not None:
            del self.input_feat
            gc.collect()
        
        self.log("\nCollecting activation statistics...")
        self.input_feat = get_calib_feat(
            self.model,
            self.tokenizer,
            self.calib_samples,
            verbose=self.verbose
        )
    
    def benchmark_raw_model(self):
        """Benchmark raw (unquantized) model."""
        self.log("\n" + "=" * 80)
        self.log("EVALUATING RAW MODEL")
        self.log("=" * 80)
        
        result = BenchmarkResult("raw", {})
        
        try:
            start_time = time.time()
            
            self.log("\nCalculating baseline perplexity...")
            perplexity = evaluate_perplexity(
                self.model,
                self.tokenizer,
                self.test_dataset,
                n_samples=self.config.get('n_test_samples', 40),
                block_size=self.config.get('test_block_size', 2048),
                verbose=self.verbose
            )
            
            model_size = get_model_size(
                self.model,
                data_width=32,
                group_size=-1
            )
            
            original_size_bytes = self._get_original_model_size_bytes()
            bits_per_byte = model_size / original_size_bytes if original_size_bytes > 0 else None
            
            runtime = time.time() - start_time
            
            result.perplexity = perplexity
            result.model_size_bits = model_size
            result.model_size_mb = model_size / (8 * MiB)
            result.bits_per_byte = bits_per_byte
            result.runtime_seconds = runtime
            
            self.log(f"✓ Raw Model - Perplexity: {perplexity:.2f}, Size: {model_size / (8 * MiB):.2f} MB, Bits/Byte: {bits_per_byte:.2f}, Time: {runtime:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            self.log(f"✗ Raw Model - Error: {e}")
        
        self.results['raw'] = result
        return result
    
    def benchmark_awq(self):
        """Benchmark AWQ quantization."""
        if 'awq' not in self.config['quantization_methods']:
            return None
        
        self.log("\n" + "=" * 80)
        self.log("BENCHMARKING AWQ")
        self.log("=" * 80)
        
        result = BenchmarkResult("awq", self.config['quantization_config']['awq'])
        
        try:
            start_time = time.time()
            
            # clean up memory before loading fresh model
            self.cleanup_memory()
            unload_model(self.model)
            self.model = None
            
            # load fresh model copy
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config['model_name'],
                device_map=self.config.get('device_map', 'auto'),
                torch_dtype=self.config.get('torch_dtype', None),
                use_fast_tokenizer=self.config.get('use_fast_tokenizer', False),
            )
            
            # prepare activations (force refresh for new model)
            self._prepare_activations(force_refresh=True)
            
            # Apply AWQ quantization
            self.log("\nApplying AWQ quantization...")
            config = self.config['quantization_config']['awq']
            
            awq_quantize_model_weight(
                self.model,
                w_bit=config['w_bit'],
                q_group_size=config['q_group_size'],
                input_feat=self.input_feat,
                protect_ratio=config.get('protect_ratio', 0.01),
                scale_factor=config.get('scale_factor', 2.0)
            )
            
            # Evaluate
            self.log("Calculating AWQ perplexity...")
            perplexity = evaluate_perplexity(
                self.model,
                self.tokenizer,
                self.test_dataset,
                n_samples=self.config.get('n_test_samples', 40),
                block_size=self.config.get('test_block_size', 2048),
                verbose=self.verbose
            )
            
            model_size = get_model_size(
                self.model,
                data_width=config['w_bit'],
                group_size=config['q_group_size'],
                use_zero_point=True  # AWQ uses zero points
            )
            
            original_size_bytes = self._get_original_model_size_bytes()
            bits_per_byte = model_size / original_size_bytes if original_size_bytes > 0 else None
            runtime = time.time() - start_time
            
            result.perplexity = perplexity
            result.model_size_bits = model_size
            result.model_size_mb = model_size / (8 * MiB)
            result.bits_per_byte = bits_per_byte
            result.runtime_seconds = runtime
            
            self.log(f"✓ AWQ - Perplexity: {perplexity:.2f}, Size: {model_size / (8 * MiB):.2f} MB, Bits/Byte: {bits_per_byte:.2f}, Time: {runtime:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            self.log(f"✗ AWQ - Error: {e}")
            import traceback
            traceback.print_exc()
        
        self.results['awq'] = result
        return result
    
    def benchmark_gptq(self):
        """Benchmark GPTQ quantization."""
        if 'gptq' not in self.config['quantization_methods']:
            return None
        
        self.log("\n" + "=" * 80)
        self.log("BENCHMARKING GPTQ")
        self.log("=" * 80)
        
        result = BenchmarkResult("gptq", self.config['quantization_config']['gptq'])
        
        try:
            start_time = time.time()
            
            # clean up memory before loading fresh model
            self.cleanup_memory()
            unload_model(self.model)
            self.model = None
            
            # load fresh model
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config['model_name'],
                device_map=self.config.get('device_map', 'auto'),
                torch_dtype=self.config.get('torch_dtype', None),
                use_fast_tokenizer=self.config.get('use_fast_tokenizer', False),
            )
            
            # prepare activations (force refresh for new model)
            self._prepare_activations(force_refresh=True)
            
            # Apply GPTQ quantization
            self.log("\nApplying GPTQ quantization...")
            config = self.config['quantization_config']['gptq']
            
            gptq_quantize_model_weight(
                self.model,
                w_bit=config['w_bit'],
                q_group_size=config['q_group_size'],
                input_feat=self.input_feat,
                perp_damp=config.get('perp_damp', 0.01),
                blocksize=config.get('blocksize', 128),
                nsamples=config.get('nsamples', 128),
                actorder=config.get('actorder', False),
                verbose=self.verbose
            )
            
            # Evaluate
            self.log("Calculating GPTQ perplexity...")
            perplexity = evaluate_perplexity(
                self.model,
                self.tokenizer,
                self.test_dataset,
                n_samples=self.config.get('n_test_samples', 40),
                block_size=self.config.get('test_block_size', 2048),
                verbose=self.verbose
            )
            
            model_size = get_model_size(
                self.model,
                data_width=config['w_bit'],
                group_size=config['q_group_size'],
                use_zero_point=True  # GPTQ uses zero points
            )
            
            original_size_bytes = self._get_original_model_size_bytes()
            bits_per_byte = model_size / original_size_bytes if original_size_bytes > 0 else None
            runtime = time.time() - start_time
            
            result.perplexity = perplexity
            result.model_size_bits = model_size
            result.model_size_mb = model_size / (8 * MiB)
            result.bits_per_byte = bits_per_byte
            result.runtime_seconds = runtime
            
            self.log(f"✓ GPTQ - Perplexity: {perplexity:.2f}, Size: {model_size / (8 * MiB):.2f} MB, Bits/Byte: {bits_per_byte:.2f}, Time: {runtime:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            self.log(f"✗ GPTQ - Error: {e}")
            import traceback
            traceback.print_exc()
        
        self.results['gptq'] = result
        return result
    
    def benchmark_pot(self):
        """Benchmark Power-of-Two quantization."""
        if 'pot' not in self.config['quantization_methods']:
            return None
        
        self.log("\n" + "=" * 80)
        self.log("BENCHMARKING POWER-OF-TWO (POT)")
        self.log("=" * 80)
        
        result = BenchmarkResult("pot", self.config['quantization_config']['pot'])
        
        try:
            start_time = time.time()
            
            # clean up memory before loading fresh model
            self.cleanup_memory()
            unload_model(self.model)
            self.model = None
            
            # load fresh model
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config['model_name'],
                device_map=self.config.get('device_map', 'auto'),
                torch_dtype=self.config.get('torch_dtype', None),
                use_fast_tokenizer=self.config.get('use_fast_tokenizer', False),
            )
            
            # Apply POT quantization
            self.log("\nApplying POT quantization...")
            config = self.config['quantization_config']['pot']
            
            pot_quantize_model_weight(
                self.model,
                w_bit=config['w_bit'],
                q_group_size=config['q_group_size']
            )
            
            # Evaluate
            self.log("Calculating POT perplexity...")
            perplexity = evaluate_perplexity(
                self.model,
                self.tokenizer,
                self.test_dataset,
                n_samples=self.config.get('n_test_samples', 40),
                block_size=self.config.get('test_block_size', 2048),
                verbose=self.verbose
            )
            
            model_size = get_model_size(
                self.model,
                data_width=config['w_bit'],
                group_size=config['q_group_size'],
                use_zero_point=False  # POT doesn't use zero points (symmetric quantization)
            )
            
            original_size_bytes = self._get_original_model_size_bytes()
            bits_per_byte = model_size / original_size_bytes if original_size_bytes > 0 else None
            runtime = time.time() - start_time
            
            result.perplexity = perplexity
            result.model_size_bits = model_size
            result.model_size_mb = model_size / (8 * MiB)
            result.bits_per_byte = bits_per_byte
            result.runtime_seconds = runtime
            
            self.log(f"✓ POT - Perplexity: {perplexity:.2f}, Size: {model_size / (8 * MiB):.2f} MB, Bits/Byte: {bits_per_byte:.2f}, Time: {runtime:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            self.log(f"✗ POT - Error: {e}")
            import traceback
            traceback.print_exc()
        
        self.results['pot'] = result
        return result
    
    def benchmark_apot(self):
        """Benchmark Additive Power-of-Two quantization."""
        if 'apot' not in self.config['quantization_methods']:
            return None
        
        self.log("\n" + "=" * 80)
        self.log("BENCHMARKING ADDITIVE POWER-OF-TWO (APOT)")
        self.log("=" * 80)
        
        result = BenchmarkResult("apot", self.config['quantization_config']['apot'])
        
        try:
            start_time = time.time()
            
            # clean up memory before loading fresh model
            self.cleanup_memory()
            unload_model(self.model)
            self.model = None
            
            # load fresh model
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config['model_name'],
                device_map=self.config.get('device_map', 'auto'),
                torch_dtype=self.config.get('torch_dtype', None),
                use_fast_tokenizer=self.config.get('use_fast_tokenizer', False),
            )
            
            # Apply APOT quantization
            self.log("\nApplying APOT quantization...")
            config = self.config['quantization_config']['apot']
            
            apot_quantize_model_weight(
                self.model,
                w_bit=config['w_bit'],
                q_group_size=config['q_group_size'],
                k=config.get('k', 2)
            )
            
            # Evaluate
            self.log("Calculating APOT perplexity...")
            perplexity = evaluate_perplexity(
                self.model,
                self.tokenizer,
                self.test_dataset,
                n_samples=self.config.get('n_test_samples', 40),
                block_size=self.config.get('test_block_size', 2048),
                verbose=self.verbose
            )
            
            model_size = get_model_size(
                self.model,
                data_width=config['w_bit'],
                group_size=config['q_group_size'],
                use_zero_point=False  # APOT doesn't use zero points (symmetric quantization)
            )
            
            original_size_bytes = self._get_original_model_size_bytes()
            bits_per_byte = model_size / original_size_bytes if original_size_bytes > 0 else None
            runtime = time.time() - start_time
            
            result.perplexity = perplexity
            result.model_size_bits = model_size
            result.model_size_mb = model_size / (8 * MiB)
            result.bits_per_byte = bits_per_byte
            result.runtime_seconds = runtime
            
            self.log(f"✓ APOT - Perplexity: {perplexity:.2f}, Size: {model_size / (8 * MiB):.2f} MB, Bits/Byte: {bits_per_byte:.2f}, Time: {runtime:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            self.log(f"✗ APOT - Error: {e}")
            import traceback
            traceback.print_exc()
        
        self.results['apot'] = result
        return result
    
    def benchmark_smoothquant(self):
        """Benchmark SmoothQuant quantization."""
        if 'smoothquant' not in self.config['quantization_methods']:
            return None
        
        self.log("\n" + "=" * 80)
        self.log("BENCHMARKING SMOOTHQUANT")
        self.log("=" * 80)
        
        result = BenchmarkResult("smoothquant", self.config['quantization_config']['smoothquant'])
        
        try:
            start_time = time.time()
            
            # clean up memory before loading fresh model
            self.cleanup_memory()
            unload_model(self.model)
            self.model = None
            
            # load fresh model
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config['model_name'],
                device_map=self.config.get('device_map', 'auto'),
                torch_dtype=self.config.get('torch_dtype', None),
                use_fast_tokenizer=self.config.get('use_fast_tokenizer', False),
            )
            
            # Apply SmoothQuant quantization
            self.log("\nApplying SmoothQuant quantization...")
            config = self.config['quantization_config']['smoothquant']
            
            # collect activation scales
            self.log("Collecting activation scales...")
            act_scales = collect_act_scales(
                self.model,
                self.calib_samples,
                verbose=self.verbose
            )
            
            # apply quantization with smoothing
            smoothquant_quantize_model_weight(
                self.model,
                w_bit=config['w_bit'],
                q_group_size=config['q_group_size'],
                act_scales=act_scales,
                alpha=config.get('alpha', 0.5),
                verbose=self.verbose
            )
            
            # Evaluate
            self.log("Calculating SmoothQuant perplexity...")
            perplexity = evaluate_perplexity(
                self.model,
                self.tokenizer,
                self.test_dataset,
                n_samples=self.config.get('n_test_samples', 40),
                block_size=self.config.get('test_block_size', 2048),
                verbose=self.verbose
            )
            
            model_size = get_model_size(
                self.model,
                data_width=config['w_bit'],
                group_size=config['q_group_size'],
                use_zero_point=False  # SmoothQuant uses symmetric quantization (no zero point)
            )
            
            original_size_bytes = self._get_original_model_size_bytes()
            bits_per_byte = model_size / original_size_bytes if original_size_bytes > 0 else None
            runtime = time.time() - start_time
            
            result.perplexity = perplexity
            result.model_size_bits = model_size
            result.model_size_mb = model_size / (8 * MiB)
            result.bits_per_byte = bits_per_byte
            result.runtime_seconds = runtime
            
            self.log(f"✓ SmoothQuant - Perplexity: {perplexity:.2f}, Size: {model_size / (8 * MiB):.2f} MB, Bits/Byte: {bits_per_byte:.2f}, Time: {runtime:.2f}s")
            
        except Exception as e:
            result.error = str(e)
            self.log(f"✗ SmoothQuant - Error: {e}")
            import traceback
            traceback.print_exc()
        
        self.results['smoothquant'] = result
        return result
    
    def run_all_benchmarks(self):
        """Run all benchmarks specified in config."""
        self.setup()
        
        # Always benchmark raw model first
        self.benchmark_raw_model()
        
        # Then benchmark selected quantization methods
        if 'awq' in self.config['quantization_methods']:
            self.benchmark_awq()
        
        if 'gptq' in self.config['quantization_methods']:
            self.benchmark_gptq()
        
        if 'pot' in self.config['quantization_methods']:
            self.benchmark_pot()
        
        if 'apot' in self.config['quantization_methods']:
            self.benchmark_apot()
        
        if 'smoothquant' in self.config['quantization_methods']:
            self.benchmark_smoothquant()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary."""
        self.log("\n" + "=" * 80)
        self.log("BENCHMARK SUMMARY")
        self.log("=" * 80)
        
        self.log(f"\nModel: {self.config['model_name']}")
        self.log(f"Calibration: {self.config['calibration_dataset']}")
        self.log(f"Test Dataset: {self.config['test_dataset']}")
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.log("\n" + "-" * 100)
        self.log(f"{'Method':<12} | {'Perplexity':>8} | {'Size (MB)':>10} | {'Bits/Byte':>9} | {'Runtime (s)':>10}")
        self.log("-" * 100)
        
        for method_name, result in self.results.items():
            if result.is_success():
                bits_str = f"{result.bits_per_byte:.2f}" if result.bits_per_byte is not None else "N/A"
                self.log(
                    f"{result.method_name:<12} | "
                    f"{result.perplexity:>8.2f} | "
                    f"{result.model_size_mb:>10.2f} | "
                    f"{bits_str:>9} | "
                    f"{result.runtime_seconds or 0:>10.2f}"
                )
            else:
                self.log(f"{result.method_name:<12} | ERROR: {result.error}")
        
        self.log("-" * 100)
        
        # Calculate improvements vs raw
        if 'raw' in self.results and self.results['raw'].is_success():
            raw_ppl = self.results['raw'].perplexity
            raw_size = self.results['raw'].model_size_mb
            
            self.log("\nImprovements vs Raw Model:")
            for method_name, result in self.results.items():
                if method_name != 'raw' and result.is_success():
                    ppl_degradation = (result.perplexity / raw_ppl - 1) * 100
                    size_reduction = (1 - result.model_size_mb / raw_size) * 100
                    self.log(
                        f"  {method_name:10s}: "
                        f"PPL +{ppl_degradation:+6.2f}% | "
                        f"Size -{size_reduction:+6.2f}%"
                    )
        
        self.log("=" * 100 + "\n")
    
    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save results to JSON file."""
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': {k: v.to_dict() for k, v in self.results.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.log(f"\nResults saved to {output_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point."""
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print(f"Loading configuration from: {config_path}")
    
    benchmark = QuantizationBenchmark(config_path, verbose=True)
    benchmark.run_all_benchmarks()
    benchmark.save_results("benchmark_results.json")


if __name__ == "__main__":
    main()

