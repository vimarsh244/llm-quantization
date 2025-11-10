### quick run

```bash
pip install torch transformers datasets tqdm
python test_quantization.py
python setup_config.py quick_test
python benchmark_runner.py config.json
# saves restults in benchmark_results.json 
```

### other useful

- list configs

```bash
python setup_config.py list
```


- change model

```bash
# edit config.json → "model_name": "your/model"
python benchmark_runner.py config.json
```

### notes

- results are saved to `benchmark_results.json`
- see `README_QUANTIZATION.md` for details if needed


