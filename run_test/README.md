# MS2Forge smoke test

`demo_forward.py` loads `configs/sample.yml`, creates a synthetic four-node
graph, executes one MS2Forge forward pass, and validates output shape,
finiteness, and probability normalization. No dataset or checkpoint is used.

```bash
conda activate ms2forge
python run_test/demo_forward.py --device auto
```

Use `--device cpu` for a CPU-only run. Successful execution prints a JSON
object with `"status": "passed"`.
