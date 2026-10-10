# GPU setup

1. Install CPU dependencies:

```bash
pip install -r requirements-cpu.txt
```

2. Install a CuPy wheel matching CUDA, e.g. `cupy-cuda12x` or `cupy-cuda11x`.

3. Check the environment:

```bash
python scripts/check_environment.py
```

4. Run the reduced two-way case:

```bash
python scripts/run_gpu_quick.py
```

The quick case is an execution/reproducibility check, not the article-scale
particle/seed ensemble.
