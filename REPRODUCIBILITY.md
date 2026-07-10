# Reproducibility Notes

## Random seeds

The scripts expose command-line seed arguments and set NumPy/PyTorch random seeds where applicable:

```bash
--seed 0
```

For EEG, the example uses `--seed 42` to match the current script default.

## Hardware and software variability

Small numerical differences can occur across:

- GPU models;
- CUDA/cuDNN versions;
- PyTorch versions;
- PyTorch Geometric builds;
- BLAS/LAPACK implementations;
- CPU vs GPU execution paths.

For this reason, exact bitwise identity is not guaranteed across environments. When checking reproduced embeddings against a reference run, use tolerance-based numerical comparison:

```python
import numpy as np

ref = np.load("reference_embedding.npy")
new = np.load("reproduced_embedding.npy")

print("same shape:", ref.shape == new.shape)
print("allclose:", np.allclose(ref, new, rtol=1e-6, atol=1e-6))
print("max abs diff:", np.max(np.abs(ref - new)))
```

## Recommended reporting

For each run, record:

```text
Python version
Operating system
GPU model
CUDA version
PyTorch version
PyTorch Geometric version
Random seed
Command line used
Input data checksum or version
```

