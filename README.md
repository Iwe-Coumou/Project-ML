# Project-ML

## CUDA Support

This project uses [PyTorch](https://pytorch.org/) with optional GPU acceleration via NVIDIA CUDA.

To keep the `requirements.txt` compatible with GitHub’s dependency graph and standard `pip` resolution, CUDA-specific version tags (e.g., `+cu126`) were removed. As a result, GPU support must be installed manually if desired.

---

### Default Installation (CPU Only)

Installing dependencies normally will install the CPU version of PyTorch:

```bash
pip install -r requirements.txt
