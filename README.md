# Project-ML

## CUDA Support

This project uses PyTorch with optional GPU acceleration via NVIDIA CUDA.

To keep the `requirements.txt` compatible with GitHub’s dependency graph and standard `pip` resolution, CUDA-specific version tags (e.g., `+cu126`) were removed. As a result, GPU support must be installed manually if desired.

---

### Default Installation (CPU Only)

Installing dependencies normally will install the CPU version of PyTorch:

    pip install -r requirements.txt

This works on any system but does not enable GPU acceleration.

---

### Installing with CUDA Support (Recommended for GPU Systems)

If you have:

- An NVIDIA GPU  
- Compatible GPU drivers installed  
- A supported CUDA version  

Install PyTorch with the appropriate CUDA build from the official PyTorch index:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Replace `cu126` with the CUDA version that matches your system.

You can check your installed CUDA driver version with:

    nvidia-smi

---

### Verifying GPU Availability

After installation, verify CUDA support in Python:

    import torch

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

If `torch.cuda.is_available()` returns `True`, GPU acceleration is enabled.

---

### Notes

- The project runs correctly on CPU; GPU usage is optional.
- CUDA installation is system-dependent and therefore not enforced in `requirements.txt`.
- For full compatibility details, see the official PyTorch installation guide:  
  https://pytorch.org/get-started/locally/
