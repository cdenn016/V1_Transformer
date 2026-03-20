"""JIT-compile and load CUDA extensions for Pure VFE kernels."""

import os
import torch

_cuda_module = None
_load_attempted = False


def get_cuda_ext():
    """Load CUDA extension, compiling on first call. Returns None if unavailable."""
    global _cuda_module, _load_attempted
    if _load_attempted:
        return _cuda_module
    _load_attempted = True

    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
        _cuda_module = load(
            name="pure_vfe_cuda",
            sources=[
                os.path.join(csrc_dir, "binding.cpp"),
                os.path.join(csrc_dir, "pairwise_kl.cu"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=False,
        )
        print("[pure_vfe] CUDA kernels compiled and loaded successfully.")
    except Exception as e:
        print(f"[pure_vfe] CUDA kernel compilation failed, falling back to PyTorch: {e}")
        _cuda_module = None

    return _cuda_module
