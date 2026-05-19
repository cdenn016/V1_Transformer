"""JIT-compile and load CUDA extensions for Pure VFE kernels."""

import logging
import os
import torch

logger = logging.getLogger(__name__)

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

        # Find ninja: Anaconda on Windows often has it installed but not on PATH.
        # PATH is scoped via try/finally so a successful compile cannot leak
        # the ninja directory into the rest of the process.
        import shutil
        ninja_path = shutil.which("ninja")
        _saved_path = os.environ.get("PATH")
        _path_mutated = False
        if ninja_path is None:
            # Try to find ninja via the Python package's bundled binary
            try:
                import ninja
                ninja_dir = os.path.dirname(ninja.__file__)
                # ninja package puts the binary in its package directory
                for candidate in [
                    os.path.join(ninja_dir, "ninja"),
                    os.path.join(ninja_dir, "ninja.exe"),
                    os.path.join(os.path.dirname(ninja_dir), "Scripts", "ninja.exe"),
                    os.path.join(os.path.dirname(ninja_dir), "bin", "ninja"),
                ]:
                    if os.path.isfile(candidate):
                        # Add its directory to PATH so cpp_extension can find it
                        os.environ["PATH"] = os.path.dirname(candidate) + os.pathsep + (_saved_path or "")
                        _path_mutated = True
                        break
            except ImportError:
                pass

        try:
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
        finally:
            if _path_mutated:
                if _saved_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = _saved_path
        logger.info("CUDA kernels compiled and loaded successfully.")
    except Exception as e:
        # Use logger.exception to preserve the traceback rather than the
        # bare exception message — library users may need it to debug
        # build failures (nvcc, ninja, CUDA version mismatch).
        logger.exception("CUDA kernel compilation failed, falling back to PyTorch: %s", e)
        _cuda_module = None

    return _cuda_module
