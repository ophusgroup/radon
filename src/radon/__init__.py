"""
GPU-accelerated radon transforms using CUDA and pybind11
"""

try:
    # This seems necessary, as otherwise the cuda kernels cannot be loaded, but it feels like
    # bad practice because now the entirity of torch is accessible from radon.torch
    import torch as torch
except ImportError as err:
    print("how the heck did you get here? torch not installed")
    raise ImportError("Torch is required for this package.") from err

try:
    from .cuda import add_arrays as add_arrays
    from .cuda import subtract_arrays as subtract_arrays

    __all__ = ["add_arrays", "subtract_arrays"]
    print("CUDA extension loaded successfully.")

except ImportError:
    print("Warning: CUDA extension not available. Make sure to build the extension.")
    __all__ = []

__version__ = "0.1.0"
