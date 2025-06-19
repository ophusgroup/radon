"""
GPU-accelerated radon transforms using CUDA and pybind11
"""

try:
    from .cuda._cuda_add import add_arrays

    __all__ = ["add_arrays"]
except ImportError:
    print("Warning: CUDA extension not available. Make sure to build the extension.")
    __all__ = []

__version__ = "0.1.0"
