"""
GPU-accelerated radon transforms using CUDA and pybind11
"""

try:
    # This seems necessary, as otherwise the cuda kernels cannot be loaded, but it feels like
    # bad practice because now the entirity of torch is accessible from radon.radon.torch
    from .radon import Radon
except ImportError as err:
    print("Unable to import Radon class from radon module.")
    raise ImportError("Radon unavailable. Check Torch availability") from err

try:
    from .cuda_backend import add_arrays, subtract_arrays, forward, backward
    __all__ = ["add_arrays", "subtract_arrays", "forward", "backward"]
    print("CUDA backend loaded successfully.")
except ImportError as err:
    print("Warning: CUDA backend not available. Make sure to build the extension.")
    __all__ = []
    raise ImportError("CUDA backend is unavailable.") from err

try:
    from .cuda import RaysCfg, TextureCache
    __all__.append("TextureCache")
    __all__.append("RaysCfg")
    print("RaysCfg loaded successfully.")
except ImportError as err:
    print("Warning: RaysCfg not available. Make sure to build the extension.")
    raise ImportError("RaysCfg is unavailable.") from err

__version__ = "0.1.0"