import os
import subprocess

import pybind11
from pybind11.setup_helpers import Pybind11Extension
from pybind11.setup_helpers import build_ext as _build_ext
from setuptools import setup


class build_ext(_build_ext):
    """Custom build extension to handle CUDA compilation"""

    def build_extensions(self):
        # Compile CUDA files to object files first
        for ext in self.extensions:
            self._compile_cuda_files(ext)
        super().build_extensions()

    def _compile_cuda_files(self, ext):
        cuda_sources = [src for src in ext.sources if src.endswith(".cu")]
        object_files = []

        # Compile CUDA files to object files
        for cuda_src in cuda_sources:
            obj_file = cuda_src.replace(".cu", ".o")
            nvcc_cmd = [
                "nvcc",
                "-c",
                cuda_src,
                "-o",
                obj_file,
                "-std=c++14",
                "--compiler-options",
                "-fPIC",
                "-I" + pybind11.get_include(),
            ]

            # Add CUDA include paths
            cuda_paths = get_cuda_paths()
            for inc_path in cuda_paths["include"]:
                nvcc_cmd.extend(["-I", inc_path])

            print(f"Compiling CUDA: {' '.join(nvcc_cmd)}")
            subprocess.run(nvcc_cmd, check=True)
            object_files.append(obj_file)

        # Remove .cu files from sources and add object files to extra_objects
        ext.sources = [s for s in ext.sources if not s.endswith(".cu")]
        if not hasattr(ext, "extra_objects"):
            ext.extra_objects = []
        ext.extra_objects.extend(object_files)


# CUDA configuration
def get_cuda_paths():
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.path.exists(cuda_home):
        cuda_home = "/usr"  # Try system CUDA installation
    return {
        "include": [os.path.join(cuda_home, "include")],
        "lib": [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")],
    }


cuda_paths = get_cuda_paths()

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "radon.cuda._cuda_add",
        [
            "src/radon/cuda/ext.cpp",
            "src/radon/cuda/csrc/add_arrays_kernel.cu",
        ],
        include_dirs=[
            pybind11.get_include(),
        ]
        + cuda_paths["include"],
        library_dirs=cuda_paths["lib"],
        libraries=["cuda", "cudart"],
        language="c++",
        cxx_std=14,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
