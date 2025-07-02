import glob
import os
import os.path as osp
import pathlib

from setuptools import setup


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


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


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("src", "radon", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
        osp.join(extensions_dir, "csrc", "*.cpp")
    )
    sources += [osp.join(extensions_dir, "ext.cpp")]

    # Updated to use C++17 or higher
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "--use_fast_math", "-std=c++17"],
    }
    extra_link_args = ["-s"]

    current_dir = pathlib.Path(__file__).parent.resolve()
    include_dirs = [osp.join(current_dir, "src", "radon", "cuda", "include")]

    # Add CUDA include paths
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.path.exists(cuda_home):
        cuda_home = "/usr"  # Try system CUDA installation

    cuda_include = os.path.join(cuda_home, "include")
    if os.path.exists(cuda_include):
        include_dirs.append(cuda_include)

    # Add library directories for CUDA
    library_dirs = []
    cuda_lib_dirs = [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]
    for lib_dir in cuda_lib_dirs:
        if os.path.exists(lib_dir):
            library_dirs.append(lib_dir)

    extension = CUDAExtension(
        "radon.cuda._cuda_add",
        sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["cuda", "cudart"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return [extension]


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_ext()},
    zip_safe=False,
    python_requires=">=3.8",
)
