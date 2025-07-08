#include <torch/extension.h>
// #include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

#include "include/utils.h"
#include "include/cache.h"
#include "include/texture.h"
#include "include/forward.h"
#include "include/backprojection.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare the CUDA function
extern "C" void launch_add_arrays(torch::Tensor& a, torch::Tensor& b, torch::Tensor& result, int n);

extern "C" void launch_subtract_arrays(torch::Tensor& a, torch::Tensor& b, torch::Tensor& result, int n);

namespace py = pybind11;

torch::Tensor radon_forward(torch::Tensor x, torch::Tensor angles, TextureCache &tex_cache, const RaysCfg rays_cfg) {
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int device = x.device().index();

    // allocate output sinogram tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
    auto y = torch::empty({batch_size, rays_cfg.n_angles, rays_cfg.det_count}, options);

    if (dtype == torch::kFloat16) {
        radon_forward_cuda((unsigned short *) x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                           (unsigned short *) y.data_ptr<at::Half>(),
                           tex_cache, rays_cfg, batch_size, device);
    } else {
        radon_forward_cuda(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                           tex_cache, rays_cfg, batch_size, device);
    }
    return y;
}

torch::Tensor
radon_backward(torch::Tensor x, torch::Tensor angles, TextureCache &tex_cache, const RaysCfg rays_cfg) {
    CHECK_INPUT(x);
    CHECK_INPUT(angles);

    auto dtype = x.dtype();

    const int batch_size = x.size(0);
    const int device = x.device().index();

    TORCH_CHECK(angles.size(0) <= 4096, "Can only support up to 4096 angles")

    // create output image tensor
    auto options = torch::TensorOptions().dtype(dtype).device(x.device());
    auto y = torch::empty({batch_size, rays_cfg.height, rays_cfg.width}, options);

    if (dtype == torch::kFloat16) {
        radon_backward_cuda((unsigned short *) x.data_ptr<at::Half>(), angles.data_ptr<float>(),
                            (unsigned short *) y.data_ptr<at::Half>(),
                            tex_cache, rays_cfg, batch_size, device);
    } else {
        radon_backward_cuda(x.data_ptr<float>(), angles.data_ptr<float>(), y.data_ptr<float>(),
                            tex_cache, rays_cfg, batch_size, device);
    }

    return y;
}

// Python wrapper function
void add_arrays_cuda(
    torch::Tensor& input_a,  // Make input parameters const references
    torch::Tensor& input_b,
    torch::Tensor& output_c
) {
    CHECK_INPUT(input_a);
    CHECK_INPUT(input_b);
    
    // Check dimensions and sizes using tensor methods
    if (input_a.dim() != 1 || input_b.dim() != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    if (input_a.size(0) != input_b.size(0)) {
        throw std::runtime_error("Input arrays must have the same size");
    }
    
    int n = input_a.size(0);

    // Launch CUDA kernel
    launch_add_arrays(input_a, input_b, output_c, n);
    
    // No need for cleanup since we are not allocating memory on the GPU here.

    // Return nothing; the output tensor refers to the same allocated memory.
}

void subtract_arrays_cuda(
    torch::Tensor& input_a,  // Make input parameters const references
    torch::Tensor& input_b,
    torch::Tensor& output_c
) {
    CHECK_INPUT(input_a);
    CHECK_INPUT(input_b);
    
    // Check dimensions and sizes using tensor methods
    if (input_a.dim() != 1 || input_b.dim() != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    if (input_a.size(0) != input_b.size(0)) {
        throw std::runtime_error("Input arrays must have the same size");
    }
    
    int n = input_a.size(0);

    // Launch CUDA kernel
    launch_subtract_arrays(input_a, input_b, output_c, n);
    
    // No need for cleanup since we are not allocating memory on the GPU here.

    // Return nothing; the output tensor refers to the same allocated memory.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Simple CUDA array addition example";
    m.def("add_arrays", &add_arrays_cuda, "Add two arrays using CUDA");
    m.def("subtract_arrays", &subtract_arrays_cuda, "Subtract two arrays using CUDA");

    m.def("forward", &radon_forward, "Radon forward projection");
    m.def("backward", &radon_backward, "Radon back projection");

    py::class_<TextureCache>(m, "TextureCache")
        .def(py::init<size_t>())
        .def("free", &TextureCache::free);

    py::class_<RaysCfg>(m, "RaysCfg")
        .def(py::init<int, int, int, float, int, bool>())  // Only 6 parameters
        .def_readwrite("det_count", &RaysCfg::det_count);
}
