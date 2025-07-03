#include <torch/extension.h>
// #include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

#include "include/utils.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare the CUDA function
extern "C" void launch_add_arrays(torch::Tensor& a, torch::Tensor& b, torch::Tensor& result, int n);

extern "C" void launch_subtract_arrays(torch::Tensor& a, torch::Tensor& b, torch::Tensor& result, int n);

namespace py = pybind11;


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
}
