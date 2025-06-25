#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

// Declare the CUDA function
extern "C" void launch_add_arrays(const float* a, const float* b, float* c, int n);

extern "C" void launch_subtract_arrays(const float* a, const float* b, float* c, int n);

// Python wrapper function
pybind11::array_t<float> add_arrays_cuda(
    pybind11::array_t<float> input_a,
    pybind11::array_t<float> input_b
) {
    auto buf_a = input_a.request();
    auto buf_b = input_b.request();

    if (buf_a.ndim != 1 || buf_b.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }

    if (buf_a.size != buf_b.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    int n = buf_a.size;

    // Allocate GPU memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(d_a, buf_a.ptr, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, buf_b.ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    launch_add_arrays(d_a, d_b, d_c, n);

    // Create output array
    auto result = pybind11::array_t<float>(n);
    auto buf_result = result.request();

    // Copy result back to host
    cudaMemcpy(buf_result.ptr, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}

pybind11::array_t<float> subtract_arrays_cuda(
    pybind11::array_t<float> input_a,
    pybind11::array_t<float> input_b
) {
    auto buf_a = input_a.request();
    auto buf_b = input_b.request();

    if (buf_a.ndim != 1 || buf_b.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }

    if (buf_a.size != buf_b.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    int n = buf_a.size;

    // Allocate GPU memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copy input data to GPU
    cudaMemcpy(d_a, buf_a.ptr, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, buf_b.ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    launch_subtract_arrays(d_a, d_b, d_c, n);

    // Create output array
    auto result = pybind11::array_t<float>(n);
    auto buf_result = result.request();

    // Copy result back to host
    cudaMemcpy(buf_result.ptr, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}

PYBIND11_MODULE(_cuda_add, m) {
    m.doc() = "Simple CUDA array addition example";
    m.def("add_arrays", &add_arrays_cuda, "Add two arrays using CUDA");
    m.def("subtract_arrays", &subtract_arrays_cuda, "Subtract two arrays using CUDA");
}
