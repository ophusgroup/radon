#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Simple CUDA kernel that adds two arrays
__global__ void add_arrays_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        // Print hello from each thread (first few threads only to avoid spam)
        if (idx < 5) {
            printf("Hello from CUDA thread %d: %.2f + %.2f = %.2f\n", idx, a[idx], b[idx], c[idx]);
        }
    }
}

// Host function to launch the kernel
extern "C" {
    void launch_add_arrays(const float* a, const float* b, float* c, int n) {
        dim3 blockSize(256);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

        add_arrays_kernel<<<gridSize, blockSize>>>(a, b, c, n);
        cudaDeviceSynchronize();
    }
}
