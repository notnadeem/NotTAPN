#ifndef ALLOC_HELPERS_CUH
#define ALLOC_HELPERS_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename T> void cudaMallocChecked(T **devPtr, size_t size) {
  cudaError_t err = cudaMalloc(reinterpret_cast<void **>(devPtr), size);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void cudaMemcpyChecked(void *dst, const void *src, size_t size, cudaMemcpyKind kind) {
  cudaError_t err = cudaMemcpy(dst, src, size, kind);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#endif /* ALLOC_HELPERS_CUH */