#ifndef VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_CUH_
#define VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_CUH_

#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {

template <typename T> struct SimpleDynamicArray {
  T *arr;
  size_t size;
  size_t capacity;

  __host__ SimpleDynamicArray() : size(0), capacity(1) { cudaMallocManaged(&arr, capacity * sizeof(T)); }

  __host__ SimpleDynamicArray(size_t cap) : size(0), capacity(cap) { cudaMallocManaged(&arr, capacity * sizeof(T)); }

  __host__ ~SimpleDynamicArray() { cudaFree(arr); }

  __host__ __device__ void resize() {
    capacity *= 2;
    T *temp;
    cudaMallocManaged(&temp, capacity * sizeof(T));
    for (size_t i = 0; i < size; i++) {
      temp[i] = arr[i];
    }
    cudaFree(arr);
    arr = temp;
  }

  __host__ __device__ void add(T value) {
    if (size >= capacity) {
      resize();
    }
    arr[size++] = value;
  }

  __host__ __device__ void remove(size_t index) {
    if (index >= size) return;
    for (size_t i = index; i < size - 1; i++) {
      arr[i] = arr[i + 1];
    }
    size--;
  }

  __host__ __device__ void set(size_t index, T value) {
    if (index >= size) return;
    arr[index] = value;
  }

  __host__ __device__ T get(size_t index) const {
    if (index >= size) {
      return T();
    }
    return arr[index];
  }

  __host__ __device__ bool empty() const { return size == 0; }
};

} // namespace VerifyTAPN::Cuda

#endif
