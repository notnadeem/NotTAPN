#ifndef VERIFYTAPN_ATLER_CUDADYNAMICARRAY_CUH_
#define VERIFYTAPN_ATLER_CUDADYNAMICARRAY_CUH_

#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {

template <typename T> struct CudaDynamicArray {
  T *arr;
  size_t size;
  size_t capacity;

  __host__ __device__ CudaDynamicArray() : size(0), capacity(1) { cudaMalloc(&arr, capacity * sizeof(T)); }

  __host__ __device__ CudaDynamicArray(size_t cap) : size(0) {
    if (cap == 0)
      capacity = 1;
    else
      capacity = cap;
    cudaMalloc(&arr, capacity * sizeof(T));
  }

  // Try no to use this constructor due to possible double memry freeing
  __host__ __device__ ~CudaDynamicArray() { cudaFree(arr); }

  __host__ __device__ void resize() {
    capacity *= 2;
    T *temp;
    cudaMalloc(&temp, capacity * sizeof(T));
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

  __host__ __device__ void insert(size_t index, T value) {
    if (index >= size) {
      printf("Index out of bounds: insert operation");
    }
    if (size >= capacity) {
      resize();
    }
    for (size_t i = size - 1; i >= index; i--) {
      arr[i + 1] = arr[i];
    }
    arr[index] = value;
    size++;
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

  __host__ __device__ void clear() {
    // NOTE: if the array contains pointers, we need to delete them and set all the values to nullptr
    size = 0;
  }

  __host__ __device__ bool empty() const { return size == 0; }
};

} // namespace VerifyTAPN::Cuda

#endif
