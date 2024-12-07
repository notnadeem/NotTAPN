#ifndef VERIFYTAPN_ATLER_CUDADYNAMICARRAY_CUH_
#define VERIFYTAPN_ATLER_CUDADYNAMICARRAY_CUH_

#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {

template <typename T> struct CudaDynamicArray {
  T *arr;
  int size;
  size_t capacity;
  bool ownsArray;

  __host__ __device__ CudaDynamicArray() : size(0), capacity(1), ownsArray(true) {
    arr = new T[capacity];
  }

  __host__ __device__ CudaDynamicArray(size_t initialCapacity) : size(0), ownsArray(true) {
    capacity = (initialCapacity == 0) ? 1 : initialCapacity;
    if (capacity == 0)
      capacity = 1;
    arr = new T[capacity];
  }

  __host__ __device__ CudaDynamicArray(const CudaDynamicArray<T> &other) {
    size = other.size;
    capacity = other.capacity;
    ownsArray = other.ownsArray;
    arr = new T[capacity];
    for (size_t i = 0; i < size; i++) {
      arr[i] = other.arr[i];
    }
  }

  __host__ __device__ ~CudaDynamicArray() {
    if (ownsArray) {
      delete[] arr;
    }
  }

  __host__ __device__ void resize() {
    capacity *= 2;
    T *temp = new T[capacity];
    for (size_t i = 0; i < size; i++) {
      temp[i] = arr[i];
    }
    delete[] arr;
    arr = temp;
  }

  __host__ __device__ void add(T value) {
    if (size >= capacity) {
      resize();
    }
    arr[size++] = value;
  }

  // This function takes an index and adds the value at that index and moves all
  // the values after it to the right
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

  __host__ __device__ void insert2(size_t index, T value) {
    if (index > size) { // Allow inserting at the end
      printf("Index out of bounds: insert operation");
    }
    if (size >= capacity) {
      resize();
    }
    // Shift elements to the right starting from the end
    for (size_t i = size; i > index; i--) {
      arr[i] = arr[i - 1];
    }
    arr[index] = value;
    size++;
  }

  // NOTE: This function is not efficient, it is only used for simplicity
  __host__ __device__ void remove(size_t index) {
    if (index >= size) {
      printf("Index out of bounds: remove operation");
    }
    for (size_t i = index; i < size - 1; i++) {
      arr[i] = arr[i + 1];
    }
    size--;
  }

  // NOTE: This might not be necessary for the implementation of the algorithm
  __host__ __device__ void remove(size_t index, size_t count) {
    if (index >= size) {
      printf("Index out of bounds: remove multiple operation");
    }
    for (size_t i = index; i < size - count; i++) {
      arr[i] = arr[i + count];
    }
    size -= count;
  }

  // NOTE: if the array contains pointers, we need to delete them and set all
  // the values to nullptr
  __host__ __device__ void clear() { size = 0; }

  __host__ __device__ void set(size_t index, T value) {
    if (index >= size) {
      printf("Index out of bounds: set operation");
    }
    arr[index] = value;
  }

  __host__ __device__ T get(size_t index) const {
    if (index >= size) {
      printf("Index out of bounds: get operation");
    }
    return arr[index];
  }

  __host__ __device__ bool empty() const { return size == 0; }
};

} // namespace VerifyTAPN::Atler

#endif
