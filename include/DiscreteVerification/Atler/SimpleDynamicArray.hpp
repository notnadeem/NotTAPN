#ifndef VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_HPP_
#define VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_HPP_

#include <cstddef>
#include <stdexcept>

namespace VerifyTAPN::Atler {

template <typename T> struct SimpleDynamicArray {
  T *arr;
  int size;
  size_t capacity;

  SimpleDynamicArray() : size(0), capacity(1) { arr = new T[capacity]; }

  SimpleDynamicArray(size_t initialCapacity) : size(0) {
      capacity = (initialCapacity == 0) ? 1 : initialCapacity;
      if(capacity == 0) capacity = 1;
      arr = new T[capacity];
  }

  void resize() {
    capacity *= 2;
    T *temp = new T[capacity];
    for (size_t i = 0; i < size; i++) {
      temp[i] = arr[i];
    }
    delete[] arr;
    arr = temp;
  }

  void add(T value) {
    if (size >= capacity) {
      resize();
    }
    arr[size++] = value;
  }

  // NOTE: This function is not efficient, it is only used for simplicity
  void remove(size_t index) {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds: remove operation");
    }
    for (size_t i = index; i < size - 1; i++) {
      arr[i] = arr[i + 1];
    }
    size--;
  }

  // NOTE: This might not be necessary for the implementation of the algorithm
  void remove(size_t index, size_t count) {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds: remove multiple operation");
    }
    for (size_t i = index; i < size - count; i++) {
      arr[i] = arr[i + count];
    }
    size -= count;
  }

  // NOTE: if the array contains pointers, we need to delete them and set all the values to nullptr
  void clear() {
    size = 0;
  }

  void set(size_t index, T value) {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds: set operation");
    }
    arr[index] = value;
  }

  T get(size_t index) const {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds: get operation");
    }
    return arr[index];
  }

  bool empty() const { return size == 0; }
};

} // namespace VerifyTAPN::Atler

#endif
