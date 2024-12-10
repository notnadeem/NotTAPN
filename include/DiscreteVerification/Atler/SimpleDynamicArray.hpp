#ifndef VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_HPP_
#define VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_HPP_

#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace VerifyTAPN::Atler {

template <typename T> struct SimpleDynamicArray {
  T *arr;
  int size;
  size_t capacity;

  SimpleDynamicArray() : size(0), capacity(1) { arr = new T[capacity]; }

  SimpleDynamicArray(size_t initialCapacity) : size(0) {
    capacity = (initialCapacity == 0) ? 1 : initialCapacity;
    if (capacity == 0)
      capacity = 1;
    arr = new T[capacity];
  }

  SimpleDynamicArray(const SimpleDynamicArray<T> &other) {
    size = other.size;
    capacity = other.capacity;
    arr = new T[capacity];
    for (size_t i = 0; i < size; i++) {
      arr[i] = other.arr[i];
    }
  }

  SimpleDynamicArray(SimpleDynamicArray<T> &&other) noexcept {
    size = other.size;
    capacity = other.capacity;
    arr = other.arr;
    other.arr = nullptr;
    other.size = 0;
    other.capacity = 0;
  }

  ~SimpleDynamicArray() {
    if (arr != nullptr) {
      delete[] arr;
    }
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

  void insert(size_t index, T value) {
    if (index > size) { // Allow inserting at the end
      throw std::out_of_range("Index out of bounds: insert operation");
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

  void remove(size_t index) {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds: remove operation");
    }
    for (size_t i = index; i < size - 1; i++) {
      arr[i] = arr[i + 1];
    }
    size--;
  }

  void clear() { size = 0; }

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
