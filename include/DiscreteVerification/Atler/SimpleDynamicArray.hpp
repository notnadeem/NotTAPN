#ifndef VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_HPP_
#define VERIFYTAPN_ATLER_SIMPLEDYNAMICARRAY_HPP_

#include <cstddef>
#include <stdexcept>

namespace VerifyTAPN::Atler {

template <typename T> struct SimpleDynamicArray {
  T *arr;
  size_t size;
  size_t capacity;

  SimpleDynamicArray() : size(0), capacity(1) { arr = new T[capacity]; }

  SimpleDynamicArray(size_t capacity) : size(0), capacity(capacity) {
    arr = new T[capacity];
  }

  ~SimpleDynamicArray() { delete[] arr; }

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

  void remove(size_t index) {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds");
    }
    for (size_t i = index; i < size - 1; i++) {
      arr[i] = arr[i + 1];
    }
    size--;
  }

  void set(size_t index, T value) {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds");
    }
    arr[index] = value;
  }

  T get(size_t index) const {
    if (index >= size) {
      throw std::out_of_range("Index out of bounds");
    }
    return arr[index];
  }

  bool empty() const { return size == 0; }
};

} // namespace VerifyTAPN::Atler

#endif
