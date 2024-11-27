// Implementation of a Cuda double-ended queue
// Make it at some point in time
// NOTE: for the AtlerRunResult clas

#ifndef VERIFICATION_ATLER_CUDA_DEQUE_CUH
#define VERIFICATION_ATLER_CUDA_DEQUE_CUH

#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

template <typename T>
struct CudaDeque {
    struct Node {
        T data;
        Node* prev;
        Node* next;

        __host__ __device__ Node(const T& value) : data(value), prev(nullptr), next(nullptr) {}
    };

    Node* front;
    Node* back;
    size_t size;

    // Constructor
    __host__ __device__ CudaDeque() : front(nullptr), back(nullptr), size(0) {}

    // Destructor
    __host__ __device__ ~CudaDeque() {
        while (front != nullptr) {
            Node* temp = front;
            front = front->next;
            delete temp;
        }
    }

    // Add element to front
    __host__ __device__ void push_front(const T& value) {
        Node* newNode = new Node(value);
        if (empty()) {
            front = back = newNode;
        } else {
            newNode->next = front;
            front->prev = newNode;
            front = newNode;
        }
        size++;
    }

    // Add element to back
    __host__ __device__ void push_back(const T& value) {
        Node* newNode = new Node(value);
        if (empty()) {
            front = back = newNode;
        } else {
            newNode->prev = back;
            back->next = newNode;
            back = newNode;
        }
        size++;
    }

    // Remove element from front
    __host__ __device__ void pop_front() {
        if (empty()) {
            printf("Deque is empty\n");
            return;
        }

        Node* temp = front;
        front = front->next;

        if (front == nullptr) {
            back = nullptr;
        } else {
            front->prev = nullptr;
        }

        delete temp;
        size--;
    }

    // Remove element from back
    __host__ __device__ void pop_back() {
        if (empty()) {
            printf("Deque is empty\n");
            return;
        }

        Node* temp = back;
        back = back->prev;

        if (back == nullptr) {
            front = nullptr;
        } else {
            back->next = nullptr;
        }

        delete temp;
        size--;
    }

    // Get front element
    __host__ __device__ T& peek_front() {
        if (empty()) {
            printf("Deque is empty\n");
            static T default_value{};
            return default_value;
        }
        return front->data;
    }

    // Get back element
    __host__ __device__ T& peek_back() {
        if (empty()) {
            printf("Deque is empty\n");
            static T default_value{};
            return default_value;
        }
        return back->data;
    }

    // Access element by index
    __host__ __device__ T& at(size_t index) {
        if (index >= size) {
            printf("Index out of bounds\n");
            static T default_value{};
            return default_value;
        }

        // Optimize traversal by choosing the closest end
        Node* current;
        if (index <= size / 2) {
            // Traverse from front
            current = front;
            for (size_t i = 0; i < index; i++) {
                current = current->next;
            }
        } else {
            // Traverse from back
            current = back;
            for (size_t i = size - 1; i > index; i--) {
                current = current->prev;
            }
        }
        return current->data;
    }

    // Operator overload for array-style access
    __host__ __device__ T& operator[](size_t index) {
        return at(index);
    }

    // Check if deque is empty
    __host__ __device__ bool empty() const {
        return size == 0;
    }

    // Get size of deque
    __host__ __device__ size_t get_size() const {
        return size;
    }
};

} // namespace Cuda
} // namespace VerifyTAPN

#endif
