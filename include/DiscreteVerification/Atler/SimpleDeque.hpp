// Implementation of a simple double-ended queue
// Make it at some point in time
// NOTE: for the AtlerRunResult clas

#ifndef VERIFICATION_ATLER_SIMPLE_DEQUE_HPP
#define VERIFICATION_ATLER_SIMPLE_DEQUE_HPP

#include <stdexcept>

namespace VerifyTAPN {
namespace Atler {


template <typename T>
struct SimpleDeque {
    struct Node {
        T data;
        Node* prev;
        Node* next;

        Node(const T& value) : data(value), prev(nullptr), next(nullptr) {}
    };

    Node* front;
    Node* back;
    size_t size;

    // Constructor
    SimpleDeque() : front(nullptr), back(nullptr), size(0) {}

    // Destructor
    ~SimpleDeque() {
        while (front != nullptr) {
            Node* temp = front;
            front = front->next;
            delete temp;
        }
    }

    // Add element to front
    void push_front(const T& value) {
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
    void push_back(const T& value) {
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
    void pop_front() {
        if (empty()) {
            throw std::runtime_error("Deque is empty");
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
    void pop_back() {
        if (empty()) {
            throw std::runtime_error("Deque is empty");
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
    T& peek_front() {
        if (empty()) {
            throw std::runtime_error("Deque is empty");
        }
        return front->data;
    }

    // Get back element
    T& peek_back() {
        if (empty()) {
            throw std::runtime_error("Deque is empty");
        }
        return back->data;
    }

    // Access element by index
    T& at(size_t index) {
        if (index >= size) {
            throw std::out_of_range("Index out of bounds");
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
    T& operator[](size_t index) {
        return at(index);
    }

    // Check if deque is empty
    bool empty() const {
        return size == 0;
    }

    // Get size of deque
    size_t get_size() const {
        return size;
    }
};

} // namespace Atler
} // namespace VerifyTAPN

#endif
