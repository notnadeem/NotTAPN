#ifndef VERIFYTAPN_ATLER_SIMPLEINTERVAL_CUH_
#define VERIFYTAPN_ATLER_SIMPLEINTERVAL_CUH_

#include "DiscreteVerification/Cuda/SimpleDynamicArray.cuh"
#include <cuda_runtime.h>
#include <math.h>

// TODO: Change numeric_limits to the cuda/thrust ones

namespace VerifyTAPN {
namespace Cuda {
namespace Util {

struct SimpleInterval {
  double low, high;

  __host__ __device__ static double boundUp() { return HUGE_VAL; }

  __host__ __device__ static double boundDown() { return -HUGE_VAL; }

  __host__ __device__ static double epsilon() { return __DBL_EPSILON__; }

  __host__ __device__ SimpleInterval() : low(0), high(0) {}
  __host__ __device__ SimpleInterval(double l, double h) : low(l), high(h) {
    if (low > high) {
      low = boundUp();
      high = boundDown();
    }
  };

  __host__ __device__ auto upper() const { return high; }

  __host__ __device__ auto lower() const { return low; }

  __host__ __device__ auto empty() const { return high < low; }

  __host__ __device__ auto length() const {
    if (low == boundDown() || high == boundUp()) {
      return boundUp();
    }
    if (empty()) return (double)0;
    return high - low;
  }

  __host__ __device__ void delta(double dx) {
    if (empty()) return;
    if (low != boundDown()) {
      low += dx;
    }
    if (high != boundUp()) {
      high += dx;
    }
  }

  __host__ __device__ SimpleInterval positive() {
    if (empty() || high < 0) return SimpleInterval((double)1, 0);
    return SimpleInterval(max(low, (double)0), high);
  }
};

__host__ __device__ SimpleInterval intersect(const SimpleInterval &l, const SimpleInterval r) {
  if (l.empty()) return l;
  if (r.empty()) return r;
  SimpleInterval n(max(l.low, r.low), min(l.high, r.high));
  return n;
}

__host__ __device__ SimpleInterval hull(const SimpleInterval &l, const SimpleInterval r) {
  return SimpleInterval(min(l.low, r.low), max(l.high, r.high));
}

__host__ __device__ bool overlap(const SimpleInterval &l, const SimpleInterval r) {
  auto i = intersect(l, r);
  return !i.empty();
}

// Fix both setAdd and setIntersection later
__host__ __device__ void setAdd(SimpleDynamicArray<SimpleInterval> &first, const SimpleInterval &element) {
  for (unsigned int i = 0; i < first.size; i++) {

    if (element.upper() < first.get(i).lower()) {
      // Add element
      first.set(i, element);
      return;
    } else if (overlap(element, first.get(i))) {
      SimpleInterval u = hull(element, first.get(i));
      // Merge with node
      first.set(i, u);
      // Clean up
      for (i++; i < first.size; i++) {
        if (first.get(i).lower() <= u.upper()) {
          // Merge items
          if (first.get(i).upper() > u.upper()) {
            first.set(i - 1, SimpleInterval(first.get(i - 1).lower(), first.get(i).upper()));
          }
          first.remove(i);
          i--;
        }
      }
      return;
    }
  }
  first.add(element);
}

__host__ __device__ SimpleDynamicArray<SimpleInterval> setIntersection(SimpleDynamicArray<SimpleInterval> first,
                                                                       SimpleDynamicArray<SimpleInterval> second) {
  SimpleDynamicArray<SimpleInterval> result;

  if (first.empty() || second.empty()) {
    return result;
  }

  unsigned int i = 0, j = 0;

  while (i < first.size && j < second.size) {
    double i1up = first.get(i).upper();
    double i2up = second.get(j).upper();

    SimpleInterval intersection = intersect(first.get(i), second.get(j));

    if (!intersection.empty()) {
      result.add(intersection);
    }

    if (i1up <= i2up) {
      i++;
    }

    if (i2up <= i1up) {
      j++;
    }
  }
  return result;
}

} /* namespace Util */
} // namespace Cuda
} /* namespace VerifyTAPN */
#endif /* SIMPLEINTERVAL_CUH_ */