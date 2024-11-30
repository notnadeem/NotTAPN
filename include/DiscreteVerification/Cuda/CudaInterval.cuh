#ifndef VERIFYTAPN_ATLER_CUDAINTERVAL_CUH_
#define VERIFYTAPN_ATLER_CUDAINTERVAL_CUH_

#include "DiscreteVerification/Cuda/CudaDynamicArray.cuh"
#include <cuda_runtime.h>
#include <math.h>

// TODO: Change numeric_limits to the cuda/thrust ones

namespace VerifyTAPN {
namespace Cuda {
namespace Util {

struct CudaInterval {
  double low, high;

  __host__ __device__ static double boundUp() { return HUGE_VAL; }

  __host__ __device__ static double boundDown() { return -HUGE_VAL; }

  __device__ static double epsilon() { return __DBL_EPSILON__; }

  __host__ __device__ CudaInterval() : low(0), high(0) {}
  __host__ __device__ CudaInterval(double l, double h) : low(l), high(h) {
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

  __host__ __device__ CudaInterval positive() {
    if (empty() || high < 0) return CudaInterval((double)1, 0);
    return CudaInterval(fmax(low, (double)0), high);
  }
};

__host__ __device__ CudaInterval intersect(const CudaInterval &l, const CudaInterval r) {
  if (l.empty()) return l;
  if (r.empty()) return r;
  CudaInterval n(fmax(l.low, r.low), fmin(l.high, r.high));
  return n;
}

__host__ __device__ CudaInterval hull(const CudaInterval &l, const CudaInterval r) {
  return CudaInterval(fmin(l.low, r.low), fmax(l.high, r.high));
}

__host__ __device__ bool overlap(const CudaInterval &l, const CudaInterval r) {
  auto i = intersect(l, r);
  return !i.empty();
}

// Fix both setAdd and setIntersection later
__host__ __device__ void setAdd(CudaDynamicArray<CudaInterval> &first, const CudaInterval &element) {
  for (unsigned int i = 0; i < first.size; i++) {

    if (element.upper() < first.get(i).lower()) {
      // Add element
      first.insert(i, element);
      return;
    } else if (overlap(element, first.get(i))) {
      CudaInterval u = hull(element, first.get(i));
      // Merge with node
      first.set(i, u);
      // Clean up
      for (i++; i < first.size; i++) {
        if (first.get(i).lower() <= u.upper()) {
          // Merge items
          if (first.get(i).upper() > u.upper()) {
            first.set(i - 1, CudaInterval(first.get(i - 1).lower(), first.get(i).upper()));
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

__host__ __device__ CudaDynamicArray<CudaInterval> setIntersection(CudaDynamicArray<CudaInterval> first,
                                                                       CudaDynamicArray<CudaInterval> second) {
  CudaDynamicArray<CudaInterval> result;

  if (first.empty() || second.empty()) {
    return result;
  }

  unsigned int i = 0, j = 0;

  while (i < first.size && j < second.size) {
    double i1up = first.get(i).upper();
    double i2up = second.get(j).upper();

    CudaInterval intersection = intersect(first.get(i), second.get(j));

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
#endif /* CUDAINTERVAL_CUH_ */