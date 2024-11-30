#ifndef VERIFYTAPN_ATLER_CUDATIMEINTERVAL_CUH_
#define VERIFYTAPN_ATLER_CUDATIMEINTERVAL_CUH_

#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimeInterval {
  bool isLowerBoundStrict;
  int lowerBound;
  int upperBound;
  bool isUpperBoundStrict;

 __host__ __device__ static double epsilon() { return __DBL_EPSILON__; }

  __host__ __device__ inline bool setUpperBound(int bound, bool isStrict) {
    if (upperBound == bound)
      isUpperBoundStrict |= isStrict;
    else if (upperBound > bound) {
      isUpperBoundStrict = isStrict;
      upperBound = bound;
    }
    if (upperBound < lowerBound)
      return false;
    else
      return true;
  }

  __host__ __device__ inline bool contains(double number) const {
      return (number >= lowerBound && number <= upperBound) ||
            (fabs(number - lowerBound) <= epsilon() * 4) ||
            (fabs(number - upperBound) <= epsilon() * 4);
  }
};

} // namespace Cuda
} // namespace VerifyTAPN

#endif
