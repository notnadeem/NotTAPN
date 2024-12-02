#ifndef REALPLACEALLOCATOR_CUH_
#define REALPLACEALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {
using namespace Cuda;

struct RealPlaceAllocator {
  __host__ static CudaRealPlace **allocate(CudaRealPlace *realPlaceHost, CudaTimedPlace placeHost) {};

private:
  __host__ static void allocatePointerMembers(CudaRealPlace *realPlaceHost) {};
};
} // namespace VerifyTAPN::Alloc

#endif /* REALPLACEALLOCATOR_CUH_ */