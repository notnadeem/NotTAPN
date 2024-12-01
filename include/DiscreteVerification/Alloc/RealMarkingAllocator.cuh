#ifndef REALMARKINGALLOCATOR_CUH_
#define REALMARKINGALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {
using namespace Cuda;

struct RealMarkingAllocator {
  __host__ static CudaRealMarking *allocate(CudaRealMarking *realMarkingHost) {};

private:
  __host__ static void allocatePointerMembers(CudaRealMarking *realMarkingHost) {};
};
} // namespace VerifyTAPN::Alloc

#endif /* REALMARKINGALLOCATOR_CUH_ */