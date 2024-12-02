#ifndef REALMARKINGALLOCATOR_CUH_
#define REALMARKINGALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {
using namespace Cuda;

struct RealMarkingAllocator {
  __host__ static CudaRealMarking *
  allocate(CudaRealMarking *h_real_marking,
           std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
           std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) {};

  __host__ static CudaRealPlace **
  cuda_allocate_places_for_marking(CudaRealMarking *h_marking,
                                   std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) {};

private:
  __host__ static void allocatePointerMembers(CudaRealMarking *realMarkingHost) {};
};
} // namespace VerifyTAPN::Alloc

#endif /* REALMARKINGALLOCATOR_CUH_ */