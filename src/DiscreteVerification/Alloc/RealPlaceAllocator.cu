#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"
#include "DiscreteVerification/Alloc/RealPlaceAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRealPlace **allocate(CudaTimedArcPetriNet *h_petrinet, CudaRealMarking *h_marking,
                                  std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
                                   std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) {

  CudaRealPlace *d_real_place;
  cudaMalloc(&d_real_place, sizeof(CudaRealPlace *));

  // For all places in the h_marking, allocate a new CudaRealPlace and token array
  // In order to allocate CudaRealPlace, we need to allocate CudaTimedPlace

  /*!For now not allocating tokes since this is already done in converter because of CudaDynamicArray managed mem*/
}

__host__ void allocatePointerMembers(CudaRealPlace *realMarkingHost) {};
} // namespace VerifyTAPN::Alloc