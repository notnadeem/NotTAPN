#include "DiscreteVerification/Alloc/RealPlaceAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRealPlace *RealPlaceAllocator::allocate(CudaRealPlace *realMarkingHost, CudaTimedPlace placeHost) {
  //Allocate the place for real marking

};

__host__ void RealPlaceAllocator::allocatePointerMembers(CudaRealPlace *realMarkingHost) {};
} // namespace VerifyTAPN::Alloc