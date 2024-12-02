#include "DiscreteVerification/Alloc/RealPlaceAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRealPlace* allocate(CudaRealPlace *realMarkingHost, CudaTimedPlace placeHost) {
  //Allocate the place for real marking

};

__host__ void allocatePointerMembers(CudaRealPlace *realMarkingHost) {};
} // namespace VerifyTAPN::Alloc