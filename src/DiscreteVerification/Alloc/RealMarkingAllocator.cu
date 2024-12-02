#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRealMarking* allocate(CudaRealMarking *realMarkingHost) {};

__host__ void allocatePointerMembers(CudaRealMarking *realMarkingHost) {};
} // namespace VerifyTAPN::Alloc