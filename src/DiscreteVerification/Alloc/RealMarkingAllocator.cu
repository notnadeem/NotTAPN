#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRealMarking *RealMarkingAllocator::allocate(CudaRealMarking *realMarkingHost) {};

__host__ void RealMarkingAllocator::allocatePointerMembers(CudaRealMarking *realMarkingHost) {};
} // namespace VerifyTAPN::Alloc