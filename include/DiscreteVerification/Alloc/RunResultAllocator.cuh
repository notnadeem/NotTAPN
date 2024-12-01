#ifndef RUNRESULTALLOCATOR_CUH_
#define RUNRESULTALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>
namespace VerifyTAPN::Alloc {

using namespace Cuda;

struct RunResultAllocator {
  __host__ static CudaRunResult *allocate(CudaRunResult *runResultHost, int blocks, int threadsPerBlock) {};

private:
  __host__ static void allocatePointerMembers(CudaRunResult *runResultHost) {};
};
} // namespace VerifyTAPN::Alloc

#endif /* RUNRESULTALLOCATOR_CUH_ */