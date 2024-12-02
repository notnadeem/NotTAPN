
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Alloc/RunResultAllocator.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRunResult* allocate(CudaRunResult *runResultHost, int blocks, int threadsPerBlock) {
  int numThreads = blocks * threadsPerBlock;

  // Allocate device memory for rngStates
  cudaMalloc(&(runResultHost->rngStates), numThreads * sizeof(curandState_t));

  // Allocate device memory for CudaRunResult
  CudaRunResult *runResultDevice;
  cudaMalloc(&runResultDevice, sizeof(CudaRunResult));

  // Copy CudaRunResult from host to device
  cudaMemcpy(runResultDevice, runResultHost, sizeof(CudaRunResult), cudaMemcpyHostToDevice);

  return runResultDevice;
}

__host__ void allocatePointerMembers(CudaRunResult *runResultHost) {
  // Allocate single oduble for dates_sampled which is an pointer to dynamic array of doubles 
  /* Look into this later */
  cudaMalloc(&runResultHost->dates_sampled, sizeof(double));

  // Allocate marking for parent
  

  // Allocate marking for origin




  // Allocate array of places for markings
}

} // namespace VerifyTAPN::Alloc