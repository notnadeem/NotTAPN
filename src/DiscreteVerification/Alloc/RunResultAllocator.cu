
#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Alloc/RunResultAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRunResult *allocate(CudaRunResult *h_run_result, int blocks, int threadsPerBlock, CudaTimedArcPetriNet *d_tapn,
                                 std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
                                 std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) {
  int numThreads = blocks * threadsPerBlock;

  // Allocate device memory for rngStates
  cudaMalloc(&(h_run_result->rngStates), numThreads * sizeof(curandState_t));

  // Allocate device memory for CudaRunResult
  CudaRunResult *runResultDevice;
  cudaMalloc(&runResultDevice, sizeof(CudaRunResult));

  CudaRunResult *temp_run_result = (CudaRunResult *)malloc(sizeof(CudaRunResult *));

  cudaMalloc(&d_tapn, sizeof(CudaTimedArcPetriNet));

  temp_run_result->maximal = h_run_result->maximal;
  temp_run_result->lastDelay = h_run_result->lastDelay;
  temp_run_result->totalTime = h_run_result->totalTime;
  temp_run_result->totalSteps = h_run_result->totalSteps;
  temp_run_result->numericPrecision = h_run_result->numericPrecision;

  RealMarkingAllocator realMarkingAllocator;

  temp_run_result->parent = realMarkingAllocator.allocate(h_run_result->parent, transition_map, place_map);
  temp_run_result->origin = realMarkingAllocator.allocate(h_run_result->origin, transition_map, place_map);


  temp_run_result->defaultTransitionIntervals = h_run_result->defaultTransitionIntervals;
  temp_run_result->transitionIntervals = h_run_result->transitionIntervals;

  // Copy CudaRunResult from host to device
  cudaMemcpy(runResultDevice, temp_run_result, sizeof(CudaRunResult), cudaMemcpyHostToDevice);

  cudaFree(temp_run_result);
  return runResultDevice;
}

} // namespace VerifyTAPN::Alloc