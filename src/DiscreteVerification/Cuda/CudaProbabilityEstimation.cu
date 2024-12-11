#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"
#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Alloc/RunResultAllocator.cuh"
#include "DiscreteVerification/Alloc/SMCQueryAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include "DiscreteVerification/Cuda/CudaQueryVisitor.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQueryConverter.cuh"
#include "DiscreteVerification/Cuda/CudaTAPNConverter.cuh"
#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
using namespace VerifyTAPN::Cuda;
using namespace VerifyTAPN::Alloc;
// For now single kernel execution per run needed
// Since every run can have different execution time could be nice to try running multiple runs per kernel to improve
// warp utilization

__global__ void runSimulationKernel(Cuda::CudaRunResult *runner, Cuda::AST::CudaSMCQuery *query, int *successCount,
                                    int *runsNeeded, curandState *states, int *rand_seed, int *timeBound,
                                    int *stepBound) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  if (tid >= 1024) return;

  curand_init(*rand_seed, tid, 0, &states[tid]);

  // Copy global state to local memory for faster access
  curandState local_r_state = states[tid];

  if (tid % 1000 == 0) {
    printf("Thread %d initialized\n", tid);
  }

  CudaSMCQuery lQuery = *query;
  CudaRunResult *lRunner = new CudaRunResult(*runner, &local_r_state);

  // TODO prepare per thread
  lRunner->prepare(&local_r_state);

  int lTimeBound = *timeBound;
  int lStepBound = *stepBound;

  // while (!lRunner->maximal && !(lRunner->totalTime >= lTimeBound || lRunner->totalSteps >= lStepBound)) {
  //   Cuda::CudaRealMarking *child = lRunner->realMarking;
  //   Cuda::CudaQueryVisitor checker(*child, *lRunner->tapn);
  //   Cuda::AST::BoolResult result;

  //   lQuery.accept(checker, result);

  //   if (result.value) {
  //     atomicAdd(successCount, 1);
  //     break;
  //   }

  //   lRunner->next(&local_r_state);
  // }
}

bool AtlerProbabilityEstimation::runCuda() {
  std::cout << "Converting TAPN and marking..." << std::endl;
  auto result = VerifyTAPN::Cuda::CudaTAPNConverter::convert(tapn, initialMarking);
  VerifyTAPN::Cuda::CudaTimedArcPetriNet ctapn = result->first;
  VerifyTAPN::Cuda::CudaRealMarking *cipMarking = result->second;

  std::cout << "Converting Query..." << std::endl;
  SMCQuery *currentSMCQuery = static_cast<SMCQuery *>(query);
  VerifyTAPN::Cuda::AST::CudaSMCQuery *cudaSMCQuery = VerifyTAPN::Cuda::CudaSMCQueryConverter::convert(currentSMCQuery);

  // std::cout << "Converting Options..." << std::endl;
  // VerifyTAPN::Cuda::CudaVerificationOptions cudaOptions = Cuda::CudaOptionsConverter::convert(options);

  // TODO: Convert the PlaceVisitor to a simple representation
  // NOTE: Also find a way to simplify the representation of the PlaceVisitor

  std::cout << "Creating run generator..." << std::endl;

  const unsigned int threadsPerBlock = 256;

  // Calculate the number of blocks needed
  unsigned int blocks = (this->runsNeeded + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Runs needed..." << this->runsNeeded << std::endl;
  std::cout << "Threads per block..." << threadsPerBlock << std::endl;
  std::cout << "Blocks..." << blocks << std::endl;

  CudaTimedArcPetriNet *cptapn = &ctapn;

  auto runres = CudaRunResult(cptapn, cipMarking);

  auto runner = new CudaRunResult(cptapn, cipMarking);

  // Allocate the run result

  RunResultAllocator allocator;

  auto allocResult = allocator.allocate(runner, cipMarking, blocks, threadsPerBlock);

  // Allocate the query
  SMCQueryAllocator queryAllocator;
  std::cout << "Allocating query" << std::endl;
  CudaSMCQuery *d_cudaSMCQuery = queryAllocator.allocate(cudaSMCQuery);
  std::cout << "Query Allocation done" << std::endl;

  int successCountVal = 0;
  int *successCount;
  cudaMalloc(&successCount, sizeof(int));
  cudaMemcpy(successCount, &successCountVal, sizeof(int), cudaMemcpyHostToDevice);

  int *runsNeeded;
  cudaMalloc(&runsNeeded, sizeof(int));
  cudaMemcpy(runsNeeded, &this->runsNeeded, sizeof(int), cudaMemcpyHostToDevice);

  int *timeBound;
  cudaMalloc(&timeBound, sizeof(int));
  cudaMemcpy(timeBound, &cudaSMCQuery->smcSettings.timeBound, sizeof(int), cudaMemcpyHostToDevice);

  int *stepBound;
  cudaMalloc(&stepBound, sizeof(int));
  cudaMemcpy(stepBound, &cudaSMCQuery->smcSettings.stepBound, sizeof(int), cudaMemcpyHostToDevice);

  int rand_seed_val = 12345;
  int *rand_seed;
  cudaMalloc(&rand_seed, sizeof(int));
  cudaMemcpy(rand_seed, &rand_seed_val, sizeof(int), cudaMemcpyHostToDevice);

  CudaRunResult *runResultDevice = allocResult->first;
  CudaRealMarking *realMarkingDevice = allocResult->second;

  cudaDeviceSetLimit(cudaLimitStackSize, 6 * 1024);

  // testAllocationKernel<<<1, 1>>>(runResultDevice, realMarkingDevice, &this->runsNeeded);

  // Allocate device memory for rngStates
  curandState *rngStates;
  cudaMalloc(&rngStates, this->runsNeeded * sizeof(curandState_t));

  VerifyTAPN::DiscreteVerification::runSimulationKernel<<<4, threadsPerBlock>>>(
      runResultDevice, d_cudaSMCQuery, successCount, runsNeeded, rngStates, rand_seed, timeBound, stepBound);

  cudaDeviceSynchronize();

  int successCountHost;
  cudaMemcpy(&successCountHost, successCount, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Success count: %d\n", successCountHost);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // testCudaSMCQueryAllocationKernel<<<1, 1>>>(d_cudaSMCQuery);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // cudaError_t allocStatus = cudaGetLastError();
  // if (allocStatus != cudaSuccess) {
  //   std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatus) << std::endl;
  // } else {
  //   std::cout << "Device memory for curand allocated successfully." << std::endl;
  // }

  std::cout << "Kernel execution completed successfully." << std::endl;

  return false;
}
} // namespace VerifyTAPN::DiscreteVerification