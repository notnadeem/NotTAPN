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

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
using namespace VerifyTAPN::Cuda;
using namespace VerifyTAPN::Alloc;
namespace cg = cooperative_groups;

__global__ void runSimulationKernel(Cuda::CudaRunResult *runner, Cuda::AST::CudaSMCQuery *query, int *successCount,
                                    int *runsNeeded, curandState *states, unsigned long long *rand_seed, int *timeBound,
                                    int *stepBound) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  if (tid >= 0) return;

  curand_init(*rand_seed, tid, 0, &states[tid]);

  // Copy global state to local memory for faster access
  curandState localState = states[tid];

  CudaSMCQuery lQuery = *query;
  CudaRunResult lRunner(*runner, &localState);

  // TODO prepare per thread
  lRunner.prepare(&localState);

  int lTimeBound = *timeBound;
  int lStepBound = *stepBound;

  while (!lRunner.maximal && !(lRunner.totalTime >= lTimeBound || lRunner.totalSteps >= lStepBound)) {
    Cuda::CudaRealMarking *child = lRunner.realMarking;
    Cuda::CudaQueryVisitor checker(*child, *lRunner.tapn);
    Cuda::AST::BoolResult result;

    lQuery.accept(checker, result);

    if (result.value) {
      atomicAdd(successCount, 1);
      break;
    }

    lRunner.next(&localState);
  }
}

__global__ void runSimulationKernel1(Cuda::CudaRunResult *runner, Cuda::AST::CudaSMCQuery *query, int *successCount,
                                    int *runsNeeded, curandState *states, unsigned long long *rand_seed, int *timeBound,
                                    int *stepBound) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  if (tid >= runNeed) return;

  curand_init(*rand_seed, tid, 0, &states[tid]);

  // Copy global state to local memory for faster access
  curandState localState = states[tid];

  CudaSMCQuery lQuery = *query;
  CudaRunResult lRunner(*runner, &localState);

  // TODO prepare per thread
  lRunner.prepare(&localState);

  int lTimeBound = *timeBound;
  int lStepBound = *stepBound;

  cg::thread_block block = cg::this_thread_block();
  auto tile32 = cg::tiled_partition<32>(block);

  while (tile32.any(!lRunner.maximal && !(lRunner.totalTime >= lTimeBound || lRunner.totalSteps >= lStepBound))) {
    Cuda::CudaRealMarking *child = lRunner.realMarking;
    Cuda::CudaQueryVisitor checker(*child, *lRunner.tapn);
    Cuda::AST::BoolResult result;

    lQuery.accept(checker, result);

    if (result.value) {
      atomicAdd(successCount, 1);
      break;
    }

    lRunner.next(&localState);
  }
}

bool setDeviceHeapSize(double fraction = 1) {
  size_t free_mem, total_mem;
  cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
  if (status != cudaSuccess) {
    std::cerr << "Error retrieving CUDA memory info: " << cudaGetErrorString(status) << std::endl;
    return false;
  }

  size_t desired_heap_size = static_cast<size_t>(free_mem * fraction);

  size_t min_heap_size = 10 * 1024 * 1024;
  size_t max_heap_size = 15ULL * 1024 * 1024 * 1024;
  if (desired_heap_size < min_heap_size) {
    desired_heap_size = min_heap_size;
  } else if (desired_heap_size > max_heap_size) {
    desired_heap_size = max_heap_size;
  }

  status = cudaDeviceSetLimit(cudaLimitMallocHeapSize, desired_heap_size);
  if (status != cudaSuccess) {
    std::cerr << "Error setting CUDA device heap size: " << cudaGetErrorString(status) << std::endl;
    return false;
  }

  std::cout << "Device heap size set to: " << desired_heap_size / (1024 * 1024) << " MB\n";
  return true;
}

bool setDeviceStackSize(size_t stackSizeBytes) {
  cudaError_t status = cudaDeviceSetLimit(cudaLimitStackSize, stackSizeBytes);
  if (status != cudaSuccess) {
    std::cerr << "Error setting device stack size: " << cudaGetErrorString(status) << std::endl;
    return false;
  }
  std::cout << "Device stack size set to: " << stackSizeBytes / 1024 << " KB" << std::endl;
  return true;
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

  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start, 0);

  const unsigned int threadsPerBlock = 256;

  // Calculate the number of blocks needed
  unsigned int blocks = (this->runsNeeded + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Runs needed..." << this->runsNeeded << std::endl;
  std::cout << "Threads per block..." << threadsPerBlock << std::endl;
  std::cout << "Blocks..." << blocks << std::endl;

  CudaTimedArcPetriNet *cptapn = &ctapn;

  auto runres = CudaRunResult(cptapn, cipMarking);

  auto runner = new CudaRunResult(cptapn, cipMarking);

  size_t stackSize = 8 * 1024; // 8 KB
  if (!setDeviceStackSize(stackSize)) {
    std::cerr << "Failed to set device stack size.\n";
    return -1;
  }

  if (!setDeviceHeapSize(1)) {
    std::cerr << "Failed to set device heap size.\n";
    return -1;
  }

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

  unsigned long long rand_seed_val = 9223372036854775807;
  unsigned long long *rand_seed;
  cudaMalloc(&rand_seed, sizeof(unsigned long long));
  cudaMemcpy(rand_seed, &rand_seed_val, sizeof(unsigned long long), cudaMemcpyHostToDevice);

  CudaRunResult *runResultDevice = allocResult->first;
  CudaRealMarking *realMarkingDevice = allocResult->second;

  // Allocate device memory for rngStates
  curandState *rngStates;
  cudaMalloc(&rngStates, this->runsNeeded * sizeof(curandState_t));

  // Launch the kernel
  VerifyTAPN::DiscreteVerification::runSimulationKernel<<<1, 1>>>(
      runResultDevice, d_cudaSMCQuery, successCount, runsNeeded, rngStates, rand_seed, timeBound, stepBound);

  // Record the stop event
  cudaEventRecord(stop, 0);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate the elapsed time in milliseconds
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Output the execution time
  std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
  std::cout << "Kernel execution time: " << (milliseconds / 1000.0f) << " seconds" << std::endl;
  
  int *successCountHost = (int *)malloc(sizeof(int));
  cudaMemcpy(&successCountHost, successCount, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Success count: %d\n", successCountHost);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

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