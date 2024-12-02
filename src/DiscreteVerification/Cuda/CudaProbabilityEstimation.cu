#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include "DiscreteVerification/Cuda/CudaQueryVisitor.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQueryConverter.cuh"
#include "DiscreteVerification/Cuda/CudaTAPNConverter.cuh"
#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
__global__ void runSimulationKernel(Cuda::CudaTimedArcPetriNet *ctapn, Cuda::CudaRealMarking *initialMarking,
                                    Cuda::AST::CudaSMCQuery *query, Cuda::CudaRunResult *runner, int *timeBound,
                                    int *stepBound, int *successCount, int *runsNeeded, curandState *states) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  if (tid >= runNeed) return;

  curand_init(clock64(), tid, 0, &states[tid]);
  if (tid % 1000 == 0) {
    printf("Thread %d initialized\n", tid);
  }

  int tBound = *timeBound;
  int sBound = *stepBound;

  Cuda::CudaTimedArcPetriNet tapn = *ctapn;

  // TODO prepare per thread
  // runner.prepare(initialMarking);
  Cuda::CudaRealMarking *newMarking = runner->parent;

  while (!runner->maximal && !(runner->totalTime >= tBound || runner->totalSteps >= sBound)) {

    Cuda::CudaRealMarking *child = newMarking->clone();
    Cuda::CudaQueryVisitor checker(*child, tapn);
    Cuda::AST::BoolResult result;

    query->accept(checker, result);

    if (result.value) {
      atomicAdd(successCount, 1);
      break;
    }
    newMarking = runner->next(tid);
  }
}

__global__ void testAllocationKernel(VerifyTAPN::Cuda::CudaTimedArcPetriNet* pn){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(tid == 0){
  printf("Petri net maxConstant: %s\n", pn->maxConstant);
  printf("Petri net placesLength: %d\n", pn->placesLength);
  for(int i = 0; i < pn->placesLength; i++){
    printf("Place id: %d\n", pn->places[i]->id);
    printf("Place name: %s\n", pn->places[i]->name);
    printf("Place invariant: %s\n", pn->places[i]->timeInvariant);
    printf("Place type: %s\n", pn->places[i]->type);
    printf("Place containsInhibitorArcs: %s\n", pn->places[i]->containsInhibitorArcs);
    printf("Place inputArcsLength: %d\n", pn->places[i]->inputArcsLength);
    for(int x = 0; x < pn->places[i]->inputArcsLength; x++){
      printf("Place inputArcs[%d]: %s\n", x, pn->places[i]->inputArcs[x]->weight);
    }
  }
  printf("Kernel executed\n");
  }
};

bool AtlerProbabilityEstimation::runCuda() {
  std::cout << "Converting TAPN and marking..." << std::endl;
  auto result = VerifyTAPN::Cuda::CudaTAPNConverter::convert(tapn, initialMarking);
  VerifyTAPN::Cuda::CudaTimedArcPetriNet ctapn = result->first;
  VerifyTAPN::Cuda::CudaRealMarking ciMarking = result->second;

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

  // Allocate the petry net

  VerifyTAPN::Alloc::CudaPetriNetAllocator pnAllocator;

  VerifyTAPN::Cuda::CudaTimedArcPetriNet* pn = pnAllocator.cuda_allocator(&ctapn);

  testAllocationKernel<<<blocks, threadsPerBlock>>>(pn);
  // Allocate the initial marking

  //Allocate the query

  cudaError_t allocStatus = cudaGetLastError();
  if (allocStatus != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(allocStatus) << std::endl;
  } else {
    std::cout << "Device memory for curand allocated successfully." << std::endl;
  }

  auto runres = new VerifyTAPN::Cuda::CudaRunResult(ctapn);

  std::cout << "Run prepare" << std::endl;

  // VerifyTAPN::DiscreteVerification::runSimulationKernel<<<blocks, threads>>>(
  //     stapn, ciMarking, cudaSMCQuery, runres, smcSettings.timeBound, smcSettings.stepBound, 0, runsNeeded);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  err = cudaDeviceSynchronize();

  if (err != cudaSuccess) {
    std::cerr << "CUDA device synchronization failed: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  std::cout << "Kernel execution completed successfully." << std::endl;

  return false;
}
} // namespace VerifyTAPN::DiscreteVerification