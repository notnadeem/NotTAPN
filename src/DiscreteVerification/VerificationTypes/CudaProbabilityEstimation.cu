#include "DiscreteVerification/Cuda/CudaQueryVisitor.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQueryConverter.cuh"
#include "DiscreteVerification/Cuda/CudaTAPNConverter.cuh"
#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
__global__ void runSimulationKernel(Cuda::CudaTimedArcPetriNet *ctapn,
                                    Cuda::CudaRealMarking *initialMarking,
                                    Cuda::AST::CudaSMCQuery *query,
                                    Cuda::CudaRunResult *origRunner,
                                    int *timeBound,
                                    int *stepBound,
                                    int *successCount,
                                    int *runsNeeded) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  int tBound = *timeBound;
  int sBound = *stepBound;
  if (tid >= runNeed) return;
  // TODO Change to cudarunsresult
  // Create local copy of the runner
  Cuda::CudaRunResult runner = *origRunner;
  // This error will disapear once CudaRunResult is implemented
  Cuda::CudaRealMarking *newMarking = runner.parent;

  while (!runner.maximal && !(runner.totalTime >= tBound || runner.totalSteps >= sBound)) {

    Cuda::CudaRealMarking child = *newMarking->clone();
    Cuda::CudaQueryVisitor checker(child, *ctapn);
    Atler::AST::BoolResult result;

    query->accept(checker, result);

    if (result.value) {
      atomicAdd(successCount, 1);
      break;
    }
    // This error will disapear once CudaRunResult is implemented
    newMarking = runner.next(tid);
  }
}

bool AtlerProbabilityEstimation::run() {
  std::cout << "Converting TAPN and marking..." << std::endl;
  auto result = VerifyTAPN::Cuda::CudaTAPNConverter::convert(tapn, initialMarking);
  VerifyTAPN::Cuda::CudaTimedArcPetriNet stapn = result->first;
  VerifyTAPN::Cuda::CudaRealMarking ciMarking = result->second;

  std::cout << "Converting Query..." << std::endl;
  SMCQuery *currentSMCQuery = static_cast<SMCQuery *>(query);
  VerifyTAPN::Cuda::AST::CudaSMCQuery *cudaSMCQuery = VerifyTAPN::Cuda::CudaSMCQueryConverter::convert(currentSMCQuery);

  // std::cout << "Converting Options..." << std::endl;
  // VerifyTAPN::Cuda::CudaVerificationOptions cudaOptions = Cuda::CudaOptionsConverter::convert(options);

  // TODO: Convert the PlaceVisitor to a simple representation
  // NOTE: Also find a way to simplify the representation of the PlaceVisitor

  // Simulate prepare func
  // setup the run generator

  std::cout << "Creating run generator..." << std::endl;
  // TODO

  int blocks = 1;
  int threads = 1;
  auto runres = VerifyTAPN::Cuda::CudaRunResult(stapn, blocks, threads);

  std::cout << "Run prepare" << std::endl;

  // Allocate

  // VerifyTAPN::DiscreteVerification::runSimulationKernel<<<blocks, threads>>>(
  //     stapn, ciMarking, cudaSMCQuery, runres, smcSettings.timeBound, smcSettings.stepBound, 0, runsNeeded);

  return false;
}
} // namespace VerifyTAPN::DiscreteVerification