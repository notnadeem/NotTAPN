#include "DiscreteVerification/Atler/AtlerRunResult.hpp"
#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Cuda/CudaDynamicArray.cuh"
#include "DiscreteVerification/Atler/SimpleInterval.hpp"
#include "DiscreteVerification/Atler/SimpleOptionsConverter.hpp"
#include "DiscreteVerification/Cuda/CudaQueryVisitor.cuh"
#include "DiscreteVerification/Atler/SimpleSMCQueryConverter.hpp"
#include "DiscreteVerification/Atler/SimpleTAPNConverter.hpp"
#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"
#include "DiscreteVerification/Cuda/CudaTimedArcPetriNet.cuh"

#include "DiscreteVerification/Atler/SimpleVerificationOptions.hpp"
#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {
__global__ void runSimulationKernel(Cuda::CudaTimedArcPetriNet *stapn, Cuda::CudaRealMarking *initialMarking,
                                    Cuda::AST::CudaSMCQuery *query,
                                    Cuda::CudaDynamicArray<Atler::AtlerRunResult *> *clones, int *timeBound,
                                    int *stepBound, int *successCount, int *runsNeeded) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int runNeed = *runsNeeded;
  int tBound = *timeBound;
  int sBound = *stepBound;
  if (tid >= runNeed) return;
  // TODO Change to cudarunsresult
  Atler::AtlerRunResult *origRunner = clones->get(tid);
  // Create local copy of the runner
  Atler::AtlerRunResult runner = *origRunner;
  //This error will disapear once CudaRunResult is implemented
  Cuda::CudaRealMarking *newMarking = runner.parent;

  while (!runner.maximal && !(runner.totalTime >= tBound || runner.totalSteps >= sBound)) {

    Cuda::CudaRealMarking child = *newMarking->clone();
    Cuda::CudaQueryVisitor checker(child, *stapn);
    Atler::AST::BoolResult result;

    query->accept(checker, result);

    if (result.value) {
      atomicAdd(successCount, 1);
      break;
    }
    //This error will disapear once CudaRunResult is implemented
    newMarking = runner.next();
  }

} // namespace Cuda
} // namespace VerifyTAPN::DiscreteVerification