#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"
#include "DiscreteVerification/Atler/AtlerRunResult.hpp"
#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Atler/SimpleDynamicArray.hpp"
#include "DiscreteVerification/Atler/SimpleInterval.hpp"
#include "DiscreteVerification/Atler/SimpleOptionsConverter.hpp"
#include "DiscreteVerification/Atler/SimpleQueryVisitor.hpp"
#include "DiscreteVerification/Atler/SimpleRealMarking.hpp"
#include "DiscreteVerification/Atler/SimpleSMCQuery.hpp"
#include "DiscreteVerification/Atler/SimpleSMCQueryConverter.hpp"
#include "DiscreteVerification/Atler/SimpleTAPNConverter.hpp"
#include "DiscreteVerification/Atler/SimpleTimedArcPetriNet.hpp"
#include "DiscreteVerification/Atler/SimpleVerificationOptions.hpp"

#include <cuda_runtime.h>

namespace VerifyTAPN::DiscreteVerification {

__global__ void runSimulationKernel(
    Atler::SimpleTimedArcPetriNet* stapn,
    Atler::SimpleRealMarking* initialMarking,
    Atler::AST::SimpleSMCQuery* query,
    SMCSettings settings,
    int* successCount,
    int runsNeeded
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= runsNeeded) return;

    // Create local copy of run result
    auto runner = Atler::AtlerRunResult(stapn);
    runner.prepare(*initialMarking);

    bool runRes = false;
    Atler::SimpleRealMarking* newMarking = runner.parent;

    while (!runner.maximal && 
           !(runner.totalTime >= settings.timeBound || 
             runner.totalSteps >= settings.stepBound)) {
        
        Atler::SimpleRealMarking child = *newMarking->clone();
        Atler::SimpleQueryVisitor checker(child, *stapn);
        Atler::AST::BoolResult result;

        query->accept(checker, result);
        runRes = result.value;

        if (runRes) {
            atomicAdd(successCount, 1);
            break;
        }

        newMarking = runner.next();
    }
}
}