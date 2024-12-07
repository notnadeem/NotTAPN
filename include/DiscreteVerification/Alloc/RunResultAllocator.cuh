#ifndef RUNRESULTALLOCATOR_CUH_
#define RUNRESULTALLOCATOR_CUH_

#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaInterval.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"
#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"

#include <cuda_runtime.h>
namespace VerifyTAPN::Alloc {

using namespace Cuda;
using namespace Util;

struct RunResultAllocator {
  __host__ static std::pair<CudaRunResult*, CudaRealMarking*>*
  allocate(CudaRunResult *h_run_result, CudaRealMarking *h_marking, int blocks, int threadsPerBlock) {
    int numThreads = blocks * threadsPerBlock;

    // Allocate device memory for rngStates
    cudaMalloc(&(h_run_result->rngStates), numThreads * sizeof(curandState_t));

    // Allocate device memory for CudaRunResult
    CudaRunResult *runResultDevice;
    cudaMalloc(&runResultDevice, sizeof(CudaRunResult));

    CudaRunResult *temp_run_result = (CudaRunResult *)malloc(sizeof(CudaRunResult));

    //Allocate petri net
    CudaPetriNetAllocator petriNetAllocator;
    CudaTimedArcPetriNet *d_tapn = petriNetAllocator.cuda_allocator(h_run_result->tapn);
    // CudaTimedArcPetriNet *temp_tapn = (CudaTimedArcPetriNet *)malloc(sizeof(CudaTimedArcPetriNet));

    // cudaMemcpy(temp_real_places, d_real_places, sizeof(CudaRealMarking **), cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp_tapn, d_tapn, sizeof(CudaTimedArcPetriNet), cudaMemcpyDeviceToHost);
    
    temp_run_result->tapn = d_tapn;
    
    RealMarkingAllocator realMarkingAllocator;

    temp_run_result->dates_sampled = h_run_result->dates_sampled;
    temp_run_result->transitionsStatistics = h_run_result->transitionsStatistics;

    CudaRealMarking *d_cmarking = realMarkingAllocator.allocate_real_marking(h_marking, petriNetAllocator.transition_map, petriNetAllocator.place_map);
    // temp_run_result->origin =
    //     realMarkingAllocator.allocate_real_marking(h_marking, petriNetAllocator.transition_map, petriNetAllocator.place_map);

    temp_run_result->defaultTransitionIntervals = h_run_result->defaultTransitionIntervals;
    temp_run_result->transitionIntervals = h_run_result->transitionIntervals;

    // Copy CudaRunResult from host to device
    cudaMemcpy(runResultDevice, temp_run_result, sizeof(CudaRunResult), cudaMemcpyHostToDevice);

    free(temp_run_result);
    auto result = new std::pair<CudaRunResult*, CudaRealMarking*>(
        runResultDevice, d_cmarking);

    return result;
  };

private:
  __host__ static void allocatePointerMembers(CudaRunResult *runResultHost) {};
};
} // namespace VerifyTAPN::Alloc

#endif /* RUNRESULTALLOCATOR_CUH_ */