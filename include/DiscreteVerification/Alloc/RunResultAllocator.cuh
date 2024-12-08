#ifndef RUNRESULTALLOCATOR_CUH_
#define RUNRESULTALLOCATOR_CUH_

#include "DiscreteVerification/Alloc/CudaPetriNetAllocator.cuh"
#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaInterval.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>
namespace VerifyTAPN::Alloc {

using namespace Cuda;
using namespace Util;

struct RunResultAllocator {
  __host__ static std::pair<CudaRunResult *, CudaRealMarking *> *
  allocate(CudaRunResult *h_run_result, CudaRealMarking *h_marking, int blocks, int threadsPerBlock) {
    int numThreads = blocks * threadsPerBlock;

    // Allocate device memory for rngStates
    cudaMalloc(&(h_run_result->rngStates), numThreads * sizeof(curandState_t));

    // Allocate device memory for CudaRunResult
    CudaRunResult *runResultDevice;
    cudaMalloc(&runResultDevice, sizeof(CudaRunResult));

    CudaRunResult *temp_run_result = (CudaRunResult *)malloc(sizeof(CudaRunResult));

    // Allocate petri net
    CudaPetriNetAllocator petriNetAllocator;
    CudaTimedArcPetriNet *d_tapn = petriNetAllocator.cuda_allocator(h_run_result->tapn);

    temp_run_result->tapn = d_tapn;

    RealMarkingAllocator realMarkingAllocator;

    CudaRealMarking *d_cmarking = realMarkingAllocator.allocate_real_marking(
        h_marking, petriNetAllocator.transition_map, petriNetAllocator.place_map);

    temp_run_result->realMarking = d_cmarking;

    // Allocate all the dynamic arrays
    if(h_run_result->datesSampled != nullptr) {
      CudaDynamicArray<double> *d_dates;
    cudaMalloc(&d_dates, sizeof(CudaDynamicArray<double>));

    CudaDynamicArray<double>* temp_d_dates = (CudaDynamicArray<double>*)malloc(sizeof(CudaDynamicArray<double>));

    double *d_dates_arr;
    cudaMalloc(&d_dates_arr, sizeof(double *) * h_run_result->datesSampled->capacity);

    double *temp_dates_arr = (double *)malloc(sizeof(double *) * h_run_result->datesSampled->capacity);

    for (int i = 0; i < h_run_result->datesSampled->size; i++) {
      temp_dates_arr[i] = h_run_result->datesSampled->get(i);
    }

    cudaMemcpy(d_dates_arr, temp_dates_arr, sizeof(double *) * h_run_result->datesSampled->capacity,
               cudaMemcpyHostToDevice);
    
    temp_d_dates->arr = d_dates_arr;
    temp_d_dates->ownsArray = h_run_result->datesSampled->ownsArray;
    temp_d_dates->size = h_run_result->datesSampled->size;
    temp_d_dates->capacity = h_run_result->datesSampled->capacity;

    cudaMemcpy(d_dates, temp_d_dates, sizeof(CudaDynamicArray<double>), cudaMemcpyHostToDevice);

    temp_run_result->datesSampled = d_dates;
    }

    //Allocate transition intervals 2d array - for now skipping this as it's always empty array

    // Copy CudaRunResult from host to device
    cudaMemcpy(runResultDevice, temp_run_result, sizeof(CudaRunResult), cudaMemcpyHostToDevice);

    free(temp_run_result);
    auto result = new std::pair<CudaRunResult *, CudaRealMarking *>(runResultDevice, d_cmarking);

    return result;
  };

private:
  __host__ static void allocatePointerMembers(CudaRunResult *runResultHost) {};
};
} // namespace VerifyTAPN::Alloc

#endif /* RUNRESULTALLOCATOR_CUH_ */