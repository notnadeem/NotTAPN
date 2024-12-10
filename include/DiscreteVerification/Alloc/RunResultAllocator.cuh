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
    if (h_run_result->datesSampled != nullptr) {
      CudaDynamicArray<double> *d_dates;
      cudaMalloc(&d_dates, sizeof(CudaDynamicArray<double>));

      CudaDynamicArray<double> *temp_d_dates = (CudaDynamicArray<double> *)malloc(sizeof(CudaDynamicArray<double>));

      double *d_dates_arr;
      cudaMalloc(&d_dates_arr, sizeof(double *) * h_run_result->datesSampled->capacity);

      double *temp_dates_arr = (double *)malloc(sizeof(double *) * h_run_result->datesSampled->capacity);

      for (int i = 0; i < h_run_result->datesSampled->size; i++) {
        temp_dates_arr[i] = h_run_result->datesSampled->get(i);
      }

      cudaMemcpy(d_dates_arr, temp_dates_arr, sizeof(double *) * h_run_result->datesSampled->capacity,
                 cudaMemcpyHostToDevice);

      temp_d_dates->arr = d_dates_arr;
      temp_d_dates->size = h_run_result->datesSampled->size;
      temp_d_dates->capacity = h_run_result->datesSampled->capacity;

      cudaMemcpy(d_dates, temp_d_dates, sizeof(CudaDynamicArray<double>), cudaMemcpyHostToDevice);

      temp_run_result->datesSampled = d_dates;
    }

    // Allocate transition intervals 2d array - for now skipping this as it's always empty array
    CudaDynamicArray<CudaDynamicArray<CudaInterval> *> *d_transitionIntervals =
        allocate_dynamic_dobbel_array(h_run_result->transitionIntervals);

    temp_run_result->transitionIntervals = d_transitionIntervals;

    // Copy CudaRunResult from host to device
    cudaMemcpy(runResultDevice, temp_run_result, sizeof(CudaRunResult), cudaMemcpyHostToDevice);

    free(temp_run_result);
    auto result = new std::pair<CudaRunResult *, CudaRealMarking *>(runResultDevice, d_cmarking);

    return result;
  };

  __host__ static CudaDynamicArray<CudaDynamicArray<CudaInterval> *> *
  allocate_dynamic_dobbel_array(CudaDynamicArray<CudaDynamicArray<CudaInterval> *> *h_transitionIntervals) {

    CudaDynamicArray<CudaDynamicArray<CudaInterval> *> *d_transitionIntervals;
    cudaMalloc(&d_transitionIntervals, sizeof(CudaDynamicArray<CudaDynamicArray<CudaInterval> *>));

    CudaDynamicArray<CudaInterval> **d_outer_arr;
    cudaMalloc(&d_outer_arr, sizeof(CudaDynamicArray<CudaInterval> *) * h_transitionIntervals->size
                                 ? h_transitionIntervals->size
                                 : 1);

    CudaDynamicArray<CudaInterval> **temp_outer_arr = (CudaDynamicArray<CudaInterval> **)malloc(
        sizeof(CudaDynamicArray<CudaInterval> *) * h_transitionIntervals->size ? h_transitionIntervals->size : 1);

    for (size_t i = 0; i < h_transitionIntervals->size; i++) {
      CudaDynamicArray<CudaInterval> *d_inner_struct;
      cudaMalloc(&d_inner_struct, sizeof(CudaDynamicArray<CudaInterval>));

      CudaInterval *d_intervals_arr;
      cudaMalloc(&d_intervals_arr, sizeof(CudaInterval) * h_transitionIntervals->arr[i]->size);

      cudaMemcpy(d_intervals_arr, h_transitionIntervals->arr[i]->arr,
                 sizeof(CudaInterval) * h_transitionIntervals->arr[i]->size, cudaMemcpyHostToDevice);

      CudaDynamicArray<CudaInterval> *temp_inner_struct =
          (CudaDynamicArray<CudaInterval> *)malloc(sizeof(CudaDynamicArray<CudaInterval>));

      temp_inner_struct->arr = d_intervals_arr;
      temp_inner_struct->size = h_transitionIntervals->arr[i]->size;
      temp_inner_struct->capacity = h_transitionIntervals->arr[i]->capacity;

      cudaMemcpy(d_inner_struct, temp_inner_struct, sizeof(CudaDynamicArray<CudaInterval>), cudaMemcpyHostToDevice);

      temp_outer_arr[i] = d_inner_struct;
    }

    cudaError_t err =
        cudaMemcpy(d_outer_arr, temp_outer_arr, sizeof(CudaDynamicArray<CudaInterval> *) * h_transitionIntervals->size,
                   cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    CudaDynamicArray<CudaDynamicArray<CudaInterval> *> *temp_transitionsIntervals =
        (CudaDynamicArray<CudaDynamicArray<CudaInterval> *> *)malloc(
            sizeof(CudaDynamicArray<CudaDynamicArray<CudaInterval> *>));

    temp_transitionsIntervals->arr = d_outer_arr;
    temp_transitionsIntervals->size = h_transitionIntervals->size;
    temp_transitionsIntervals->capacity = h_transitionIntervals->capacity;

    cudaMemcpy(d_transitionIntervals, temp_transitionsIntervals,
               sizeof(CudaDynamicArray<CudaDynamicArray<CudaInterval> *>), cudaMemcpyHostToDevice);

    return d_transitionIntervals;
  };

private:
  __host__ static void allocatePointerMembers(CudaRunResult *runResultHost) {};
};
} // namespace VerifyTAPN::Alloc

#endif /* RUNRESULTALLOCATOR_CUH_ */