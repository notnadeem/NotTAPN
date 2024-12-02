#include "DiscreteVerification/Alloc/RealMarkingAllocator.cuh"
#include "DiscreteVerification/Cuda/CudaRunResult.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda;

__host__ CudaRealMarking *allocate(CudaRealMarking *h_real_marking,
                                   std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
                                   std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) {

  // Allocate device memory for marking
  CudaRealMarking *d_real_marking;
  cudaMalloc(&d_real_marking, sizeof(CudaRealMarking));

  CudaRealMarking *temp_real_marking = (CudaRealMarking *)malloc(sizeof(CudaRealMarking *));

  RealMarkingAllocator realMarkingAllocator;
  
  CudaRealPlace **d_real_places = realMarkingAllocator.cuda_allocate_places_for_marking(h_real_marking, place_map);

  temp_real_marking->places = (CudaRealPlace **)malloc(sizeof(CudaRealPlace) * h_real_marking->placesLength);

  for (int i = 0; i < h_real_marking->placesLength; i++) {
    temp_real_marking->places[i] = d_real_places[i];
  }

  temp_real_marking->placesLength = h_real_marking->placesLength;
  temp_real_marking->deadlocked = h_real_marking->deadlocked;
  temp_real_marking->fromDelay = h_real_marking->fromDelay;

  CudaTimedTransition *d_generated_by;
  cudaMalloc(&d_generated_by, sizeof(CudaTimedTransition));

  if (h_real_marking->generatedBy != nullptr) {
    // Remove const to match map key type
    CudaTimedTransition *non_const_generatedBy = const_cast<CudaTimedTransition *>(h_real_marking->generatedBy);
    auto it = transition_map.find(non_const_generatedBy);
    if (it != transition_map.end()) {
      d_generated_by = it->second;
    }
  }

  temp_real_marking->generatedBy = d_generated_by;

  cudaMemcpy(d_real_marking, temp_real_marking, sizeof(CudaRealMarking), cudaMemcpyHostToDevice);

  cudaFree(temp_real_marking);

  return d_real_marking;
};

__host__ CudaRealPlace **
cuda_allocate_places_for_marking(CudaRealMarking *h_marking,
                                 std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) {

  CudaRealPlace **d_real_places;
  cudaMalloc(&d_real_places, sizeof(CudaRealPlace *) * h_marking->placesLength);

  CudaRealPlace **temp_real_places = (CudaRealPlace **)malloc(sizeof(CudaRealPlace *) * h_marking->placesLength);

  for (int i = 0; i < h_marking->placesLength; i++) {
    CudaRealPlace *d_real_place;
    cudaMalloc(&d_real_place, sizeof(CudaRealPlace));

    CudaTimedPlace *d_place;
    cudaMalloc(&d_place, sizeof(CudaTimedPlace));

    d_place = place_map[h_marking->places[i]->place];

    cudaMemcpy(d_real_place->place, d_place, sizeof(CudaTimedPlace), cudaMemcpyHostToDevice);

    temp_real_places[i] = d_real_place;
  }

  cudaMemcpy(d_real_places, temp_real_places, sizeof(CudaRealPlace *) * h_marking->placesLength,
             cudaMemcpyHostToDevice);

  cudaFree(temp_real_places);

  return d_real_places;
}

__host__ void allocatePointerMembers(CudaRealMarking *realMarkingHost) {};
} // namespace VerifyTAPN::Alloc