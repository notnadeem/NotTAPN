#ifndef REALMARKINGALLOCATOR_CUH_
#define REALMARKINGALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {
using namespace Cuda;

struct RealMarkingAllocator {
  __host__ static CudaRealMarking *allocate_real_marking(
      CudaRealMarking *h_real_marking, std::unordered_map<CudaTimedTransition *, CudaTimedTransition *> transition_map,
      std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) { // Allocate device memory for marking
    CudaRealMarking *d_real_marking;
    cudaMalloc(&d_real_marking, sizeof(CudaRealMarking));

    CudaRealMarking *temp_real_marking = (CudaRealMarking *)malloc(sizeof(CudaRealMarking));

    CudaRealPlace **d_real_places = cuda_allocate_places_for_marking(h_real_marking, place_map);

    temp_real_marking->places = d_real_places;

    temp_real_marking->placesLength = h_real_marking->placesLength;
    temp_real_marking->deadlocked = h_real_marking->deadlocked;
    temp_real_marking->fromDelay = h_real_marking->fromDelay;

    if (h_real_marking->generatedBy != nullptr) {
      CudaTimedTransition *d_generated_by;
      cudaMalloc(&d_generated_by, sizeof(CudaTimedTransition));

      // Remove const to match map key type
      CudaTimedTransition *non_const_generatedBy = const_cast<CudaTimedTransition *>(h_real_marking->generatedBy);
      auto it = transition_map.find(non_const_generatedBy);
      if (it != transition_map.end()) {
        d_generated_by = it->second;
      }

      temp_real_marking->generatedBy = (CudaTimedTransition *)malloc(sizeof(CudaTimedTransition));

      temp_real_marking->generatedBy = d_generated_by;
    }

    cudaMemcpy(d_real_marking, temp_real_marking, sizeof(CudaRealMarking), cudaMemcpyHostToDevice);

    free(temp_real_marking);

    return d_real_marking;
  };

  __host__ static CudaRealPlace **
  cuda_allocate_places_for_marking(CudaRealMarking *h_marking,
                                   std::unordered_map<CudaTimedPlace *, CudaTimedPlace *> place_map) {
    CudaRealPlace **d_real_places;
    cudaMalloc(&d_real_places, sizeof(CudaRealPlace *) * h_marking->placesLength);

    CudaRealPlace **temp_real_places = (CudaRealPlace **)malloc(sizeof(CudaRealPlace *) * h_marking->placesLength);

    for (int i = 0; i < h_marking->placesLength; i++) {
      CudaRealPlace *temp_real_place = (CudaRealPlace *)malloc(sizeof(CudaRealPlace));

      CudaRealPlace *d_real_place;
      cudaMalloc(&d_real_place, sizeof(CudaRealPlace));

      CudaTimedPlace *temp_place = (CudaTimedPlace *)malloc(sizeof(CudaTimedPlace));

      //Allocate for the tokens array
      CudaDynamicArray<CudaRealToken *>* d_tokens;
      cudaMalloc(&d_tokens, sizeof(CudaDynamicArray<CudaRealToken *>));

      CudaDynamicArray<CudaRealToken *>* temp_tokens = (CudaDynamicArray<CudaRealToken *>*)malloc(sizeof(CudaDynamicArray<CudaRealToken *>));

      CudaRealToken** d_arr;
      cudaMalloc(&d_arr, sizeof(CudaRealToken*) * h_marking->places[i]->tokens->capacity);

      CudaRealToken **temp_arr =
        (CudaRealToken **)malloc(sizeof(CudaRealToken *) * h_marking->places[i]->tokens->capacity);

      for(int j = 0; j < h_marking->places[i]->tokens->size; j++) {
        CudaRealToken *temp_real_token = (CudaRealToken *)malloc(sizeof(CudaRealToken));
        CudaRealToken *d_real_token;
        cudaMalloc(&d_real_token, sizeof(CudaRealToken));

        temp_real_token->age = h_marking->places[i]->tokens->get(j)->age;
        temp_real_token->count = h_marking->places[i]->tokens->get(j)->count;

        cudaMemcpy(d_real_token, temp_real_token, sizeof(CudaRealToken), cudaMemcpyHostToDevice);
        
        temp_arr[j] = d_real_token;
      }

      cudaMemcpy(d_arr, temp_arr, sizeof(CudaRealToken *) * h_marking->places[i]->tokens->capacity, cudaMemcpyHostToDevice);

      temp_tokens->arr = d_arr;
      
      temp_tokens->size = h_marking->places[i]->tokens->size;
      temp_tokens->capacity = h_marking->places[i]->tokens->capacity;

      cudaMemcpy(d_tokens, temp_tokens, sizeof(CudaDynamicArray<CudaRealToken *>), cudaMemcpyHostToDevice);

      CudaTimedPlace *d_place;
      cudaMalloc(&d_place, sizeof(CudaTimedPlace));

      temp_place = place_map[h_marking->places[i]->place];

      cudaMemcpy(d_place, temp_place, sizeof(CudaTimedPlace), cudaMemcpyHostToDevice);

      temp_real_place->place = d_place;
      temp_real_place->tokens = d_tokens;

      cudaMemcpy(d_real_place, temp_real_place, sizeof(CudaRealPlace), cudaMemcpyHostToDevice);

      temp_real_places[i] = d_real_place;
    }

    cudaMemcpy(d_real_places, temp_real_places, sizeof(CudaRealPlace *) * h_marking->placesLength,
               cudaMemcpyHostToDevice);

    free(temp_real_places);

    return d_real_places;
  };
};
} // namespace VerifyTAPN::Alloc

#endif /* REALMARKINGALLOCATOR_CUH_ */