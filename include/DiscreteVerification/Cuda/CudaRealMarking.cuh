#ifndef VERIFYTAPN_ATLER_CUDAREALMARKING_CUH_
#define VERIFYTAPN_ATLER_CUDAREALMARKING_CUH_

#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Cuda/CudaDynamicArray.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {

struct CudaRealToken {
  double age;
  int count;

  __host__ __device__ inline void deltaAge(double x) { age += x; }
};

struct CudaRealPlace {
  Atler::SimpleTimedPlace place;
  CudaDynamicArray<CudaRealToken> tokens;

  __host__ __device__ CudaRealPlace() { tokens = CudaDynamicArray<CudaRealToken>(); }

  __host__ __device__ CudaRealPlace(Atler::SimpleTimedPlace place, size_t tokensLength) : place(place) {
    tokens = CudaDynamicArray<CudaRealToken>(tokensLength);
  }

  __host__ __device__ inline void deltaAge(double x) {
    for (size_t i = 0; i < tokens.size; i++) {
      auto newAge = tokens.get(i).age + x;
      tokens.set(i, CudaRealToken{newAge, tokens.get(i).count});
    }
  }

  __host__ __device__ inline double maxTokenAge() const {
    if (tokens.size == 0) {
      return HUGE_VAL;
    }
    return tokens.get(tokens.size - 1).age;
  }

  __host__ __device__ inline int totalTokenCount() const {
    int count = 0;
    for (size_t i = 0; i < tokens.size; i++) {
      count += tokens.get(i).count;
    }
    return count;
  }

  __host__ __device__ double availableDelay() const {
    // TODO: Change to CUDA implementation
    if (tokens.size == 0) return HUGE_VAL;
    ;
    double delay = ((double)place.timeInvariant.bound) - maxTokenAge();
    return delay <= 0.0f ? 0.0f : delay;
  }

  __host__ __device__ inline bool isEmpty() const { return tokens.size == 0; }
};

struct CudaRealMarking {
  CudaRealPlace *places;
  size_t placesLength = 0;

  bool deadlocked;
  const CudaTimedTransition *generatedBy = nullptr;
  double fromDelay = 0.0;
  // static std::vector<SimpleRealToken> emptyTokenList;

  __host__ __device__ CudaRealMarking() {
    places = new CudaRealPlace[placesLength];
    deadlocked = false;
    generatedBy = nullptr;
    fromDelay = 0.0;
  }

  CudaRealMarking(size_t placesLength) : placesLength(placesLength) {
    places = new CudaRealPlace[placesLength];
    deadlocked = false;
    generatedBy = nullptr;
    fromDelay = 0.0;
  }

  // NOTE: (For CUDA) You don't have to care about the destructor
  // Or just use teh cudaFree() function
  // Also not sure if destructors work in CUDA

  // UPDATE: new and delete should be supported but slower, fix this later
  // Try not to use this
  __host__ __device__ ~CudaRealMarking() { delete[] places; }

  __host__ __device__ void deltaAge(double x) {
    for (size_t i = 0; i < placesLength; i++) {
      places[i].deltaAge(x);
    }
  }

  __host__ __device__ CudaRealMarking *clone() const {
    CudaRealMarking *result = new CudaRealMarking();
    result->placesLength = placesLength;
    result->places = new CudaRealPlace[placesLength];
    for (size_t i = 0; i < placesLength; i++) {
      result->places[i].place = places[i].place;
      for (size_t j = 0; j < places[i].tokens.size; j++) {
        result->places[i].tokens.add(places[i].tokens.get(j));
      }
    }
    result->deadlocked = deadlocked;
    result->generatedBy = generatedBy;
    result->fromDelay = fromDelay;
    return result;
  }

  __host__ __device__ uint32_t numberOfTokensInPlace(int placeId) const { return places[placeId].totalTokenCount(); }

  __host__ __device__ double availableDelay() const {
    double available = HUGE_VAL;
    for (size_t i = 0; i < placesLength; i++) {
      if (places[i].tokens.size == 0) continue;
      double delay = places[i].availableDelay();
      if (delay < available) {
        available = delay;
      }
    }
    return available;
  }
};

} // namespace VerifyTAPN::Cuda

#endif