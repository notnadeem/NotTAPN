#ifndef VERIFYTAPN_ATLER_SIMPLEREALMARKING_CUH_
#define VERIFYTAPN_ATLER_SIMPLEREALMARKING_CUH_

#include "DiscreteVerification/Cuda/SimpleDynamicArray.cuh"
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Cuda/SimpleTimedTransition.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {

struct SimpleRealToken {
  double age;
  int count;
};

struct SimpleRealPlace {
  Atler::SimpleTimedPlace place;
  SimpleDynamicArray<SimpleRealToken> tokens;

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

struct SimpleRealMarking {
  SimpleRealPlace *places;
  size_t placesLength;

  bool deadlocked;
  const SimpleTimedTransition *generatedBy = nullptr;
  double fromDelay = 0.0;
  // static std::vector<SimpleRealToken> emptyTokenList;

  __host__ __device__ SimpleRealMarking() {
    placesLength = 0;
    places = nullptr;
    deadlocked = false;
    generatedBy = nullptr;
    fromDelay = 0.0;
  }

  // NOTE: (For CUDA) You don't have to care about the destructor
  // Or just use teh cudaFree() function
  // Also not sure if destructors work in CUDA

  // UPDATE: new and delete should be supported but slower, fix this later
  __host__ __device__ ~SimpleRealMarking() { delete[] places; }

  __host__ __device__ SimpleRealMarking clone() const {
    SimpleRealMarking result;
    result.placesLength = placesLength;
    result.places = new SimpleRealPlace[placesLength];
    for (size_t i = 0; i < placesLength; i++) {
      result.places[i].place = places[i].place;
      for (size_t j = 0; j < places[i].tokens.size; j++) {
        result.places[i].tokens.add(places[i].tokens.get(j));
      }
    }
    result.deadlocked = deadlocked;
    result.generatedBy = generatedBy;
    result.fromDelay = fromDelay;
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

} // namespace VerifyTAPN::Alter

#endif