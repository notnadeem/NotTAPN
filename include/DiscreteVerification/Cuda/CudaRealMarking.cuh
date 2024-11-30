#ifndef VERIFYTAPN_ATLER_CUDAREALMARKING_CUH_
#define VERIFYTAPN_ATLER_CUDAREALMARKING_CUH_

#include "DiscreteVerification/Cuda/CudaTimedPlace.cuh"
#include "DiscreteVerification/Cuda/CudaDynamicArray.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {
struct CudaRealToken {
  double age;
  int count;

  __host__ __device__ inline void add(int num) { count = count + num; }
  __host__ __device__ inline void remove(int num) { count = count - num; }
  __host__ __device__ inline void deltaAge(double x) { age += x; }
};

struct CudaRealPlace {
  CudaTimedPlace place;
  CudaDynamicArray<CudaRealToken *>* tokens;

  __host__ __device__ CudaRealPlace() { tokens = new CudaDynamicArray<CudaRealToken *>(10); }

  __host__ __device__ CudaRealPlace(CudaTimedPlace place, size_t tokensLength) : place(place) {
    tokens = new CudaDynamicArray<CudaRealToken *>(tokensLength);
  }

  __host__ __device__ inline void deltaAge(double x) {
    // print all the places
    printf("place name: %s\n", place.name);
    printf("deltaAge: %f\n", x);
    for (size_t i = 0; i < tokens->size; i++) {
      tokens->get(i)->deltaAge(x);
    }
  }

  __host__ __device__ inline void addToken(CudaRealToken& newToken) {
    // got rid of pointers here, might break
    size_t index = 0;
    for (size_t i = 0; i < tokens->size; i++) {
      CudaRealToken *token = tokens->get(i);
      if (token->age == newToken.age) {
        token->add(newToken.count);
        return;
      }
      if (token->age > newToken.age) break;
      index++;
    }
    // NOTE: Check if this works, might be a issue
    if (index >= tokens->size) {
      tokens->add(&newToken);
    } else {
      tokens->set(index, &newToken);
    }
  }

  __host__ __device__ inline double maxTokenAge() const {
    if (tokens->size == 0) {
      return HUGE_VAL;
    }
    return tokens->get(tokens->size - 1)->age;
  }

  __host__ __device__ inline int totalTokenCount() const {
    int count = 0;
    for (size_t i = 0; i < tokens->size; i++) {
      count += tokens->get(i)->count;
    }
    return count;
  }

  __host__ __device__ double availableDelay() const {
    if (tokens->size == 0) return HUGE_VAL;
    
    double delay = ((double)place.timeInvariant.bound) - maxTokenAge();
    return delay <= 0.0f ? 0.0f : delay;
  }

  __host__ __device__ inline bool isEmpty() const { return tokens->size == 0; }
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

  __host__ __device__ CudaRealMarking(size_t placesLength) : placesLength(placesLength) {
    places = new CudaRealPlace[placesLength];
    deadlocked = false;
    generatedBy = nullptr;
    fromDelay = 0.0;
  }

  __host__ __device__ CudaRealMarking(const CudaRealMarking& other) {
      placesLength = other.placesLength;
      places = other.places;
      deadlocked = other.deadlocked;
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
      for (size_t j = 0; j < places[i].tokens->size; j++) {
        result->places[i].tokens->add(places[i].tokens->get(j));
      }
    }
    result->deadlocked = deadlocked;
    result->generatedBy = generatedBy;
    result->fromDelay = fromDelay;
    return result;
  }

  __host__ __device__ uint32_t numberOfTokensInPlace(int placeId) const { return places[placeId].totalTokenCount(); }

  __host__ __device__ void addTokenInPlace(CudaTimedPlace &place, CudaRealToken &newToken) {
    places[place.index].addToken(newToken);
  }

  __host__ __device__ bool canDeadlock(const CudaTimedArcPetriNet &tapn, int maxDelay, bool ignoreCanDelay) const {
      return deadlocked;
  }

  __host__ __device__ inline bool canDeadlock(const CudaTimedArcPetriNet &tapn, const int maxDelay) const {
      return canDeadlock(tapn, maxDelay, false);
  };

  __host__ __device__ double availableDelay() const {
    double available = HUGE_VAL;
    for (size_t i = 0; i < placesLength; i++) {
      if (places[i].tokens->size == 0) continue;
      double delay = places[i].availableDelay();
      if (delay < available) {
        available = delay;
      }
    }
    return available;
  }
};
} // namespace Cuda
} // namespace VerifyTAPN

#endif