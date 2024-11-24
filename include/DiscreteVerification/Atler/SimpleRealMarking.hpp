#ifndef VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_
#define VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_

#include "SimpleTimedPlace.hpp"
#include "SimpleTimedTransition.hpp"
#include "SimpleDynamicArray.hpp"
#include <limits>

namespace VerifyTAPN::Atler {

struct SimpleRealToken {
  double age;
  int count;
};

struct SimpleRealPlace {
  SimpleTimedPlace place;
  SimpleDynamicArray<SimpleRealToken> tokens;

  inline double maxTokenAge() const {
      if(tokens.size == 0) {
          return -std::numeric_limits<double>::infinity();
      }
      return tokens.get(tokens.size - 1).age;
  }

  inline int totalTokenCount() const {
      int count = 0;
      for(size_t i = 0; i < tokens.size; i++) {
          count += tokens.get(i).count;
      }
      return count;
  }

  double availableDelay() const {
      // TODO: Change to CUDA implementation
      if(tokens.size == 0) return std::numeric_limits<double>::infinity();
      double delay = ((double) place.timeInvariant.bound) - maxTokenAge();
      return delay <= 0.0f ? 0.0f : delay;
  }

  inline bool isEmpty() const {
      return tokens.size == 0;
  }
};

struct SimpleRealMarking {
  SimpleRealPlace *places;
  size_t placesLength;

  bool deadlocked;
  const SimpleTimedTransition *generatedBy = nullptr;
  double fromDelay = 0.0;
  // static std::vector<SimpleRealToken> emptyTokenList;

  SimpleRealMarking() {
      placesLength = 0;
      places = nullptr;
      deadlocked = false;
      generatedBy = nullptr;
      fromDelay = 0.0;
  }

  // NOTE: (For CUDA) You don't have to care about the destructor
  // Or just use teh cudaFree() function
  // Also not sure if destructors work in CUDA
  ~SimpleRealMarking() {
      delete[] places;
  }

  SimpleRealMarking clone() const {
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

  uint32_t numberOfTokensInPlace(int placeId) const {
      return places[placeId].totalTokenCount();
  }

  double availableDelay() const {
      double available = std::numeric_limits<double>::infinity();
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

} // namespace VerifyTAPN::Atler

#endif
