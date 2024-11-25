#ifndef VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_
#define VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_

#include "SimpleTimedPlace.hpp"
#include "SimpleTimedTransition.hpp"
#include "SimpleDynamicArray.hpp"
#include <iostream>
#include <limits>

namespace VerifyTAPN::Atler {

struct SimpleRealToken {
  double age;
  int count;

  inline void deltaAge(double x) {
      age += x;
  }
};

struct SimpleRealPlace {
  SimpleTimedPlace place;
  SimpleDynamicArray<SimpleRealToken> tokens;

  SimpleRealPlace() {
     tokens = SimpleDynamicArray<SimpleRealToken>();
  }

  SimpleRealPlace(SimpleTimedPlace place, size_t tokensLength): place(place) {
     tokens = SimpleDynamicArray<SimpleRealToken>(tokensLength);
  }

  inline void deltaAge(double x) {
      for(size_t i = 0; i < tokens.size; i++) {
          auto newAge = tokens.get(i).age + x;
          tokens.set(i, SimpleRealToken{newAge, tokens.get(i).count});
      }
  }

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
  size_t placesLength = 0;

  bool deadlocked;
  const SimpleTimedTransition *generatedBy = nullptr;
  double fromDelay = 0.0;
  // static std::vector<SimpleRealToken> emptyTokenList;

  SimpleRealMarking() {
      places = new SimpleRealPlace[placesLength];
      deadlocked = false;
      generatedBy = nullptr;
      fromDelay = 0.0;
  }

  SimpleRealMarking(size_t placesLength) : placesLength(placesLength) {
      places = new SimpleRealPlace[placesLength];
      deadlocked = false;
      generatedBy = nullptr;
      fromDelay = 0.0;
  }

  // NOTE: (For CUDA) You don't have to care about the destructor
  // Or just use teh cudaFree() function
  // Also not sure if destructors work in CUDA
  // ~SimpleRealMarking() {
  //     delete[] places;
  // }

  void deltaAge(double x) {
      for (size_t i = 0; i < placesLength; i++) {
          places[i].deltaAge(x);
      }
  }

  SimpleRealMarking *clone() const {
        SimpleRealMarking *result = new SimpleRealMarking();
        result->placesLength = placesLength;
        result->places = new SimpleRealPlace[placesLength];
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
