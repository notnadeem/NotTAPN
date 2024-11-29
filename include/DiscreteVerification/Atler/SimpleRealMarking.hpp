#ifndef VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_
#define VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_

#include "DiscreteVerification/Atler/SimpleTimedArcPetriNet.hpp"
#include "SimpleDynamicArray.hpp"
#include "SimpleTimedPlace.hpp"
#include "SimpleTimedTransition.hpp"
#include <iostream>
#include <limits>

namespace VerifyTAPN::Atler {

struct SimpleRealToken {
  double age;
  int count;

  inline void add(int num) { count = count + num; }
  inline void remove(int num) { count = count - num; }
  inline void deltaAge(double x) { age += x; }
};

struct SimpleRealPlace {
  SimpleTimedPlace place;
  SimpleDynamicArray<SimpleRealToken *> tokens;

  SimpleRealPlace() { tokens = SimpleDynamicArray<SimpleRealToken *>(); }

  SimpleRealPlace(SimpleTimedPlace place, size_t tokensLength) : place(place) {
    tokens = SimpleDynamicArray<SimpleRealToken *>(tokensLength);
  }

  // TODO: check if this works
  inline void deltaAge(double x) {
    for (size_t i = 0; i < tokens.size; i++) {
      tokens.get(i)->deltaAge(x);
    }
  }

  inline void addToken(SimpleRealToken newToken) {
    size_t index = 0;
    for (size_t i = 0; i < tokens.size; i++) {
      SimpleRealToken *token = tokens.get(i);
      if (token->age == newToken.age) {
        token->add(newToken.count);
        return;
      }
      if (token->age > newToken.age)
        break;
      index++;
    }
    // NOTE: Check if this works, might be a issue
    if (index >= tokens.size) {
      tokens.add(&newToken);
    } else {
      tokens.set(index, &newToken);
    }
  }

  inline double maxTokenAge() const {
    if (tokens.size == 0) {
      return -std::numeric_limits<double>::infinity();
    }
    return tokens.get(tokens.size - 1)->age;
  }

  inline int totalTokenCount() const {
    int count = 0;
    for (size_t i = 0; i < tokens.size; i++) {
      count += tokens.get(i)->count;
    }
    return count;
  }

  double availableDelay() const {
    // TODO: Change to CUDA implementation
    if (tokens.size == 0)
      return std::numeric_limits<double>::infinity();
    double delay = ((double)place.timeInvariant.bound) - maxTokenAge();
    return delay <= 0.0f ? 0.0f : delay;
  }

  inline bool isEmpty() const { return tokens.size == 0; }
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

  void addTokenInPlace(SimpleTimedPlace &place, SimpleRealToken& newToken) {
      places[place.index].addToken(newToken);
  }

  bool canDeadlock(const SimpleTimedArcPetriNet &tapn, int maxDelay, bool ignoreCanDelay) const {
      return deadlocked;
  }

  inline bool canDeadlock(const SimpleTimedArcPetriNet &tapn, const int maxDelay) const {
      return canDeadlock(tapn, maxDelay, false);
  };

  double availableDelay() const {
    double available = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < placesLength; i++) {
      if (places[i].tokens.size == 0)
        continue;
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
