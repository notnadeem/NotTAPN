#ifndef VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_
#define VERIFYTAPN_ATLER_SIMPLEREALMARKING_HPP_

#include "SimpleTimedPlace.hpp"
#include "SimpleTimedTransition.hpp"

namespace VerifyTAPN::Atler {

struct SimpleRealToken {
  double age;
  int count;
};

struct SimpleRealPlace {
  SimpleTimedPlace place;
  SimpleRealToken *tokens;
  size_t tokensLength;

};

struct SimpleRealMarking {
  SimpleRealPlace *places;
  size_t placesLength;

  bool deadlocked;
  const SimpleTimedTransition *generatedBy = nullptr;
  double fromDelay = 0.0;
  // static std::vector<SimpleRealToken> emptyTokenList;
};

} // namespace VerifyTAPN::Atler

#endif
