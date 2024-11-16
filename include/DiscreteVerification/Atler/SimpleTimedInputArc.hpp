#ifndef VERIFYTAPN_ATLER_SIMPLETIMEDINPUTARC_HPP_
#define VERIFYTAPN_ATLER_SIMPLETIMEDINPUTARC_HPP_

#include "SimpleTimeInterval.hpp"
#include <cstdint>

namespace VerifyTAPN {
namespace Atler {

struct SimpleTimedTransition;
struct SimpleTimedPlace;

struct SimpleTimedInputArc {
  SimpleTimeInterval interval;
  SimpleTimedPlace &inputPlace;
  SimpleTimedTransition &outputTransition;
  const uint32_t weight;
};

} // namespace Atler
}; // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_SIMPLETIMEDINPUTARC_HPP_ */
