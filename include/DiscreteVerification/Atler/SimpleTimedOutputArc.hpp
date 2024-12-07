#ifndef VERIFYTAPN_ATLER_SIMPLETIMEDOUTPUTARC_HPP_
#define VERIFYTAPN_ATLER_SIMPLETIMEDOUTPUTARC_HPP_

#include <cstdint>

namespace VerifyTAPN {
namespace Atler {

struct SimpleTimedTransition;
struct SimpleTimedPlace;

struct SimpleTimedOutputArc {
  SimpleTimedTransition *inputTransition;
  SimpleTimedPlace *outputPlace;
  const uint32_t weight;
};

} // namespace Atler
}; // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_SIMPLETIMEDOUTPUTARC_HPP_ */
