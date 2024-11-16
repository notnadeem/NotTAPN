#ifndef VERIFYTAPN_ATLER_SIMPLETIMEDTRANSPORTARC_HPP_
#define VERIFYTAPN_ATLER_SIMPLETIMEDTRANSPORTARC_HPP_

#include "SimpleTimeInterval.hpp"

namespace VerifyTAPN {
namespace Atler {

class SimpleTimedTransition;
class SimpleTimedPlace;

struct SimpleTimedTransportArc {
  Atler::SimpleTimeInterval interval;
  SimpleTimedPlace &source;
  SimpleTimedTransition &transition;
  SimpleTimedPlace &destination;
  const int weight;
};

} // namespace
} // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_SIMPLETIMEDTRANSPORTARC_HPP_ */
