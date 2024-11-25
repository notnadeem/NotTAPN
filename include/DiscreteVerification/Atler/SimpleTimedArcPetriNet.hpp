#ifndef VERIFYYAPN_ATLER_SIMPLETIMEDARCPETRINET_HPP_
#define VERIFYYAPN_ATLER_SIMPLETIMEDARCPETRINET_HPP_

#include "SimpleTimedInhibitorArc.hpp"
#include "SimpleTimedInputArc.hpp"
#include "SimpleTimedOutputArc.hpp"
#include "SimpleTimedPlace.hpp"
#include "SimpleTimedTransportArc.hpp"

namespace VerifyTAPN {
namespace Atler {

struct SimpleTimedArcPetriNet {
  SimpleTimedPlace *places;
  size_t placesLength;
  SimpleTimedTransition **transitions;
  size_t transitionsLength;
  SimpleTimedInputArc **inputArcs;
  size_t inputArcsLength;
  SimpleTimedOutputArc **outputArcs;
  size_t outputArcsLength;
  SimpleTimedTransportArc **transportArcs;
  size_t transportArcsLength;
  SimpleTimedInhibitorArc **inhibitorArcs;
  size_t inhibitorArcsLength;
  int maxConstant;
  int gcd;

  void print(std::ostream &out) const;
};

} // namespace Atler
}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_SIMPLETIMEDARCPETRINET_HPP_ */
