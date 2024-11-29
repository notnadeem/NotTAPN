#ifndef VERIFYYAPN_ATLER_SIMPLETIMEDPLACE_HPP_
#define VERIFYYAPN_ATLER_SIMPLETIMEDPLACE_HPP_

#include "SimpleTimeInvariant.hpp"
#include "SimpleTimedInhibitorArc.hpp"
#include "SimpleTimedInputArc.hpp"
#include "SimpleTimedOutputArc.hpp"
#include "SimpleTimedTransportArc.hpp"

namespace VerifyTAPN {
namespace Atler {
// TODO change to cuda because of arcs
enum PlaceType { Inv, Dead, Std };

struct SimpleTimedPlace {
  int index;
  PlaceType type;

  const char *name;
  size_t nameLength;

  const char *id;
  size_t idLength;

  SimpleTimeInvariant timeInvariant;
  bool untimed;
  int maxConstant;
  bool containsInhibitorArcs;
  SimpleTimedInputArc **inputArcs;
  size_t inputArcsLength;
  SimpleTimedTransportArc **transportArcs;
  size_t transportArcsLength;
  SimpleTimedTransportArc **prodTransportArcs;
  size_t prodTransportArcsLength;
  SimpleTimedInhibitorArc **inhibitorArcs;
  size_t inhibitorArcsLength;
  SimpleTimedOutputArc **outputArcs;
  size_t outputArcsLength;
  std::pair<double, double> _position;
};

} // namespace Atler

}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_SIMPLETIMEDPLACE_HPP_ */
