#ifndef VERIFYYAPN_ATLER_SIMPLETIMEDPLACE_HPP_
#define VERIFYYAPN_ATLER_SIMPLETIMEDPLACE_HPP_

#include <string>
#include <vector>

#include "SimpleTimeInvariant.hpp"
#include "SimpleTimedInputArc.hpp"
#include "SimpleTimedTransportArc.hpp"
#include "SimpleTimedInhibitorArc.hpp"
#include "SimpleTimedOutputArc.hpp"

namespace VerifyTAPN {
namespace Atler {

enum PlaceType { Inv, Dead, Std };

struct SimpleTimedPlace {
  int index;
  PlaceType type;
  std::string name;
  std::string id;
  SimpleTimeInvariant timeInvariant;
  bool untimed;
  int maxConstant;
  bool containsInhibitorArcs;
  SimpleTimedInputArc* inputArcs;
  size_t inputArcsLength;
  SimpleTimedTransportArc* transportArcs;
  size_t transportArcsLength;
  SimpleTimedTransportArc* prodTransportArcs;
  size_t prodTransportArcsLength;
  SimpleTimedInhibitorArc* inhibitorArcs;
  size_t inhibitorArcsLength;
  SimpleTimedOutputArc* outputArcs;
  size_t outputArcsLength;
  std::pair<double, double> _position;
};

} // namespace Atler

}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_SIMPLETIMEDPLACE_HPP_ */
