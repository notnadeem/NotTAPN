#ifndef VERIFYTAPN_ATLER_SIMPLETIMEDTRANSITION_HPP_
#define VERIFYTAPN_ATLER_SIMPLETIMEDTRANSITION_HPP_

#include "SimpleStochasticStructure.hpp"
#include "SimpleTimedInhibitorArc.hpp"
#include "SimpleTimedInputArc.hpp"
#include "SimpleTimedOutputArc.hpp"
#include "SimpleTimedTransportArc.hpp"
#include <string>

namespace VerifyTAPN {
namespace Atler {
struct SimpleTimedTransition {
  int index = 0;
  std::string name;
  std::string id;

  SimpleTimedInputArc** preset = nullptr;
  int presetLength = 0;
  SimpleTimedOutputArc** postset = nullptr;
  int postsetLength = 0;
  SimpleTimedTransportArc** transportArcs = nullptr;
  int transportArcsLength = 0;
  SimpleTimedInhibitorArc** inhibitorArcs = nullptr;
  int inhibitorArcsLength = 0;

  bool untimedPostset = false;
  bool urgent = false;
  bool controllable{};
  std::pair<double, double> _position;
  SimpleSMC::Distribution _distribution;
  double _weight;
  SimpleSMC::FiringMode _firingMode = SimpleSMC::Oldest;

  inline int getPresetSize() const {
      return presetLength + transportArcsLength;
  }
};
} // namespace Atler
} // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_SIMPLETIMEDTRANSITION_HPP_ */
