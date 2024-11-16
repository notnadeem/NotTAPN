#ifndef VERIFYYAPN_ATLER_SIMPLETIMEDINHIBITORARC_HPP_
#define VERIFYYAPN_ATLER_SIMPLETIMEDINHIBITORARC_HPP_

#include "Core/TAPN/InhibitorArc.hpp"
#include <cstdint>

namespace VerifyTAPN {
namespace Atler {

struct SimpleTimedTransition;
struct SimpleTimedPlace;

struct SimpleTimedInhibitorArc {
  SimpleTimedPlace &inputPlace;
  SimpleTimedTransition &outputTransition;
  uint32_t weight;

  /*SimpleTimedInhibitorArc(const TAPN::InhibitorArc &inhibitorArc) {*/
  /*  // set input place and output transition to default values*/
  /*  weight = inhibitorArc.getWeight();*/
  /*}*/
};

} // namespace Atler
}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_SIMPLETIMEDINHIBITORARC_HPP_ */
