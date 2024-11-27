#ifndef VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_
#define VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_

#include "DiscreteVerification/Atler/SimpleTimedInhibitorArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedInputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedOutputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Atler/SimpleTimedTransportArc.hpp"
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedArcPetriNet {
  Atler::SimpleTimedPlace *places;
  size_t placesLength;
  Atler::SimpleTimedTransition **transitions;
  size_t transitionsLength;
  Atler::SimpleTimedInputArc **inputArcs;
  size_t inputArcsLength;
  Atler::SimpleTimedOutputArc **outputArcs;
  size_t outputArcsLength;
  Atler::SimpleTimedTransportArc **transportArcs;
  size_t transportArcsLength;
  Atler::SimpleTimedInhibitorArc **inhibitorArcs;
  size_t inhibitorArcsLength;
  int maxConstant;
  int gcd;

  void print(std::ostream &out) const;
};

} // namespace Cuda
}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_ */
