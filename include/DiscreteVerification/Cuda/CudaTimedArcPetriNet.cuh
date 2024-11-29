#ifndef VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_
#define VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_

#include "DiscreteVerification/Atler/SimpleTimedInputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedOutputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"
#include "DiscreteVerification/Cuda/CudaTimedInhibitorArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransportArc.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedArcPetriNet {
  Atler::SimpleTimedPlace *places;
  size_t placesLength;
  CudaTimedTransition **transitions;
  size_t transitionsLength;
  Atler::SimpleTimedInputArc **inputArcs;
  size_t inputArcsLength;
  Atler::SimpleTimedOutputArc **outputArcs;
  size_t outputArcsLength;
  CudaTimedTransportArc **transportArcs;
  size_t transportArcsLength;
  CudaTimedInhibitorArc **inhibitorArcs;
  size_t inhibitorArcsLength;
  int maxConstant;
  int gcd;

  void print(std::ostream &out) const;
};

} // namespace Cuda
}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_ */
