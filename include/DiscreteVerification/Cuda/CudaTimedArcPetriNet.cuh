#ifndef VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_
#define VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CUH_

#include "DiscreteVerification/Cuda/CudaTimedInputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedOutputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedPlace.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"
#include "DiscreteVerification/Cuda/CudaTimedInhibitorArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransportArc.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedArcPetriNet {
  CudaTimedPlace **places;
  size_t placesLength;
  CudaTimedTransition **transitions;
  size_t transitionsLength;
  CudaTimedInputArc **inputArcs;
  size_t inputArcsLength;
  CudaTimedOutputArc **outputArcs;
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
