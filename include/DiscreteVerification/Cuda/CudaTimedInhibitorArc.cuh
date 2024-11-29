#ifndef VERIFYYAPN_ATLER_CUDATIMEDINHIBITORARC_CUH_
#define VERIFYYAPN_ATLER_CUDATIMEDINHIBITORARC_CUH_

#include "Core/TAPN/InhibitorArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedTransition;
struct SimpleTimedPlace;

struct CudaTimedInhibitorArc {
  SimpleTimedPlace &inputPlace;
  CudaTimedTransition &outputTransition;
  uint32_t weight;

  /*SimpleTimedInhibitorArc(const TAPN::InhibitorArc &inhibitorArc) {*/
  /*  // set input place and output transition to default values*/
  /*  weight = inhibitorArc.getWeight();*/
  /*}*/
};

} // namespace Cuda
}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_CUDATIMEDINHIBITORARC_CUH_ */
