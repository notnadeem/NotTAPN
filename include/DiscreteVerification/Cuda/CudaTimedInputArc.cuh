#ifndef VERIFYTAPN_ATLER_CUDATIMEDINPUTARC_HPP_
#define VERIFYTAPN_ATLER_CUDATIMEDINPUTARC_HPP_

#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"
#include "DiscreteVerification/Cuda/CudaTimeInterval.cuh"
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedInputArc {
  CudaTimeInterval interval;
  Atler::SimpleTimedPlace &inputPlace;
  CudaTimedTransition &outputTransition;
  const uint32_t weight;
};

} // namespace Atler
}; // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_CUDATIMEDINPUTARC_HPP_ */