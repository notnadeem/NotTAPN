#ifndef VERIFYTAPN_ATLER_CUDATIMEDOUTPUTARC_HPP_
#define VERIFYTAPN_ATLER_CUDATIMEDOUTPUTARC_HPP_

#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"
#include "DiscreteVerification/Cuda/CudaTimedPlace.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedOutputArc {
  CudaTimedTransition *inputTransition;
  CudaTimedPlace *outputPlace;
  const uint32_t weight;
};

} // namespace Atler
}; // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_CUDATIMEDOUTPUTARC_HPP_ */
