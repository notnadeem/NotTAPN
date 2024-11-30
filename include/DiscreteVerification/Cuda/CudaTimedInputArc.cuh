#ifndef VERIFYTAPN_ATLER_CUDATIMEDINPUTARC_CUH_
#define VERIFYTAPN_ATLER_CUDATIMEDINPUTARC_CUH_

#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"
#include "DiscreteVerification/Cuda/CudaTimeInterval.cuh"
#include "DiscreteVerification/Cuda/CudaTimedPlace.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedInputArc {
  CudaTimeInterval interval;
  CudaTimedPlace &inputPlace;
  CudaTimedTransition &outputTransition;
  const uint32_t weight;
};

} // namespace Cuda
}; // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_CUDATIMEDINPUTARC_CUH_ */