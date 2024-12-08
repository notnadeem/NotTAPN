#ifndef VERIFYTAPN_ATLER_CUDATIMEDTRANSPORTARC_CUH_
#define VERIFYTAPN_ATLER_CUDATIMEDTRANSPORTARC_CUH_

#include "DiscreteVerification/Cuda/CudaTimeInterval.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

//MAKE THIS 
class CudaTimedPlace;

struct CudaTimedTransportArc {
  CudaTimeInterval interval;
  CudaTimedPlace *source;
  CudaTimedTransition *transition;
  CudaTimedPlace *destination;
  const uint32_t weight;
};

} // namespace Cuda
} // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_CUDATIMEDTRANSPORTARC_CUH_ */
