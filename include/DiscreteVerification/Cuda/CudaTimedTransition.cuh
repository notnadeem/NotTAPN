#ifndef VERIFYTAPN_ATLER_CUDATIMEDTRANSITION_CUH_
#define VERIFYTAPN_ATLER_CUDATIMEDTRANSITION_CUH_

#include "DiscreteVerification/Atler/SimpleStochasticStructure.hpp"
#include "DiscreteVerification/Cuda/CudaTimedInhibitorArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedInputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedOutputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransportArc.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {
struct CudaTimedTransition {
  int index = 0;

  //Not sure about this
  char* name;
  char* id;

  CudaTimedInputArc** preset = nullptr;
  int presetLength = 0;
  CudaTimedOutputArc** postset = nullptr;
  int postsetLength = 0;
  CudaTimedTransportArc** transportArcs = nullptr;
  int transportArcsLength = 0;
  CudaTimedInhibitorArc** inhibitorArcs = nullptr;
  int inhibitorArcsLength = 0;

  bool untimedPostset = false;
  bool urgent = false;
  bool controllable{};
  std::pair<double, double> _position;
  Atler::SimpleSMC::Distribution _distribution;
  double _weight;
  Atler::SimpleSMC::FiringMode _firingMode = Atler::SimpleSMC::Oldest;

  __host__ __device__ inline int getPresetSize() const {
      return presetLength + transportArcsLength;
  }
};
} // namespace Cuda
} // namespace VerifyTAPN

#endif /* VERIFYTAPN_ATLER_CUDATIMEDTRANSITION_CUH_ */