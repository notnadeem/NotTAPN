#ifndef VERIFYYAPN_ATLER_CUDATIMEDPLACE_HPP_
#define VERIFYYAPN_ATLER_CUDATIMEDPLACE_HPP_

#include "DiscreteVerification/Atler/SimpleTimeInvariant.hpp"
#include "DiscreteVerification/Cuda/CudaTimedInhibitorArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedInputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedOutputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransportArc.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

struct CudaTimedInputArc;
struct CudaTimedInhibitorArc;
struct CudaTimedInputArc;

enum PlaceType { Inv, Dead, Std };

struct CudaTimedPlace {
  int index;
  PlaceType type;

  const char *name;
  size_t nameLength;

  const char *id;
  size_t idLength;

  Atler::SimpleTimeInvariant timeInvariant;
  bool untimed;
  int maxConstant;
  bool containsInhibitorArcs;
  CudaTimedInputArc **inputArcs;
  size_t inputArcsLength;
  CudaTimedTransportArc **transportArcs;
  size_t transportArcsLength;
  CudaTimedTransportArc **prodTransportArcs;
  size_t prodTransportArcsLength;
  CudaTimedInhibitorArc **inhibitorArcs;
  size_t inhibitorArcsLength;
  CudaTimedOutputArc **outputArcs;
  size_t outputArcsLength;
  std::pair<double, double> _position;
};

} // namespace Atler
}; // namespace VerifyTAPN

#endif /* VERIFYYAPN_ATLER_SIMPLETIMEDPLACE_HPP_ */