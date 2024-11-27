#ifndef VERIFYTAPN_ATLER_SIMPLETIMEDTRANSITION_CUH_
#define VERIFYTAPN_ATLER_SIMPLETIMEDTRANSITION_CUH_

#include "DiscreteVerification/Atler/SimpleStochasticStructure.hpp"
#include "DiscreteVerification/Atler/SimpleTimedInhibitorArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedInputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedOutputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedTransportArc.hpp"
#include <string>

#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {
struct SimpleTimedTransition {
  int index = 0;

  //Not sure about this
  char* name;
  char* id;

  Atler::SimpleTimedInputArc** preset = nullptr;
  int presetLength = 0;
  Atler::SimpleTimedOutputArc** postset = nullptr;
  int postsetLength = 0;
  Atler::SimpleTimedTransportArc** transportArcs = nullptr;
  int transportArcsLength = 0;
  Atler::SimpleTimedInhibitorArc** inhibitorArcs = nullptr;
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

#endif /* VERIFYTAPN_ATLER_SIMPLETIMEDTRANSITION_CUH_ */