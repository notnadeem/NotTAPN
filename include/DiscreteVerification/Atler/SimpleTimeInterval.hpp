#ifndef VERIFYTAPN_ATLER_SIMPLETIMEINTERVAL_HPP_
#define VERIFYTAPN_ATLER_SIMPLETIMEINTERVAL_HPP_

#include "Core/TAPN/TimeInterval.hpp"

namespace VerifyTAPN {
namespace Atler {

struct SimpleTimeInterval {
  bool isLowerBoundStrict;
  int lowerBound;
  int upperBound;
  bool isUpperBoundStrict;
};

} // namespace Atler
} // namespace VerifyTAPN

#endif
