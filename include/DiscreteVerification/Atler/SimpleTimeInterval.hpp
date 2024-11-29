#ifndef VERIFYTAPN_ATLER_SIMPLETIMEINTERVAL_HPP_
#define VERIFYTAPN_ATLER_SIMPLETIMEINTERVAL_HPP_

#include <cstdlib>
#include <limits>
namespace VerifyTAPN {
namespace Atler {

struct SimpleTimeInterval {
  bool isLowerBoundStrict;
  int lowerBound;
  int upperBound;
  bool isUpperBoundStrict;

  inline bool setUpperBound(int bound, bool isStrict) {
    if (upperBound == bound)
      isUpperBoundStrict |= isStrict;
    else if (upperBound > bound) {
      isUpperBoundStrict = isStrict;
      upperBound = bound;
    }
    if (upperBound < lowerBound)
      return false;
    else
      return true;
  }

  inline bool contains(double number) const {
    return (number >= (double)lowerBound && number <= (double)upperBound) ||
           (std::abs(number - (double)lowerBound) <=
            std::numeric_limits<double>::epsilon() * 4) ||
           (std::abs(number - (double)upperBound) <=
            std::numeric_limits<double>::epsilon() * 4);
  }
};

} // namespace Atler
} // namespace VerifyTAPN

#endif
