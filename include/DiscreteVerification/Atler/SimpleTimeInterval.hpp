#ifndef VERIFYTAPN_ATLER_SIMPLETIMEINTERVAL_HPP_
#define VERIFYTAPN_ATLER_SIMPLETIMEINTERVAL_HPP_

namespace VerifyTAPN {
namespace Atler {

struct SimpleTimeInterval {
  bool isLowerBoundStrict;
  int lowerBound;
  int upperBound;
  bool isUpperBoundStrict;

  inline bool setUpperBound(int bound, bool isStrict) {
      if (upperBound == bound) isUpperBoundStrict |= isStrict;
      else if (upperBound > bound) {
          isUpperBoundStrict = isStrict;
          upperBound = bound;
      }
      if (upperBound < lowerBound) return false;
      else return true;
  }

};

} // namespace Atler
} // namespace VerifyTAPN

#endif
