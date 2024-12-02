#ifndef SIMPLEINTERVAL_HPP_
#define SIMPLEINTERVAL_HPP_

#include "DiscreteVerification/Atler/SimpleDynamicArray.hpp"
#include <algorithm>
#include <limits>
#include <iostream>

// TODO: Change numeric_limits to the cuda/thrust ones

namespace VerifyTAPN {
namespace Atler {
namespace Util {

struct SimpleInterval {
  double low, high;

  static double boundUp() {
    return std::numeric_limits<double>::has_infinity
               ? std::numeric_limits<double>::infinity()
               : std::numeric_limits<double>::max();
  }

  static double boundDown() {
    return std::numeric_limits<double>::has_infinity
               ? (-std::numeric_limits<double>::infinity())
               : std::numeric_limits<double>::min();
  }

  static double epsilon() {
    return std::numeric_limits<double>::is_integer
               ? 1
               : std::numeric_limits<double>::epsilon();
  }

  SimpleInterval() : low(0), high(0) {}
  SimpleInterval(double l, double h) : low(l), high(h) {
    if (low > high) {
      low = boundUp();
      high = boundDown();
    }
  };

  auto upper() const { return high; }

  auto lower() const { return low; }

  auto empty() const { return high < low; }

  auto length() const {
    if (low == boundDown() || high == boundUp()) {
      return boundUp();
    }
    if (empty())
      return (double)0;
    return high - low;
  }

  void delta(double dx) {
    if (empty())
      return;
    if (low != boundDown()) {
      low += dx;
    }
    if (high != boundUp()) {
      high += dx;
    }
  }

  SimpleInterval positive() {
    if (empty() || high < 0)
      return SimpleInterval((double)1, 0);
    return SimpleInterval(std::max(low, (double)0), high);
  }
};

inline SimpleInterval intersect(const SimpleInterval &l, const SimpleInterval r) {
  if (l.empty())
    return l;
  if (r.empty())
    return r;
  SimpleInterval n(std::max(l.low, r.low), std::min(l.high, r.high));
  return n;
}

inline SimpleInterval hull(const SimpleInterval &l, const SimpleInterval r) {
  return SimpleInterval(std::min(l.low, r.low), std::max(l.high, r.high));
}

inline bool overlap(const SimpleInterval &l, const SimpleInterval r) {
  auto i = intersect(l, r);
  return !i.empty();
}

// Fix both setAdd and setIntersection later
inline void setAdd(SimpleDynamicArray<SimpleInterval> &first, const SimpleInterval &element) {
    for (unsigned int i = 0; i < first.size; i++) {

        if (element.upper() < first.get(i).lower()) {
            //Add element
            first.insert2(i, element);
            return;
        } else if (overlap(element, first.get(i))) {
            SimpleInterval u = hull(element, first.get(i));
            // Merge with node
            first.set(i, u);
            // Clean up
            for (i++; i < first.size; i++) {
                if (first.get(i).lower() <= u.upper()) {
                    // Merge items
                    if (first.get(i).upper() > u.upper()) {
                        first.set(i-1, SimpleInterval(first.get(i - 1).lower(), first.get(i).upper()));
                    }
                    first.remove(i);
                    i--;
                }
            }
            return;
        }
    }
    first.add(element);
}

inline SimpleDynamicArray<SimpleInterval> setIntersection(const SimpleDynamicArray<SimpleInterval>& first, const SimpleDynamicArray<SimpleInterval>& second) {
    SimpleDynamicArray<SimpleInterval> result = SimpleDynamicArray<SimpleInterval>(first.size + second.size);

    if (first.empty() || second.empty()) {
        return result;
    }

    unsigned int i = 0, j = 0;

    while (i < first.size && j < second.size) {
        double i1up = first.get(i).upper();
        double i2up = second.get(j).upper();

        SimpleInterval intersection = intersect(first.get(i), second.get(j));

        if (!intersection.empty()) {
            result.add(intersection);
        }

        if (i1up <= i2up) {
            i++;
        }

        if (i2up <= i1up) {
            j++;
        }
    }
    return result;
}

} /* namespace Util */
} // namespace Atler
} /* namespace VerifyTAPN */
#endif /* INTERVALOPS_HPP_ */
