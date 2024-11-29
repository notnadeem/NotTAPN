#ifndef VERIFYTAPN_CUDA_RUNGEN_CHU_
#define VERIFYTAPN_CUDA_RUNGEN_CHU_

#include "DiscreteVerification/Atler/SimpleStochasticStructure.hpp" //TODO: Use the Cuda one when you fetch the new structure
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Atler/SimpleTimedOutputArc.hpp"
#include "DiscreteVerification/Cuda/CudaDeque.cuh"
#include "DiscreteVerification/Cuda/CudaDynamicArray.cuh"
#include "DiscreteVerification/Cuda/CudaInterval.cuh"
#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"
#include "DiscreteVerification/Cuda/CudaTimeInterval.cuh"
#include "DiscreteVerification/Cuda/CudaTimedArcPetriNet.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <random>

namespace VerifyTAPN::Cuda {

struct CudaRunResult {
  bool maximal = false;
  CudaTimedArcPetriNet tapn;
  CudaDynamicArray<CudaDynamicArray<Util::CudaInterval>> defaultTransitionIntervals;
  CudaDynamicArray<CudaDynamicArray<Util::CudaInterval>> transitionIntervals;
  CudaDynamicArray<double> *dates_sampled;
  CudaDynamicArray<uint32_t> transitionsStatistics;
  CudaRealMarking *origin = nullptr;
  CudaRealMarking *parent = nullptr;
  double lastDelay = 0;
  double totalTime = 0;
  int totalSteps = 0;

  uint numericPrecision;
  std::ranlux48 rng;

  // add default constructor
  CudaRunResult() {}

  CudaRunResult(CudaTimedArcPetriNet tapn, const unsigned int numericPrecision = 0)
      : tapn(tapn), defaultTransitionIntervals(tapn.transitionsLength), transitionIntervals(tapn.transitionsLength),
        numericPrecision(numericPrecision) {
    std::random_device rd;
    rng = std::ranlux48(rd());
  }

  CudaRunResult(const CudaRunResult &other)
      : tapn(other.tapn), defaultTransitionIntervals(other.defaultTransitionIntervals),
        transitionIntervals(other.transitionIntervals), numericPrecision(other.numericPrecision), rng(other.rng),
        maximal(other.maximal) {
    if (other.parent != nullptr) {
      // Deep clone parent
      parent = other.parent->clone();
    } else {
      parent = nullptr;
    }
    if (other.origin != nullptr) {
      // Deep clone origin
      origin = other.origin->clone();
    } else {
      origin = nullptr;
    }
  }

  CudaRunResult *copy() const {
    CudaRunResult *clone = new CudaRunResult(tapn);
    clone->origin = new CudaRealMarking(*origin);
    clone->numericPrecision = numericPrecision;
    clone->defaultTransitionIntervals = defaultTransitionIntervals;
    clone->reset();
    return clone;
  }

  void prepare(CudaRealMarking initMarking) {
    origin = initMarking.clone();
    parent = initMarking.clone();
    printf("Prepared\n");
    double originMaxDelay = origin->availableDelay();
    CudaDynamicArray<Util::CudaInterval> *invIntervals =
        new CudaDynamicArray<Util::CudaInterval>(tapn.transitionsLength);
    invIntervals->add(Util::CudaInterval(0, originMaxDelay));
    printf("invIntervals size: %d\n", invIntervals->size);

    for (size_t i = 0; i < tapn.transitionsLength; i++) {
      printf("Intersecting transition no: %d\n", i);
      CudaTimedTransition *transition = tapn.transitions[i];
      // print transition length
      printf("Inhibitor length first: %d\n", transition->inhibitorArcsLength);
      printf("Intersecting transition no after: %d\n", i);
      if (transition->presetLength == 0 && printf("first\n"), transition->inhibitorArcs == 0) {
        defaultTransitionIntervals.add(*invIntervals);
      } else {
        printf("Intersecting transition no inside: %d\n", i);
        CudaDynamicArray<Util::CudaInterval> firingDates = transitionFiringDates(*transition);
        defaultTransitionIntervals.add(Util::setIntersection(firingDates, *invIntervals));
        printf("End of Intersection\n");
      }
      printf("End of Intersection 2\n");
    }
    reset();
  }

  void reset() {
    printf("Reset begining\n");
    transitionIntervals = defaultTransitionIntervals;
    // print transition intervals length

    dates_sampled = new CudaDynamicArray<double>(transitionIntervals.size);

    printf("Reset begining 2\n");
    // print dates sampled length
    printf("Dates sampled length: %d\n", dates_sampled->capacity);
    for (size_t i = 0; i < transitionIntervals.size; i++) {
      dates_sampled->add(std::numeric_limits<double>::infinity());
    }
    // print old dates sampled length
    printf("Old dates sampled length: %d\n", dates_sampled->size);
    lastDelay = 0.0;
    totalTime = 0.0;
    totalSteps = 0;
    bool deadlock = true;
    for (size_t i = 0; i < dates_sampled->size; i++) {
      printf("Before %d: \n", i);
      auto intervals = transitionIntervals.get(i);
      // print intervals length
      printf("Intervals length: %d\n", intervals.size);
      // print if intervals is empty
      printf("Intervals empty: %d\n", intervals.empty());
      if (!intervals.empty() && intervals.get(0).lower() == 0) {
        const SimpleSMC::Distribution distribution = tapn.transitions[i]->distribution;
        dates_sampled->set(i, distribution.sample(rng, numericPrecision));
      }
      // print check
      printf("Check %d: \n", i);
      deadlock &= transitionIntervals.get(i).empty() ||
                  (transitionIntervals.get(i).size == 0 && transitionIntervals.get(i).get(0).lower() == 0 &&
                   dates_sampled->get(i) == 0);
    }

    parent->deadlocked = deadlock;
    printf("Reset\n");
  }

  void refreshTransitionsIntervals() {
    double max_delay = parent->availableDelay();
    CudaDynamicArray<Util::CudaInterval> invIntervals(10);
    invIntervals.add(Util::CudaInterval(0, max_delay));
    bool deadlocked = true;

    for (size_t i = 0; i < tapn.transitionsLength; i++) {
      auto transition = tapn.transitions[i];
      int index = transition->index;
      if (transition->getPresetSize() == 0 && transition->inhibitorArcsLength == 0) {
        transitionIntervals.set(i, invIntervals);
      } else {
        CudaDynamicArray<Util::CudaInterval> firingDates = transitionFiringDates(*transition);
        transitionIntervals.set(i, Util::setIntersection(firingDates, invIntervals));
      }
      printf("Transition intervals inter size: %d\n", transitionIntervals.get(i).size);
      bool enabled = (!transitionIntervals.get(i).empty()) && (transitionIntervals.get(i).get(0).lower() == 0);
      bool newlyEnabled = enabled && (dates_sampled->get(i) == std::numeric_limits<double>::infinity());
      bool reachedUpper =
          enabled && !newlyEnabled && (transitionIntervals.get(i).get(0).upper() == 0 && dates_sampled->get(i) > 0);
      if (!enabled || reachedUpper) {
        dates_sampled->set(i, std::numeric_limits<double>::infinity());
      } else if (newlyEnabled) {
        const auto distribution = tapn.transitions[i]->distribution;
        double date = distribution.sample(rng, numericPrecision);
        if (transitionIntervals.get(i).get(0).upper() > 0 || date == 0) {
          dates_sampled->set(i, date);
        }
      }
      deadlocked &= transitionIntervals.get(i).empty() ||
                    (transitionIntervals.get(i).size == 1 && transitionIntervals.get(i).get(0).lower() == 0 &&
                     dates_sampled->get(i) > 0);
    }
    parent->deadlocked = deadlocked;
  }

  CudaRealMarking *next() {
    auto [winner, delay] = getWinnerTransitionAndDelay();

    if (delay == std::numeric_limits<double>::infinity()) {
      // print delay is infinity
      maximal = true;
      return nullptr;
    }

    parent->deltaAge(delay);
    totalTime += delay;

    parent->fromDelay = delay + parent->fromDelay;

    if (winner != nullptr) {
      printf("Winner: %s\n", winner->name);
      totalSteps++;
      dates_sampled->set(winner->index, std::numeric_limits<double>::infinity());
      auto child = fire(winner);
      child->generatedBy = winner;
      delete parent;
      parent = child;
    }

    for (size_t i = 0; i < transitionIntervals.size; i++) {
      double date = dates_sampled->get(i);
      double newVal =
          (date == std::numeric_limits<double>::infinity()) ? std::numeric_limits<double>::infinity() : date - delay;
    }

    refreshTransitionsIntervals();
    return parent;
  }

  std::pair<CudaTimedTransition *, double> getWinnerTransitionAndDelay() {
    CudaDynamicArray<size_t> winner_indexes(10);
    double date_min = std::numeric_limits<double>::infinity();
    // print transition intervals length
    printf("Transition intervals length: %d\n", transitionIntervals.size);

    for (size_t i = 0; i < transitionIntervals.size; i++) {
      auto intervals = transitionIntervals.get(i);
      printf("Intervals length: %d\n", intervals.size);
      if (intervals.empty()) continue;
      double date = std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < intervals.size; j++) {
        // print the word before
        printf("Before %d: ", j);
        auto interval = intervals.get(j);
        printf("After %d: ", j);
        // print after
        if (interval.lower() > 0) {
          date = interval.lower();
          break;
        }
        if (interval.upper() > date) {
          date = interval.upper();
          break;
        }
      }
      printf("Date: %f\n", date);
      // print length of dates sampled
      printf("Dates sampled length: %d\n", dates_sampled->size);
      date = std::min(dates_sampled->get(i), date);
      if (date < date_min) {
        date_min = date;
        winner_indexes.clear();
      }
      if (dates_sampled->get(i) == date_min) {
        winner_indexes.add(i);
      }
    }
    printf("Winner indexes size: \n");
    CudaTimedTransition *winner;
    if (winner_indexes.empty()) {
      winner = nullptr;
    } else if (winner_indexes.size == 1) {
      winner = tapn.transitions[winner_indexes.get(0)];
    } else {
      winner = chooseWeightedWinner(winner_indexes);
    }
    return std::make_pair(winner, date_min);
  }

  CudaTimedTransition *chooseWeightedWinner(const CudaDynamicArray<size_t> winner_indexes) {
    double total_weight = 0;
    CudaDynamicArray<size_t> infinite_weights(10);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      double priority = tapn.transitions[candidate]->_weight;
      if (priority == std::numeric_limits<double>::infinity()) {
        infinite_weights.add(candidate);
      } else {
        total_weight += priority;
      }
    }

    if (!infinite_weights.empty()) {
      int winner_index = std::uniform_int_distribution<>(0, infinite_weights.size - 1)(rng);
      return tapn.transitions[infinite_weights.get(winner_index)];
    }
    if (total_weight == 0) {
      int winner_index = std::uniform_int_distribution<>(0, winner_indexes.size - 1)(rng);
      return tapn.transitions[winner_indexes.get(winner_index)];
    }
    double winning_weight = std::uniform_real_distribution<>(0, total_weight)(rng);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      CudaTimedTransition *transition = tapn.transitions[candidate];
      winning_weight -= transition->_weight;
      if (winning_weight <= 0) {
        return transition;
      }
    }

    return tapn.transitions[winner_indexes.get(0)];
  }

  CudaDynamicArray<Util::CudaInterval> transitionFiringDates(const CudaTimedTransition &transition) {
    auto firingIntervals = CudaDynamicArray<Util::CudaInterval>(10);
    firingIntervals.add(Util::CudaInterval(0, HUGE_VAL));
    auto disabled = CudaDynamicArray<Util::CudaInterval>();

    // for each inhibitor arc
    for (size_t i = 0; i < transition.inhibitorArcsLength; i++) {
      printf("Inhibitor arc length: %d\n", transition.inhibitorArcsLength);
      printf("Preset arc length: %d\n", transition.presetLength);
      auto inhib = transition.inhibitorArcs[i];
      printf("Inhibitor arc 2\n");
      if (parent->numberOfTokensInPlace(inhib->inputPlace.index) >= inhib->weight) {
        printf("Inhibitor arc 3\n");
        return disabled;
      }
    }

    for (size_t i = 0; i < transition.presetLength; i++) {
      auto arc = transition.preset[i];
      auto place = parent->places[arc->inputPlace.index];

      printf("Preset arc\n");
      printf("place name: %s\n", place.place.name);
      printf("place size: %d\n", place.tokens.size);
      printf("place capacity: %d\n", place.tokens.capacity);
      if (place.isEmpty()) {
        return disabled;
      }
      printf("Preset arc 2\n");
      firingIntervals =
          Util::setIntersection(firingIntervals, arcFiringDates(arc->interval, arc->weight, place.tokens));
      if (firingIntervals.empty()) return firingIntervals;
    }

    for (size_t i = 0; i < transition.transportArcsLength; i++) {
      printf("Transport arc\n");
      auto transport = transition.transportArcs[i];
      auto &place = parent->places[transport->source.index];
      if (place.isEmpty()) return disabled;

      Atler::SimpleTimeInvariant targetInvariant = transport->destination.timeInvariant;
      CudaTimeInterval arcInterval = transport->interval;
      if (targetInvariant.bound < arcInterval.upperBound) {
        arcInterval.setUpperBound(targetInvariant.bound, targetInvariant.isBoundStrict);
      }
      firingIntervals =
          Util::setIntersection(firingIntervals, arcFiringDates(arcInterval, transport->weight, place.tokens));
      if (firingIntervals.empty()) return firingIntervals;
    }
    printf("Firing in the hole\n");

    return firingIntervals;
  }

  CudaDynamicArray<Util::CudaInterval> arcFiringDates(CudaTimeInterval time_interval, uint32_t weight,
                                                          CudaDynamicArray<CudaRealToken *> tokens) {

    printf("Arc firing dates\n");
    Util::CudaInterval arcInterval(time_interval.lowerBound, time_interval.upperBound);
    size_t total_tokens = 0;
    printf("Tokens size: %d\n", tokens.size);
    for (size_t i = 0; i < tokens.size; i++) {
      total_tokens += tokens.get(i)->count;
    }
    printf("Arc firing dates 2\n");
    if (total_tokens < weight) return CudaDynamicArray<Util::CudaInterval>();

    CudaDynamicArray<Util::CudaInterval> firingIntervals(10);
    CudaDeque<double> selected = CudaDeque<double>();
    for (size_t i = 0; i < tokens.size; i++) {
      // print the counts from all the tokens
      // print size of tokens
      printf("Tokens z size: %d\n", tokens.size);
      printf("Tokens get z count: %d\n", tokens.get(i)->count);
    }
    for (size_t i = 0; i < tokens.size; i++) {
      for (int j = 0; j < tokens.get(i)->count; j++) {
        // print tokens get i count
        printf("Tokens get i count: %d\n", tokens.get(i)->count);
        printf("Tokens get i age: %f\n", tokens.get(i)->age);
        printf("Before push_back\n");
        selected.push_back(tokens.get(i)->age);
        if (selected.size > weight) {
          selected.pop_front();
        }
        if (selected.size == weight) {
          Util::CudaInterval tokenSetInterval(0, std::numeric_limits<double>::infinity());
          for (size_t k = 0; k < selected.size; k++) {
            tokenSetInterval = Util::intersect(tokenSetInterval, Util::CudaInterval(selected.at(k), selected.at(k)));
          }
          Util::setAdd(firingIntervals, tokenSetInterval);
        }
      }
    }
    return firingIntervals;
  }

  CudaDynamicArray<CudaRealToken *> removeRandom(CudaDynamicArray<CudaRealToken *> &tokenList,
                                                     const CudaTimeInterval &interval, const int weight) {
    printf("Remove random method is being called\n");
    auto res = CudaDynamicArray<CudaRealToken *>(tokenList.size);
    int remaning = weight;
    std::uniform_int_distribution<> randomTokenIndex(0, tokenList.size - 1);
    size_t tok_index = randomTokenIndex(rng);
    size_t tested = 0;

    while (remaning > 0 && tested < tokenList.size) {
      CudaRealToken *token = tokenList.get(tok_index);
      if (interval.contains(token->age)) {
        res.add(new CudaRealToken{.age = token->age, .count = 1});
        remaning--;
        tokenList.get(tok_index)->remove(1);
        if (tokenList.get(tok_index)->count == 0) {
          tokenList.remove(tok_index);
          randomTokenIndex = std::uniform_int_distribution<>(0, tokenList.size - 1);
        }
        if (remaning > 0) {
          tok_index = randomTokenIndex(rng);
          tested = 0;
        }
      } else {
        tok_index = (tok_index + 1) % tokenList.size;
        tested++;
      }
    }
    assert(remaning == 0);
    return res;
  }

  CudaDynamicArray<CudaRealToken *> removeYoungest(CudaDynamicArray<CudaRealToken *> &tokenList,
                                                       const CudaTimeInterval &interval, const int weight) {
    printf("Remove youngest method is being called\n");

    auto res = CudaDynamicArray<CudaRealToken *>();
    int remaining = weight;
    for (size_t i = 0; i < tokenList.size; i++) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!interval.contains(age)) {
        continue;
      }
      int count = token->count;
      if (count >= remaining) {
        res.add(new CudaRealToken{.age = age, .count = count});
        token->remove(remaining);
        if (token->count == 0) tokenList.remove(i);
        remaining = 0;
        break;
      } else {
        res.add(new CudaRealToken{.age = age, .count = count});
        remaining -= count;
        tokenList.remove(i);
      }
    }

    assert(remaining == 0);
    return res;
  }

  // NOTE: Double check this method to ensure it is correct
  CudaDynamicArray<CudaRealToken *> removeOldest(CudaDynamicArray<CudaRealToken *> &tokenList,
                                                     const CudaTimeInterval &timeInterval, const int weight) {

    auto res = CudaDynamicArray<CudaRealToken *>();
    int remaining = weight;
    // for loop in reverse order
    for (int i = tokenList.size - 1; i >= 0; i--) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!timeInterval.contains(age)) {
        continue;
      }
      int count = token->count;
      if (count >= remaining) {
        res.add(new CudaRealToken{.age = age, .count = count});
        token->remove(remaining);
        if (token->count == 0) tokenList.remove(i);
        remaining = 0;
        break;
      } else {
        res.add(new CudaRealToken{.age = age, .count = count});
        remaining -= count;
        tokenList.remove(i);
      }
    }
    assert(remaining == 0);
    return res;
  }

  CudaRealMarking *fire(CudaTimedTransition *transition) {
    if (transition == nullptr) {
      assert(false);
      return nullptr;
    }

    CudaRealMarking *child = parent->clone();
    CudaRealPlace *placeList = child->places;

    for (size_t i = 0; i < transition->presetLength; i++) {
      Atler::SimpleTimedInputArc *input = transition->preset[i];
      CudaRealPlace place = placeList[transition->preset[i]->inputPlace.index];
      CudaDynamicArray<CudaRealToken *> &tokenList = place.tokens;
      switch (transition->_firingMode) {
      case SimpleSMC::FiringMode::Random:
        removeRandom(tokenList, input->interval, input->weight);
        break;
      case SimpleSMC::FiringMode::Oldest:
        removeOldest(tokenList, input->interval, input->weight);
        break;
      case SimpleSMC::FiringMode::Youngest:
        removeYoungest(tokenList, input->interval, input->weight);
        break;
      default:
        removeOldest(tokenList, input->interval, input->weight);
        break;
      }

      auto toCreate = CudaDynamicArray<std::pair<Atler::SimpleTimedPlace *, CudaRealToken *>>(10);
      for (size_t i = 0; i < transition->transportArcsLength; i++) {
        auto transport = transition->transportArcs[i];
        int destInv = transport->destination.timeInvariant.bound;
        CudaRealPlace place = placeList[transport->source.index];
        CudaDynamicArray<CudaRealToken *> &tokenList = place.tokens;
        CudaDynamicArray<CudaRealToken *> consumed(10);
        CudaTimeInterval &arcInterval = transport->interval;
        if (destInv < arcInterval.upperBound) arcInterval.setUpperBound(destInv, false);
        switch (transition->_firingMode) {
        case SimpleSMC::FiringMode::Random:
          consumed = removeRandom(tokenList, arcInterval, transport->weight);
          break;
        case SimpleSMC::FiringMode::Oldest:
          consumed = removeOldest(tokenList, arcInterval, transport->weight);
          break;
        case SimpleSMC::FiringMode::Youngest:
          consumed = removeYoungest(tokenList, arcInterval, transport->weight);
          break;
        default:
          consumed = removeOldest(tokenList, arcInterval, transport->weight);
          break;
        }
        for (size_t j = 0; j < consumed.size; j++) {
          toCreate.add(std::make_pair(&(transport->destination), consumed.get(j)));
        }
      }

      for (size_t i = 0; i < transition->postsetLength; i++) {
        Atler::SimpleTimedPlace &place = transition->postset[i]->outputPlace;
        Atler::SimpleTimedOutputArc *post = transition->postset[i];
        auto token = CudaRealToken{.age = 0.0, .count = static_cast<int>(post->weight)};
        child->addTokenInPlace(place, token);
      }
      for (size_t i = 0; i < toCreate.size; i++) {
        auto [place, token] = toCreate.get(i);
        child->addTokenInPlace(*place, *token);
      }
    }
    return child;
  }
};
} // namespace VerifyTAPN::Atler

#endif
