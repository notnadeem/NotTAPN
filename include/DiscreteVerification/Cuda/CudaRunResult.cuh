#ifndef VERIFYTAPN_CUDA_RUNGEN_CHU_
#define VERIFYTAPN_CUDA_RUNGEN_CHU_

#include "DiscreteVerification/Cuda/CudaDeque.cuh"
#include "DiscreteVerification/Cuda/CudaDynamicArray.cuh"
#include "DiscreteVerification/Cuda/CudaInterval.cuh"
#include "DiscreteVerification/Cuda/CudaPair.cuh"
#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"
#include "DiscreteVerification/Cuda/CudaStochasticStructure.cuh"
#include "DiscreteVerification/Cuda/CudaTimeInterval.cuh"
#include "DiscreteVerification/Cuda/CudaTimedArcPetriNet.cuh"
#include "DiscreteVerification/Cuda/CudaTimedOutputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedPlace.cuh"
#include "DiscreteVerification/Cuda/CudaTimedTransition.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
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
  curandState_t *rngStates;

  // add default constructor
  __host__ CudaRunResult() {}

  __host__ CudaRunResult(CudaTimedArcPetriNet tapn)
      : tapn(tapn), defaultTransitionIntervals(tapn.transitionsLength), transitionIntervals(tapn.transitionsLength),
        numericPrecision(numericPrecision) {
  }

  // Change this to cuda if needed
  //  CudaRunResult(const CudaRunResult &other)
  //      : tapn(other.tapn), defaultTransitionIntervals(other.defaultTransitionIntervals),
  //        transitionIntervals(other.transitionIntervals), numericPrecision(other.numericPrecision), rng(other.rng),
  //        maximal(other.maximal) {
  //    if (other.parent != nullptr) {
  //      // Deep clone parent
  //      parent = other.parent->clone();
  //    } else {
  //      parent = nullptr;
  //    }
  //    if (other.origin != nullptr) {
  //      // Deep clone origin
  //      origin = other.origin->clone();
  //    } else {
  //      origin = nullptr;
  //    }
  //  }

  __host__ ~CudaRunResult() {
    cudaError_t freeStatus = cudaFree(rngStates);

    if (freeStatus != cudaSuccess) {
      std::cerr << "cudaFree failed: " << cudaGetErrorString(freeStatus) << std::endl;
    } else {
      std::cout << "Device memory freed successfully." << std::endl;
    }
  }

  __device__ CudaRunResult *copy(int tid) const {
    CudaRunResult *clone = new CudaRunResult(tapn);
    clone->origin = new CudaRealMarking(*origin);
    clone->numericPrecision = numericPrecision;
    clone->defaultTransitionIntervals = defaultTransitionIntervals;
    clone->reset(tid);
    return clone;
  }

  __device__ void prepare(CudaRealMarking initMarking, int tid) {
    origin = new CudaRealMarking(initMarking);
    parent = new CudaRealMarking(initMarking);
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
        for (size_t i = 0; i < firingDates.size; i++) {
          printf("Firing date: %d, %d \n", firingDates.get(i).lower(), firingDates.get(i).upper());
        }
        defaultTransitionIntervals.add(Util::setIntersection(firingDates, *invIntervals));
        printf("End of Intersection\n");
      }
      printf("End of Intersection 2\n");
    }
    reset(tid);
  }

  __device__ void reset(int tid) {
    printf("Reset begining\n");
    transitionIntervals = defaultTransitionIntervals;
    // print transition intervals length

    dates_sampled = new CudaDynamicArray<double>(transitionIntervals.size);

    printf("Reset begining 2\n");
    // print dates sampled length
    printf("Dates sampled length: %d\n", dates_sampled->capacity);
    for (size_t i = 0; i < transitionIntervals.size; i++) {
      dates_sampled->add(HUGE_VAL);
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
        const CudaSMC::Distribution distribution = tapn.transitions[i]->distribution;
        dates_sampled->set(i, distribution.sample(&rngStates[tid], numericPrecision));
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

  __device__ void refreshTransitionsIntervals(int tid) {
    double max_delay = parent->availableDelay();
    CudaDynamicArray<Util::CudaInterval> invIntervals(10);
    invIntervals.add(Util::CudaInterval(0, max_delay));
    bool deadlocked = true;

    for (size_t i = 0; i < tapn.transitionsLength; i++) {
      auto transition = tapn.transitions[i];
      int index = transition->index;
      if (transition->getPresetSize() == 0 && transition->inhibitorArcsLength == 0) {
        transitionIntervals.set(index, invIntervals);
      } else {
        CudaDynamicArray<Util::CudaInterval> firingDates = transitionFiringDates(*transition);
        transitionIntervals.set(i, Util::setIntersection(firingDates, invIntervals));
      }
      printf("Transition intervals inter size: %d\n", transitionIntervals.get(i).size);
      bool enabled = (!transitionIntervals.get(i).empty()) && (transitionIntervals.get(i).get(0).lower() == 0);
      bool newlyEnabled = enabled && (dates_sampled->get(i) == HUGE_VAL);
      bool reachedUpper =
          enabled && !newlyEnabled && (transitionIntervals.get(i).get(0).upper() == 0) && (dates_sampled->get(i) > 0);
      if (!enabled || reachedUpper) {
        dates_sampled->set(i, HUGE_VAL);
      } else if (newlyEnabled) {
        const auto distribution = tapn.transitions[i]->distribution;
        double date = distribution.sample(&rngStates[tid], numericPrecision);
        // printf("Transition intervals inter size: %d\n", transitionIntervals.get(i).size);
        printf("Transition intervals get i 0.lower(): %d\n", transitionIntervals.get(i).get(0).lower());
        printf("Transition intervals get i 0.upper(): %d\n", transitionIntervals.get(i).get(0).upper());
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

  __device__ CudaRealMarking *next(int tid) {
    auto [winner, delay] = getWinnerTransitionAndDelay(tid);

    if (delay == HUGE_VAL) {
      // print delay is infinity
      printf("Delay is infinity/Deadlocked\n");
      maximal = true;
      return nullptr;
    }

    parent->deltaAge(delay);
    totalTime += delay;

    parent->fromDelay = delay + parent->fromDelay;

    if (winner != nullptr) {
      printf("Winner: %s\n", winner->name);
      totalSteps++;
      dates_sampled->set(winner->index, HUGE_VAL);
      auto child = fire(winner, tid);
      child->generatedBy = winner;
      delete parent;
      parent = child;
    }

    for (size_t i = 0; i < transitionIntervals.size; i++) {
      double date = dates_sampled->get(i);
      double newVal = (date == HUGE_VAL) ? HUGE_VAL : date - delay;

      if (date != HUGE_VAL) {
        printf("Date: %d\n", date);
        printf("newVal: %d\n", newVal);
      }
      dates_sampled->set(i, newVal);
    }

    refreshTransitionsIntervals(tid);
    return parent;
  }

  __device__ CudaPair<CudaTimedTransition *, double> getWinnerTransitionAndDelay(int tid) {
    CudaDynamicArray<size_t> winner_indexes(10);
    double date_min = HUGE_VAL;
    // print transition intervals length
    printf("Transition intervals length: %d\n", transitionIntervals.size);

    for (size_t i = 0; i < transitionIntervals.size; i++) {
      auto intervals = transitionIntervals.get(i);
      printf("Intervals length: %d\n", intervals.size);
      if (intervals.empty()) continue;
      double date = HUGE_VAL;
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
      date = fmin(dates_sampled->get(i), date);
      if (date < date_min) {
        printf("New minimum date: %f\n", date);
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
      winner = chooseWeightedWinner(winner_indexes, tid);
    }
    return makeCudaPair(winner, date_min);
  }

  __device__ CudaTimedTransition *chooseWeightedWinner(const CudaDynamicArray<size_t> winner_indexes, int tid) {
    double total_weight = 0;
    CudaDynamicArray<size_t> infinite_weights(10);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      double priority = tapn.transitions[candidate]->_weight;
      if (priority == HUGE_VAL) {
        infinite_weights.add(candidate);
      } else {
        total_weight += priority;
      }
    }

    if (!infinite_weights.empty()) {
      int winner_index = CudaSMC::getRandomTokenIndex(&rngStates[tid], infinite_weights.size - 1);
      return tapn.transitions[infinite_weights.get(winner_index)];
    }
    if (total_weight == 0) {
      int winner_index = CudaSMC::getRandomTokenIndex(&rngStates[tid], winner_indexes.size - 1);
      return tapn.transitions[winner_indexes.get(winner_index)];
    }
    double winning_weight = CudaSMC::getRandomTokenIndex(&rngStates[tid], total_weight);
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

  __host__ __device__ CudaDynamicArray<Util::CudaInterval>
  transitionFiringDates(const CudaTimedTransition &transition) {
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
      CudaRealPlace place = parent->places[arc->inputPlace.index];

      printf("Preset arc\n");
      printf("place name: %s\n", place.place.name);
      printf("place size: %d\n", place.tokens->size);
      printf("place capacity: %d\n", place.tokens->capacity);
      if (place.isEmpty()) {
        return disabled;
      }
      printf("Preset arc 2\n");
      firingIntervals =
          Util::setIntersection(firingIntervals, arcFiringDates(arc->interval, arc->weight, *place.tokens));
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
          Util::setIntersection(firingIntervals, arcFiringDates(arcInterval, transport->weight, *place.tokens));
      if (firingIntervals.empty()) return firingIntervals;
    }
    printf("Firing in the hole\n");

    return firingIntervals;
  }

  __host__ __device__ CudaDynamicArray<Util::CudaInterval>
  arcFiringDates(CudaTimeInterval time_interval, uint32_t weight, CudaDynamicArray<CudaRealToken *> &tokens) {

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
          Util::CudaInterval tokenSetInterval(0, HUGE_VAL);
          for (size_t k = 0; k < selected.size; k++) {
            Util::CudaInterval shifted = arcInterval;
            shifted.delta(-selected.at(k));
            tokenSetInterval = Util::intersect(tokenSetInterval, shifted);
          }
          Util::setAdd(firingIntervals, tokenSetInterval);
        }
      }
    }
    return firingIntervals;
  }

  __device__ CudaDynamicArray<CudaRealToken *> removeRandom(CudaDynamicArray<CudaRealToken *> tokenList,
                                                            const CudaTimeInterval &interval, const int weight,
                                                            int tid) {
    printf("Remove random method is being called\n");
    auto res = CudaDynamicArray<CudaRealToken *>(tokenList.size);
    int remaning = weight;

    size_t tok_index = CudaSMC::getRandomTokenIndex(&rngStates[tid], tokenList.size);
    size_t tested = 0;

    while (remaning > 0 && tested < tokenList.size) {
      CudaRealToken *token = tokenList.get(tok_index);
      if (interval.contains(token->age)) {
        res.add(new CudaRealToken{.age = token->age, .count = 1});
        remaning--;
        tokenList.get(tok_index)->remove(1);
        if (tokenList.get(tok_index)->count == 0) {
          tokenList.remove(tok_index);
        }
        if (remaning > 0) {
          tok_index = CudaSMC::getRandomTokenIndex(&rngStates[tid], tokenList.size);
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

  __device__ CudaDynamicArray<CudaRealToken *> removeYoungest(CudaDynamicArray<CudaRealToken *> &tokenList,
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
  __device__ CudaDynamicArray<CudaRealToken *> removeOldest(CudaDynamicArray<CudaRealToken *> &tokenList,
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

  __device__ CudaRealMarking *fire(CudaTimedTransition *transition, int tid) {
    if (transition == nullptr) {
      assert(false);
      return nullptr;
    }

    CudaRealMarking *child = parent->clone();
    CudaRealPlace *placeList = child->places;

    for (size_t i = 0; i < transition->presetLength; i++) {
      CudaTimedInputArc *input = transition->preset[i];
      CudaRealPlace place = placeList[transition->preset[i]->inputPlace.index];
      CudaDynamicArray<CudaRealToken *> *&tokenList = place.tokens;
      switch (transition->_firingMode) {
      case CudaSMC::FiringMode::Random:
        removeRandom(*tokenList, input->interval, input->weight, tid);
        break;
      case CudaSMC::FiringMode::Oldest:
        removeOldest(*tokenList, input->interval, input->weight);
        break;
      case CudaSMC::FiringMode::Youngest:
        removeYoungest(*tokenList, input->interval, input->weight);
        break;
      default:
        removeOldest(*tokenList, input->interval, input->weight);
        break;
      }

      auto toCreate = CudaDynamicArray<CudaPair<CudaTimedPlace *, CudaRealToken *>>(10);
      for (size_t i = 0; i < transition->transportArcsLength; i++) {
        auto transport = transition->transportArcs[i];
        int destInv = transport->destination.timeInvariant.bound;
        CudaRealPlace place = placeList[transport->source.index];
        CudaDynamicArray<CudaRealToken *> *&tokenList = place.tokens;
        CudaDynamicArray<CudaRealToken *> consumed(10);
        CudaTimeInterval &arcInterval = transport->interval;
        if (destInv < arcInterval.upperBound) arcInterval.setUpperBound(destInv, false);
        switch (transition->_firingMode) {
        case CudaSMC::FiringMode::Random:
          consumed = removeRandom(*tokenList, arcInterval, transport->weight, tid);
          break;
        case CudaSMC::FiringMode::Oldest:
          consumed = removeOldest(*tokenList, arcInterval, transport->weight);
          break;
        case CudaSMC::FiringMode::Youngest:
          consumed = removeYoungest(*tokenList, arcInterval, transport->weight);
          break;
        default:
          consumed = removeOldest(*tokenList, arcInterval, transport->weight);
          break;
        }
        for (size_t j = 0; j < consumed.size; j++) {
          toCreate.add(makeCudaPair(&(transport->destination), consumed.get(j)));
        }
      }

      for (size_t i = 0; i < transition->postsetLength; i++) {
        CudaTimedPlace &place = transition->postset[i]->outputPlace;
        CudaTimedOutputArc *post = transition->postset[i];
        auto token = new CudaRealToken{.age = 0.0, .count = static_cast<int>(post->weight)};
        child->addTokenInPlace(place, *token);
      }
      for (size_t i = 0; i < toCreate.size; i++) {
        auto [place, token] = toCreate.get(i);
        child->addTokenInPlace(*place, *token);
      }
    }
    return child;
  }
};
} // namespace VerifyTAPN::Cuda

#endif
