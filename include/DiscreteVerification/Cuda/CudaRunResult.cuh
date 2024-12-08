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
#include "DiscreteVerification/Atler/SimpleTimeInvariant.hpp"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <limits>
#include <random>

namespace VerifyTAPN::Cuda {

struct CudaRunResult {
  bool maximal = false;
  CudaTimedArcPetriNet *tapn;
  CudaDynamicArray<CudaDynamicArray<Util::CudaInterval> *> *transitionIntervals;
  CudaDynamicArray<double> *datesSampled;
  CudaRealMarking *realMarking;
  double totalTime = 0;
  int totalSteps = 0;

  uint numericPrecision = 0;

  curandState_t *rngStates;

  __host__ __device__ CudaRunResult(CudaTimedArcPetriNet *tapn, CudaRealMarking *srm,
                                    const unsigned int numericPrecision = 0)
      : tapn(tapn), realMarking(srm), numericPrecision(numericPrecision) {

    // Initialize transition intervals
    transitionIntervals = new CudaDynamicArray<CudaDynamicArray<Util::CudaInterval> *>(tapn->transitionsLength);
  }

  __host__ __device__ ~CudaRunResult() {
    for (size_t i = 0; i < transitionIntervals->size; i++) {
      delete transitionIntervals->get(i);
    }
    delete transitionIntervals;
    delete datesSampled;
  }

  // private: (fix this later)
  __device__ void prepare(curandState *local_r_state) {
    double originMaxDelay = realMarking->availableDelay();

    auto invIntervals = CudaDynamicArray<Util::CudaInterval>(10);
    invIntervals.add(Util::CudaInterval(0, originMaxDelay));

    for (size_t i = 0; i < tapn->transitionsLength; i++) {
      auto *transition = tapn->transitions[i];
      if (transition->getPresetSize() == 0 && transition->inhibitorArcs == 0) {
        transitionIntervals->add(new CudaDynamicArray<Util::CudaInterval>(invIntervals));
      } else {
        CudaDynamicArray<Util::CudaInterval> firingDates = transitionFiringDates(*transition);

        auto intersection = Util::setIntersection(firingDates, invIntervals);

        transitionIntervals->add(new CudaDynamicArray<Util::CudaInterval>(intersection));
      }
    }

    reset(local_r_state);
  }

  __device__ void reset(curandState *local_r_state) {
    datesSampled = new CudaDynamicArray<double>(transitionIntervals->size);
    for (size_t i = 0; i < transitionIntervals->size; i++) {
      datesSampled->add(HUGE_VAL);
    }

    bool deadlock = true;
    for (size_t i = 0; i < datesSampled->size; i++) {
      auto *intervals = transitionIntervals->get(i);
      if (!intervals->empty() && intervals->get(0).lower() == 0) {
        const CudaSMC::Distribution distribution = tapn->transitions[i]->distribution;
        datesSampled->set(i, distribution.sample(local_r_state, numericPrecision));
      }
      deadlock &= transitionIntervals->get(i)->empty() ||
                  (transitionIntervals->get(i)->size == 1 && transitionIntervals->get(i)->get(0).lower() == 0 &&
                   datesSampled->get(i) == 0);
    }

    realMarking->deadlocked = deadlock;
  }

  // next function
  __device__ bool next(curandState *local_r_state) {
    auto [winner, delay] = getWinnerTransitionAndDelay(local_r_state);

    if (delay == HUGE_VAL) {
      maximal = true;
      return false;
    }

    realMarking->deltaAge(delay);
    totalTime += delay;

    if (winner != nullptr) {
      totalSteps++;
      datesSampled->set(winner->index, HUGE_VAL);
      fire(winner, local_r_state);
    }

    for (size_t i = 0; i < transitionIntervals->size; i++) {
      double date = datesSampled->get(i);
      double newVal = (date == HUGE_VAL) ? HUGE_VAL : date - delay;
      datesSampled->set(i, newVal);
    }

    refreshTransitionsIntervals(local_r_state);
    return true;
  }

  // refresh intervals
  __device__ void refreshTransitionsIntervals(curandState *local_r_state) {
    double max_delay = realMarking->availableDelay();
    auto invIntervals = CudaDynamicArray<Util::CudaInterval>(10);
    invIntervals.add(Util::CudaInterval(0, max_delay));
    bool deadlocked = true;

    for (size_t i = 0; i < tapn->transitionsLength; i++) {
      auto *transition = tapn->transitions[i];
      int index = transition->index;
      if (transition->getPresetSize() == 0 && transition->inhibitorArcsLength == 0) {
        delete[] transitionIntervals->get(i);
        transitionIntervals->set(i, new CudaDynamicArray<Util::CudaInterval>(invIntervals));
      } else {
        // print i and the length of transitionIntervals
        CudaDynamicArray<Util::CudaInterval> firingDates = transitionFiringDates(*transition);

        auto intersection = Util::setIntersection(firingDates, invIntervals);

        delete transitionIntervals->get(i);
        transitionIntervals->set(i, new CudaDynamicArray<Util::CudaInterval>(intersection));
      }

      bool enabled = (!transitionIntervals->get(i)->empty()) && (transitionIntervals->get(i)->get(0).lower() == 0);
      bool newlyEnabled = enabled && (datesSampled->get(i) == HUGE_VAL);
      bool reachedUpper =
          enabled && !newlyEnabled && (transitionIntervals->get(i)->get(0).upper() == 0) && (datesSampled->get(i) > 0);

      if (!enabled || reachedUpper) {
        datesSampled->set(i, HUGE_VAL);
      } else if (newlyEnabled) {
        const auto distribution = tapn->transitions[i]->distribution;
        double date = distribution.sample(local_r_state, numericPrecision);
        // print transitionIntervals.get(i).get(0)
        if (transitionIntervals->get(i)->get(0).upper() > 0 || date == 0) {
          datesSampled->set(i, date);
        }
      }
      deadlocked &= transitionIntervals->get(i)->empty() ||
                    (transitionIntervals->get(i)->size == 1 && transitionIntervals->get(i)->get(0).lower() == 0 &&
                     datesSampled->get(i) > 0);
    }

    realMarking->deadlocked = deadlocked;
  }

  // get winner transtion
  __device__ CudaPair<CudaTimedTransition *, double> getWinnerTransitionAndDelay(curandState *local_r_state) {
    CudaDynamicArray<size_t> winner_indexes(10);
    double date_min = HUGE_VAL;

    for (size_t i = 0; i < transitionIntervals->size; i++) {
      auto intervals = transitionIntervals->get(i);
      if (intervals->empty()) continue;
      double date = HUGE_VAL;
      for (size_t j = 0; j < intervals->size; j++) {
        // print the word before
        auto interval = intervals->get(j);
        // print after
        if (interval.lower() > 0) {
          date = interval.lower();
          break;
        }
        if (interval.upper() > 0) {
          date = interval.upper();
          break;
        }
      }
      date = fmin(datesSampled->get(i), date);
      if (date < date_min) {
        date_min = date;
        winner_indexes.clear();
      }
      if (datesSampled->get(i) == date_min) {
        winner_indexes.add(i);
      }
    }
    CudaTimedTransition *winner;
    if (winner_indexes.empty()) {
      winner = nullptr;
    } else if (winner_indexes.size == 1) {
      winner = tapn->transitions[winner_indexes.get(0)];
    } else {
      winner = chooseWeightedWinner(winner_indexes, local_r_state);
    }
    return makeCudaPair(winner, date_min);
  }
  // choose weightedWinner

  __device__ CudaTimedTransition *chooseWeightedWinner(const CudaDynamicArray<size_t> winner_indexes,
                                                                curandState *local_r_state) {
    double total_weight = 0;
    CudaDynamicArray<size_t> infinite_weights(10);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      double priority = tapn->transitions[candidate]->_weight;
      if (priority == HUGE_VAL) {
        infinite_weights.add(candidate);
      } else {
        total_weight += priority;
      }
    }

    if (!infinite_weights.empty()) {
      // int winner_index = std::uniform_int_distribution<>(0, infinite_weights.size - 1)(local_r_state);
      int winner_index = CudaSMC::getRandomTokenIndex(local_r_state, infinite_weights.size - 1);
      return tapn->transitions[infinite_weights.get(winner_index)];
    }
    if (total_weight == 0) {
      // int winner_index = std::uniform_int_distribution<>(0, winner_indexes.size - 1)(local_r_state);
      int winner_index = CudaSMC::getRandomTokenIndex(local_r_state, winner_indexes.size - 1);
      return tapn->transitions[winner_indexes.get(winner_index)];
    }
    // double winning_weight = std::uniform_real_distribution<>(0.0, total_weight)(local_r_state);
    double winning_weight = CudaSMC::getRandomTokenIndex(local_r_state, total_weight);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      CudaTimedTransition *transition = tapn->transitions[candidate];
      winning_weight -= transition->_weight;
      if (winning_weight <= 0) {
        return transition;
      }
    }

    return tapn->transitions[winner_indexes.get(0)];
  }

  __device__ CudaDynamicArray<Util::CudaInterval>
  transitionFiringDates(const CudaTimedTransition &transition) {
    auto initialFiringIntervals = CudaDynamicArray<Util::CudaInterval>(10);

    initialFiringIntervals.add(Util::CudaInterval(0, HUGE_VAL));

    auto firingIntervalsStack = CudaDeque<CudaDynamicArray<Util::CudaInterval>>();
    firingIntervalsStack.push_front(initialFiringIntervals);

    auto disabled = CudaDynamicArray<Util::CudaInterval>();

    // for each inhibitor arc
    for (size_t i = 0; i < transition.inhibitorArcsLength; i++) {
      auto inhib = transition.inhibitorArcs[i];
      if (realMarking->numberOfTokensInPlace(inhib->inputPlace->index) >= inhib->weight) {
        return CudaDynamicArray(disabled);
      }
    }

    for (size_t i = 0; i < transition.presetLength; i++) {
      auto arc = transition.preset[i];
      CudaRealPlace *place = realMarking->places[arc->inputPlace->index];
      if (place->isEmpty()) {
        return CudaDynamicArray(disabled);
      }

      auto newFiringIntevals = Util::setIntersection(firingIntervalsStack.front->data,
                                                     arcFiringDates(arc->interval, arc->weight, *place->tokens));
      firingIntervalsStack.push_front(newFiringIntevals);

      if (newFiringIntevals.empty()) {
        return CudaDynamicArray(newFiringIntevals);
      }
    }

    for (size_t i = 0; i < transition.transportArcsLength; i++) {
      CudaTimedTransportArc *transport = transition.transportArcs[i];
      auto place = realMarking->places[transport->source->index];

      if (place->isEmpty()) {
        return CudaDynamicArray(disabled);
      }

      Atler::SimpleTimeInvariant targetInvariant = transport->destination->timeInvariant;
      CudaTimeInterval arcInterval = transport->interval;
      if (targetInvariant.bound < arcInterval.upperBound) {
        arcInterval.setUpperBound(targetInvariant.bound, targetInvariant.isBoundStrict);
      }

      auto newFiringIntevals = Util::setIntersection(firingIntervalsStack.front->data,
                                                     arcFiringDates(arcInterval, transport->weight, *place->tokens));
      firingIntervalsStack.push_front(newFiringIntevals);

      if (newFiringIntevals.empty()) {
        return CudaDynamicArray(newFiringIntevals);
      }
    }

    return CudaDynamicArray<Util::CudaInterval>(firingIntervalsStack.front->data);
  }

  __device__ CudaDynamicArray<Util::CudaInterval>
  arcFiringDates(CudaTimeInterval time_interval, const uint32_t weight, CudaDynamicArray<CudaRealToken *> &tokens) {
    Util::CudaInterval arcInterval(time_interval.lowerBound, time_interval.upperBound);

    size_t total_tokens = 0;
    for (size_t i = 0; i < tokens.size; i++) {
      total_tokens += tokens.get(i)->count;
    }

    if (total_tokens < weight) {
      return CudaDynamicArray<Util::CudaInterval>();
    }

    CudaDynamicArray<Util::CudaInterval> firingIntervals = CudaDynamicArray<Util::CudaInterval>(10);

    CudaDeque<double> selected = CudaDeque<double>();
    for (size_t i = 0; i < tokens.size; i++) {
      for (int j = 0; j < tokens.get(i)->count; j++) {
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

    return CudaDynamicArray<Util::CudaInterval>(firingIntervals);
  }

  __device__ CudaDynamicArray<CudaRealToken> removeRandom(CudaDynamicArray<CudaRealToken *> &tokenList,
                                                                   const CudaTimeInterval &timeInterval,
                                                                   const int weight, curandState *local_r_state) {
    auto res = CudaDynamicArray<CudaRealToken>(10);
    int remaining = weight;
    size_t tok_index = CudaSMC::getRandomTokenIndex(local_r_state, tokenList.size - 1);
    size_t tested = 0;

    while (remaining > 0 && tested < tokenList.size) {
      CudaRealToken *token = tokenList.get(tok_index);
      if (timeInterval.contains(token->age)) {
        res.add(CudaRealToken{.age = token->age, .count = 1});
        remaining--;
        tokenList.get(tok_index)->remove(1);
        if (tokenList.get(tok_index)->count == 0) {
          delete tokenList.get(tok_index);
          tokenList.remove(tok_index);
        }
        if (remaining > 0) {
          tok_index = CudaSMC::getRandomTokenIndex(local_r_state, tokenList.size - 1);
          tested = 0;
        }
      } else {
        tok_index = (tok_index + 1) % tokenList.size;
        tested++;
      }
    }
    assert(remaining == 0);
    return CudaDynamicArray<CudaRealToken>(res);
  }

  __device__ CudaDynamicArray<CudaRealToken>
  removeYoungest(CudaDynamicArray<CudaRealToken *> &tokenList, const CudaTimeInterval &timeInterval, const int weight) {
    auto res = CudaDynamicArray<CudaRealToken>(10);
    int remaining = weight;
    for (size_t i = 0; i < tokenList.size; i++) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!timeInterval.contains(age)) {
        continue;
      }
      int count = token->count;
      if (count >= remaining) {
        res.add(CudaRealToken{.age = age, .count = remaining});
        token->remove(remaining);
        if (token->count == 0) {
          delete token;
          tokenList.remove(i);
        }
        remaining = 0;
        break;
      } else {
        res.add(CudaRealToken{.age = age, .count = count});
        remaining -= count;
        delete token;
        tokenList.remove(i);
      }
    }
    assert(remaining == 0);
    return CudaDynamicArray<CudaRealToken>(res);
  }

  __device__ CudaDynamicArray<CudaRealToken>
  removeOldest(CudaDynamicArray<CudaRealToken *> &tokenList, const CudaTimeInterval &timeInterval, const int weight) {
    auto res = CudaDynamicArray<CudaRealToken>(10);
    int remaining = weight;
    for (size_t i = 0; i < tokenList.size; i++) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!timeInterval.contains(age)) {
        continue;
      }
      int count = token->count;
      if (count >= remaining) {
        res.add(CudaRealToken{.age = age, .count = remaining});
        token->remove(remaining);
        if (token->count == 0) {
          delete token;
          tokenList.remove(i);
        }
        remaining = 0;
        break;
      } else {
        res.add(CudaRealToken{.age = age, .count = count});
        remaining -= count;
        delete token;
        tokenList.remove(i);
      }
    }

    return CudaDynamicArray<CudaRealToken>(res);
  }

  __device__ bool fire(CudaTimedTransition *transition, curandState *local_r_state) {
    if (transition == nullptr) {
      return false;
    }

    CudaRealPlace **placeList = realMarking->places;

    for (size_t i = 0; i < transition->presetLength; i++) {
      CudaTimedInputArc *input = transition->preset[i];
      CudaRealPlace *place = placeList[input->inputPlace->index];
      CudaDynamicArray<CudaRealToken *> *tokenList = place->tokens;
      switch (transition->_firingMode) {
      case CudaSMC::FiringMode::Random:
        removeRandom(*tokenList, input->interval, input->weight, local_r_state);
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
    }

    auto toCreate = CudaDynamicArray<CudaPair<CudaTimedPlace *, CudaRealToken>>(10);

    for (size_t i = 0; i < transition->transportArcsLength; i++) {
      auto transport = transition->transportArcs[i];
      int destInv = transport->destination->timeInvariant.bound;
      CudaRealPlace *place = placeList[transport->source->index];
      CudaDynamicArray<CudaRealToken *> *tokenList = place->tokens;
      CudaTimeInterval &arcInterval = transport->interval;
      CudaDynamicArray<CudaRealToken> consumed = switcher(transition, tokenList, arcInterval, transport->weight, local_r_state);

      if (destInv < arcInterval.upperBound) {
        arcInterval.setUpperBound(destInv, false);
      }

      // switch (transition->_firingMode) {
      // case CudaSMC::FiringMode::Random: {
      //   consumed = removeRandom(*tokenList, arcInterval, transport->weight);
      //   break;
      // }
      // case CudaSMC::FiringMode::Oldest: {
      //   consumed = removeOldest(*tokenList, arcInterval, transport->weight);
      //   break;
      // }
      // case CudaSMC::FiringMode::Youngest: {
      //   consumed = removeYoungest(*tokenList, arcInterval,
      //   transport->weight); break;
      // }
      // default: {
      //   consumed = removeOldest(*tokenList, arcInterval, transport->weight);
      //   break;
      // }
      // }

      for (size_t j = 0; j < consumed.size; j++) {
        auto con = consumed.get(j);
        toCreate.add(makeCudaPair(transport->destination, CudaRealToken{con.age, con.count}));
      }
    }

    for (size_t i = 0; i < transition->postsetLength; i++) {
      CudaTimedPlace *place = transition->postset[i]->outputPlace;
      CudaTimedOutputArc *post = transition->postset[i];
      auto token = new CudaRealToken{.age = 0.0, .count = static_cast<int>(post->weight)};
      realMarking->addTokenInPlace(*place, *token);
    }
    for (size_t i = 0; i < toCreate.size; i++) {
      auto [place, token] = toCreate.get(i);
      realMarking->addTokenInPlace(*place, token);
    }

    return true;
  }

  __device__ CudaDynamicArray<CudaRealToken> switcher(CudaTimedTransition *transition,
                                                               CudaDynamicArray<CudaRealToken *> *tokens,
                                                               CudaTimeInterval &interval, int weight,
                                                               curandState *local_r_state) {

    switch (transition->_firingMode) {
    case CudaSMC::FiringMode::Random:
      return removeRandom(*tokens, interval, weight, local_r_state);
    case CudaSMC::FiringMode::Oldest:
      return removeOldest(*tokens, interval, weight);
    case CudaSMC::FiringMode::Youngest:
      return removeYoungest(*tokens, interval, weight);
    default:
      return removeOldest(*tokens, interval, weight);
    }
  }
};
} // namespace VerifyTAPN::Cuda

#endif
