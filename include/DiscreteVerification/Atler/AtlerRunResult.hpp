#ifndef VERIFYTAPN_ATLER_RUNGEN_HPP_
#define VERIFYTAPN_ATLER_RUNGEN_HPP_

// #include "Core/TAPN/TimedTransition.hpp"
#include "DiscreteVerification/Atler/SimpleDeque.hpp"
#include "DiscreteVerification/Atler/SimpleDynamicArray.hpp"
#include "DiscreteVerification/Atler/SimpleInterval.hpp"
#include "DiscreteVerification/Atler/SimpleRealMarking.hpp"
#include "DiscreteVerification/Atler/SimpleStochasticStructure.hpp"
#include "DiscreteVerification/Atler/SimpleTimeInterval.hpp"
#include "DiscreteVerification/Atler/SimpleTimedArcPetriNet.hpp"
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Atler/SimpleTimedTransition.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>

namespace VerifyTAPN::Atler {

struct AtlerRunResult {
  bool maximal = false;
  SimpleTimedArcPetriNet tapn;
  SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval>>
      defaultTransitionIntervals;
  SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval>>
      transitionIntervals;
  SimpleDynamicArray<double> dates_sampled;
  SimpleDynamicArray<uint32_t> transitionsStatistics;
  SimpleRealMarking *origin = nullptr;
  SimpleRealMarking *parent = nullptr;
  double lastDelay = 0;
  double totalTime = 0;
  int totalSteps = 0;

  uint numericPrecision;
  std::ranlux48 rng;

  AtlerRunResult(SimpleTimedArcPetriNet tapn,
                 const unsigned int numericPrecision = 0)
      : tapn(tapn), defaultTransitionIntervals(tapn.transitionsLength),
        transitionIntervals(tapn.transitionsLength),
        numericPrecision(numericPrecision) {
    std::random_device rd;
    rng = std::ranlux48(rd());
  }

  AtlerRunResult(const AtlerRunResult &other)
      : tapn(other.tapn),
        defaultTransitionIntervals(other.defaultTransitionIntervals),
        transitionIntervals(other.transitionIntervals),
        dates_sampled(other.dates_sampled),
        transitionsStatistics(other.transitionsStatistics),
        lastDelay(other.lastDelay), totalTime(other.totalTime),
        totalSteps(other.totalSteps), numericPrecision(other.numericPrecision),
        rng(other.rng), maximal(other.maximal) {
    if (other.parent != nullptr) {
      parent = other.parent->clone();
    }
    if (other.origin != nullptr) {
      origin = other.origin->clone();
    }
  }

  ~AtlerRunResult() {
    if (parent != nullptr) {
      delete parent;
      parent = nullptr;
    }
    if (origin != nullptr) {
      delete origin;
      origin = nullptr;
    }
  }

  void prepare(SimpleRealMarking initMarking) {
    origin = initMarking.clone();
    parent = initMarking.clone();
    std::cout << "Prepared" << std::endl;
    double originMaxDelay = origin->availableDelay();
    SimpleDynamicArray<Util::SimpleInterval> *invIntervals =
        new SimpleDynamicArray<Util::SimpleInterval>(tapn.transitionsLength);
    invIntervals->add(Util::SimpleInterval(0, originMaxDelay));
    std::cout << "invIntervals size: " << invIntervals->size << std::endl;

    for (size_t i = 0; i < tapn.transitionsLength; i++) {
      std::cout << "Intersecting transition no: " << i << std::endl;
      SimpleTimedTransition *transition = tapn.transitions[i];
      // print transition length
      std::cout << "Inhibitor length first: " << transition->inhibitorArcsLength
                << std::endl;
      std::cout << "Intersecting transition no after: " << i << std::endl;
      if (transition->presetLength == 0 && std::cout << "first" << std::endl,
          transition->inhibitorArcs == 0) {
        defaultTransitionIntervals.add(*invIntervals);
      } else {
        std::cout << "Intersecting transition no inside: " << i << std::endl;
        SimpleDynamicArray<Util::SimpleInterval> firingDates =
            transitionFiringDates(*transition);
        // defaultTransitionIntervals.add(
        //     Util::setIntersection(*firingDates, *invIntervals));
        std::cout << "End of Intersection" << std::endl;
      }
      std::cout << "End of Intersection 2" << std::endl;
    }
    reset();
  }

  void reset() {
    if (parent != nullptr) {
      delete parent;
      parent = nullptr;
    }
    parent = origin->clone();
    transitionIntervals = defaultTransitionIntervals;
    dates_sampled = SimpleDynamicArray<double>(transitionIntervals.size);
    for (size_t i = 0; i < dates_sampled.size; i++) {
      dates_sampled.add(std::numeric_limits<double>::infinity());
    }
    lastDelay = 0.0;
    totalTime = 0.0;
    totalSteps = 0;
    bool deadlock = true;
    for (size_t i = 0; i < dates_sampled.size; i++) {
      auto intervals = transitionIntervals.get(i);
      if (intervals.empty() && intervals.get(0).lower() == 0) {
        const SimpleSMC::Distribution distribution =
            tapn.transitions[i]->distribution;
        dates_sampled.set(i, distribution.sample(rng, numericPrecision));
      }
      deadlock &= transitionIntervals.get(i).empty() ||
                  (transitionIntervals.get(i).size == 0 &&
                   transitionIntervals.get(i).get(0).lower() == 0 &&
                   dates_sampled.get(i) == 0);
    }

    parent->deadlocked = deadlock;
    std::cout << "Reset" << std::endl;
  }

  void refreshTransitionsIntervals() {
    double max_delay = parent->availableDelay();
    SimpleDynamicArray<Util::SimpleInterval> invIntervals(10);
    invIntervals.add(Util::SimpleInterval(0, max_delay));
    bool deadlocked = true;

    for (size_t i = 0; i < tapn.transitionsLength; i++) {
      auto transition = tapn.transitions[i];
      int index = transition->index;
      if (transition->getPresetSize() == 0 &&
          transition->inhibitorArcsLength == 0) {
        transitionIntervals.set(i, invIntervals);
      } else {
        SimpleDynamicArray<Util::SimpleInterval> firingDates =
            transitionFiringDates(*transition);
        transitionIntervals.set(
            i, Util::setIntersection(firingDates, invIntervals));
      }
      bool enabled = (!transitionIntervals.empty()) &&
                     (transitionIntervals.get(i).get(0).lower() == 0);
      bool newlyEnabled = enabled && (dates_sampled.get(i) ==
                                      std::numeric_limits<double>::infinity());
      bool reachedUpper = enabled && !newlyEnabled &&
                          (transitionIntervals.get(i).get(0).upper() == 0 &&
                           dates_sampled.get(i) > 0);
      if (!enabled || reachedUpper) {
        dates_sampled.set(i, std::numeric_limits<double>::infinity());
      } else if (newlyEnabled) {
        const auto distribution = tapn.transitions[i]->distribution;
        double date = distribution.sample(rng, numericPrecision);
        if (transitionIntervals.get(i).get(0).upper() > 0 || date == 0) {
          dates_sampled.set(i, date);
        }
      }
      deadlocked &= transitionIntervals.get(i).empty() ||
                    (transitionIntervals.get(i).size == 1 &&
                     transitionIntervals.get(i).get(0).lower() == 0 &&
                     dates_sampled.get(i) > 0);
    }
    parent->deadlocked = deadlocked;
  }

  SimpleRealMarking *next() {
    auto [winner, delay] = getWinnerTransitionAndDelay();

    if (delay == std::numeric_limits<double>::infinity()) {
      maximal = true;
      return nullptr;
    }

    parent->deltaAge(delay);
    totalTime += delay;

    parent->fromDelay = delay + parent->fromDelay;

    if (winner != nullptr) {
        std::cout << "Winner: " << winner->name << std::endl;
        totalSteps++;
      dates_sampled.set(winner->index, std::numeric_limits<double>::infinity());
      auto child = fire(winner);
      child->generatedBy = winner;
      delete parent;
      parent = child;
    }

    for (size_t i = 0; i < transitionIntervals.size; i++) {
        double date = dates_sampled.get(i);
        double newVal = (date == std::numeric_limits<double>::infinity()) ? std::numeric_limits<double>::infinity() : date - delay;
    }

    refreshTransitionsIntervals();
    return parent;
  }

  std::pair<SimpleTimedTransition *, double> getWinnerTransitionAndDelay() {
    SimpleDynamicArray<size_t> winner_indexes;
    double date_min = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < transitionIntervals.size; i++) {
      auto intervals = transitionIntervals.get(i);
      if (intervals.empty())
        continue;
      double date = std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < intervals.size; j++) {
        auto interval = intervals.get(j);
        if (interval.lower() > 0) {
          date = interval.lower();
          break;
        }
        if (interval.upper() > date) {
          date = interval.upper();
          break;
        }
      }
      date = std::min(dates_sampled.get(i), date);
      if (date < date_min) {
        date_min = date;
        winner_indexes.clear();
      }
      if (dates_sampled.get(i) == date_min) {
        winner_indexes.add(i);
      }
    }
    SimpleTimedTransition *winner;
    if (winner_indexes.empty()) {
      winner = nullptr;
    } else if (winner_indexes.size == 1) {
      winner = tapn.transitions[winner_indexes.get(0)];
    } else {
      winner = chooseWeightedWinner(winner_indexes);
    }
    return std::make_pair(winner, date_min);
  }

  SimpleTimedTransition *
  chooseWeightedWinner(const SimpleDynamicArray<size_t> winner_indexes) {
    double total_weight = 0;
    SimpleDynamicArray<size_t> infinite_weights(10);
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
      int winner_index =
          std::uniform_int_distribution<>(0, infinite_weights.size - 1)(rng);
      return tapn.transitions[infinite_weights.get(winner_index)];
    }
    if (total_weight == 0) {
      int winner_index =
          std::uniform_int_distribution<>(0, winner_indexes.size - 1)(rng);
      return tapn.transitions[winner_indexes.get(winner_index)];
    }
    double winning_weight =
        std::uniform_real_distribution<>(0, total_weight)(rng);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      SimpleTimedTransition *transition = tapn.transitions[candidate];
      winning_weight -= transition->_weight;
      if (winning_weight <= 0) {
        return transition;
      }
    }

    return tapn.transitions[winner_indexes.get(0)];
  }

  SimpleDynamicArray<Util::SimpleInterval>
  transitionFiringDates(const SimpleTimedTransition &transition) {
    auto firingIntervals = SimpleDynamicArray<Util::SimpleInterval>(10);
    firingIntervals.add(
        Util::SimpleInterval(0, std::numeric_limits<double>::infinity()));
    auto disabled = SimpleDynamicArray<Util::SimpleInterval>();

    // for each inhibitor arc
    for (size_t i = 0; i < transition.inhibitorArcsLength; i++) {
      std::cout << "Inhibitor arc length: " << transition.inhibitorArcsLength
                << std::endl;
      std::cout << "Preset arc length: " << transition.presetLength
                << std::endl;
      auto inhib = transition.inhibitorArcs[i];
      std::cout << "Inhibitor arc 2" << std::endl;
      if (parent->numberOfTokensInPlace(inhib->inputPlace.index) >=
          inhib->weight) {
        std::cout << "Inhibitor arc 3" << std::endl;
        return disabled;
      }
    }

    for (size_t i = 0; i < transition.presetLength; i++) {
      auto arc = transition.preset[i];
      auto &place = parent->places[arc->inputPlace.index];

      std::cout << "Preset arc" << std::endl;
      std::cout << "place name: " << place.place.name << std::endl;
      std::cout << "place size: " << place.tokens.size << std::endl;
      std::cout << "place capacity: " << place.tokens.capacity << std::endl;
      if (place.isEmpty()) {
        return disabled;
      }
      std::cout << "Preset arc 2" << std::endl;
      firingIntervals = Util::setIntersection(
          firingIntervals,
          arcFiringDates(arc->interval, arc->weight, place.tokens));
      if (firingIntervals.empty())
        return firingIntervals;
    }

    for (size_t i = 0; i < transition.transportArcsLength; i++) {
      std::cout << "Transport arc" << std::endl;
      auto transport = transition.transportArcs[i];
      auto &place = parent->places[transport->source.index];
      if (place.isEmpty())
        return disabled;

      SimpleTimeInvariant targetInvariant =
          transport->destination.timeInvariant;
      SimpleTimeInterval arcInterval = transport->interval;
      if (targetInvariant.bound < arcInterval.upperBound) {
        arcInterval.setUpperBound(targetInvariant.bound,
                                  targetInvariant.isBoundStrict);
      }
      firingIntervals = Util::setIntersection(
          firingIntervals,
          arcFiringDates(arcInterval, transport->weight, place.tokens));
      if (firingIntervals.empty())
        return firingIntervals;
    }
    std::cout << "Firing in the hole" << std::endl;

    return firingIntervals;
  }

  SimpleDynamicArray<Util::SimpleInterval>
  arcFiringDates(SimpleTimeInterval time_interval, uint32_t weight,
                 SimpleDynamicArray<SimpleRealToken *> tokens) {

    std::cout << "Arc firing dates" << std::endl;
    Util::SimpleInterval arcInterval(time_interval.lowerBound,
                                     time_interval.upperBound);
    size_t total_tokens = 0;
    std::cout << "tokens size: " << tokens.size << std::endl;
    for (size_t i = 0; i < tokens.size; i++) {
      total_tokens += tokens.get(i)->count;
    }
    std::cout << "Arc firing dates" << std::endl;
    if (total_tokens < weight)
      return SimpleDynamicArray<Util::SimpleInterval>();

    SimpleDynamicArray<Util::SimpleInterval> firingIntervals(10);
    SimpleDeque<double> selected = SimpleDeque<double>();
    for (size_t i = 0; i < tokens.size; i++) {
      for (int j = 0; j < tokens.get(i)->count; j++) {
        selected.push_back(tokens.get(i)->age);
        if (selected.size > weight) {
          selected.pop_front();
        }
        if (selected.size == weight) {
          Util::SimpleInterval tokenSetInterval(
              0, std::numeric_limits<double>::infinity());
          for (size_t k = 0; k < selected.size; k++) {
            tokenSetInterval = Util::intersect(
                tokenSetInterval,
                Util::SimpleInterval(selected.at(k), selected.at(k)));
          }
          Util::setAdd(firingIntervals, tokenSetInterval);
        }
      }
    }
    return firingIntervals;
  }

  SimpleDynamicArray<SimpleRealToken *>
  removeRandom(SimpleDynamicArray<SimpleRealToken *> &tokenList,
               const SimpleTimeInterval &interval, const int weight) {
    auto res = SimpleDynamicArray<SimpleRealToken *>(tokenList.size);
    int remaning = weight;
    std::uniform_int_distribution<> randomTokenIndex(0, tokenList.size - 1);
    size_t tok_index = randomTokenIndex(rng);
    size_t tested = 0;

    while (remaning > 0 && tested < tokenList.size) {
      SimpleRealToken *token = tokenList.get(tok_index);
      if (interval.contains(token->age)) {
        res.add(new SimpleRealToken{.age = token->age, .count = 1});
        remaning--;
        tokenList.get(tok_index)->remove(1);
        if (tokenList.get(tok_index)->count == 0) {
          tokenList.remove(tok_index);
          randomTokenIndex =
              std::uniform_int_distribution<>(0, tokenList.size - 1);
        }
        if (remaning > 0) {
          tok_index = randomTokenIndex(rng);
          tested = 0;
        }
      } else {
        tok_index = (tok_index + 1) % tokenList.size;
      }
    }
    assert(remaning == 0);
    return res;
  }

  SimpleDynamicArray<SimpleRealToken *>
  removeYoungest(SimpleDynamicArray<SimpleRealToken *> &tokenList,
                 const SimpleTimeInterval &interval, const int weight) {

    auto res = SimpleDynamicArray<SimpleRealToken *>();
    int remaining = weight;
    for (size_t i = 0; i < tokenList.size; i++) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!interval.contains(age)) {
        continue;
      }
      int count = token->count;
      if (count >= remaining) {
        res.add(new SimpleRealToken{.age = age, .count = count});
        token->remove(remaining);
        if (token->count == 0)
          tokenList.remove(i);
        remaining = 0;
        break;
      } else {
        res.add(new SimpleRealToken{.age = age, .count = count});
        remaining -= count;
        tokenList.remove(i);
      }
    }

    assert(remaining == 0);
    return res;
  }

  // NOTE: Double check this method to ensure it is correct
  SimpleDynamicArray<SimpleRealToken *>
  removeOldest(SimpleDynamicArray<SimpleRealToken *> &tokenList,
               const SimpleTimeInterval &timeInterval, const int weight) {
    auto res = SimpleDynamicArray<SimpleRealToken *>();
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
        res.add(new SimpleRealToken{.age = age, .count = count});
        token->remove(remaining);
        if (token->count == 0)
          tokenList.remove(i);
        remaining = 0;
        break;
      } else {
        res.add(new SimpleRealToken{.age = age, .count = count});
        remaining -= count;
        tokenList.remove(i);
      }
    }
    assert(remaining == 0);
    return res;
  }

  SimpleRealMarking *fire(SimpleTimedTransition *transition) {
    if (transition == nullptr) {
      assert(false);
      return nullptr;
    }

    SimpleRealMarking *child = parent->clone();
    SimpleRealPlace *placeList = child->places;

    for (size_t i = 0; i < transition->presetLength; i++) {
      SimpleTimedInputArc *input = transition->preset[i];
      SimpleRealPlace &place =
          placeList[transition->preset[i]->inputPlace.index];
      SimpleDynamicArray<SimpleRealToken *> &tokenList = place.tokens;
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

      auto toCreate =
          SimpleDynamicArray<std::pair<SimpleTimedPlace *, SimpleRealToken *>>(
              10);
      for (size_t i = 0; i < transition->transportArcsLength; i++) {
        auto transport = transition->transportArcs[i];
        int destInv = transport->destination.timeInvariant.bound;
        SimpleRealPlace &place = placeList[transport->source.index];
        SimpleDynamicArray<SimpleRealToken *> &tokenList = place.tokens;
        SimpleDynamicArray<SimpleRealToken *> consumed(10);
        SimpleTimeInterval &arcInterval = transport->interval;
        if (destInv < arcInterval.upperBound)
          arcInterval.setUpperBound(destInv, false);
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
          toCreate.add(
              std::make_pair(&(transport->destination), consumed.get(j)));
        }
      }

      for (size_t i = 0; i < transition->postsetLength; i++) {
        SimpleTimedPlace &place = transition->postset[i]->outputPlace;
        SimpleTimedOutputArc *post = transition->postset[i];
        auto token = SimpleRealToken{.age = 0.0,
                                     .count = static_cast<int>(post->weight)};
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
