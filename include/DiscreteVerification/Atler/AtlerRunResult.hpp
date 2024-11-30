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
  SimpleDynamicArray<double> *dates_sampled;
  SimpleDynamicArray<uint32_t> transitionsStatistics;
  SimpleRealMarking *origin = nullptr;
  SimpleRealMarking *parent = nullptr;
  double lastDelay = 0;
  double totalTime = 0;
  int totalSteps = 0;

  uint numericPrecision;
  std::ranlux48 rng;

  // add default constructor
  AtlerRunResult() {}

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
        numericPrecision(other.numericPrecision), rng(other.rng),
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

  AtlerRunResult *copy() const {
    AtlerRunResult *clone = new AtlerRunResult(tapn);
    clone->origin = new SimpleRealMarking(*origin);
    clone->numericPrecision = numericPrecision;
    clone->defaultTransitionIntervals = defaultTransitionIntervals;
    clone->reset();
    return clone;
  }

  void prepare(SimpleRealMarking initMarking) {
    origin = new SimpleRealMarking(initMarking);
    parent = new SimpleRealMarking(initMarking);
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
        for (size_t i = 0; i < firingDates.size; i++) {
          std::cout << "firing date: " << firingDates.get(i).lower() << ", "
                    << firingDates.get(i).upper() << std::endl;
        }
        defaultTransitionIntervals.add(
            Util::setIntersection(firingDates, *invIntervals));
        std::cout << "End of Intersection" << std::endl;
      }
      std::cout << "End of Intersection 2" << std::endl;
    }
    reset();

    // print all the tokens form the parent marking and their counts
    // for (size_t i = 0; i < parent->placesLength; i++) {
    //   std::cout << "Parent place tokens length: "
    //             << parent->places[i].tokens.size << std::endl;
    //   for (size_t j = 0; j < parent->places[i].tokens.size; j++) {
    //     auto token = parent->places[i].tokens.get(j);
    //     std::cout << "Token age: " << token->age << std::endl;
    //     std::cout << "Token count: " << token->count << std::endl;
    //   }
    // }
  }

  void reset() {
    std::cout << "Reset begining" << std::endl;
    transitionIntervals = defaultTransitionIntervals;
    // print transition intervals length

    dates_sampled = new SimpleDynamicArray<double>(transitionIntervals.size);

    std::cout << "Reset begining 2" << std::endl;
    // print dates sampled length
    std::cout << "Dates sampled length: " << dates_sampled->capacity
              << std::endl;
    for (size_t i = 0; i < transitionIntervals.size; i++) {
      dates_sampled->add(std::numeric_limits<double>::infinity());
    }
    // print old dates sampled length
    std::cout << "Old dates sampled length: " << dates_sampled->size
              << std::endl;
    lastDelay = 0.0;
    totalTime = 0.0;
    totalSteps = 0;
    bool deadlock = true;
    for (size_t i = 0; i < dates_sampled->size; i++) {
      std::cout << "Before " << i << ": " << std::endl;
      auto intervals = transitionIntervals.get(i);
      // print intervals length
      std::cout << "Intervals length: " << intervals.size << std::endl;
      // print if intervals is empty
      std::cout << "Intervals empty: " << intervals.empty() << std::endl;
      if (!intervals.empty() && intervals.get(0).lower() == 0) {
        const SimpleSMC::Distribution distribution =
            tapn.transitions[i]->distribution;
        dates_sampled->set(i, distribution.sample(rng, numericPrecision));
      }
      // print check
      std::cout << "Check " << i << ": " << std::endl;
      deadlock &= transitionIntervals.get(i).empty() ||
                  (transitionIntervals.get(i).size == 0 &&
                   transitionIntervals.get(i).get(0).lower() == 0 &&
                   dates_sampled->get(i) == 0);
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
        transitionIntervals.set(index, invIntervals);
      } else {
        SimpleDynamicArray<Util::SimpleInterval> firingDates =
            transitionFiringDates(*transition);
        transitionIntervals.set(
            i, Util::setIntersection(firingDates, invIntervals));
      }
      std::cout << "Transition intervals inter size: "
                << transitionIntervals.get(i).size << std::endl;
      bool enabled = (!transitionIntervals.get(i).empty()) &&
                     (transitionIntervals.get(i).get(0).lower() == 0);
      bool newlyEnabled = enabled && (dates_sampled->get(i) ==
                                      std::numeric_limits<double>::infinity());
      bool reachedUpper = enabled && !newlyEnabled &&
                          (transitionIntervals.get(i).get(0).upper() == 0) &&
                          (dates_sampled->get(i) > 0);
      if (!enabled || reachedUpper) {
        dates_sampled->set(i, std::numeric_limits<double>::infinity());
      } else if (newlyEnabled) {
        const auto distribution = tapn.transitions[i]->distribution;
        double date = distribution.sample(rng, numericPrecision);
        // print transitionIntervals.get(i).get(0)
        std::cout << "Transition intervals get i 0: " << std::endl;
        std::cout << "Transition intervals get i 0.lower(): "
                  << transitionIntervals.get(i).get(0).lower() << std::endl;
        std::cout << "Transition intervals get i 0.upper(): "
                  << transitionIntervals.get(i).get(0).upper() << std::endl;
        std::cout << "Date: " << date << std::endl;
        if (transitionIntervals.get(i).get(0).upper() > 0 || date == 0) {
          dates_sampled->set(i, date);
        }
      }
      deadlocked &= transitionIntervals.get(i).empty() ||
                    (transitionIntervals.get(i).size == 1 &&
                     transitionIntervals.get(i).get(0).lower() == 0 &&
                     dates_sampled->get(i) > 0);
    }
    parent->deadlocked = deadlocked;
  }

  SimpleRealMarking *next() {
    auto [winner, delay] = getWinnerTransitionAndDelay();

    if (delay == std::numeric_limits<double>::infinity()) {
      // print delay is infinity
      std::cout << "Delay is infinity/Deadlocked" << std::endl;
      maximal = true;
      return nullptr;
    }

    parent->deltaAge(delay);
    totalTime += delay;

    parent->fromDelay = delay + parent->fromDelay;

    if (winner != nullptr) {
      std::cout << "Winner: " << winner->name << std::endl;
      totalSteps++;
      dates_sampled->set(winner->index,
                         std::numeric_limits<double>::infinity());
      auto child = fire(winner);
      child->generatedBy = winner;
      delete parent;
      parent = child;
    }

    for (size_t i = 0; i < transitionIntervals.size; i++) {
      double date = dates_sampled->get(i);
      double newVal = (date == std::numeric_limits<double>::infinity())
                          ? std::numeric_limits<double>::infinity()
                          : date - delay;
      // print any date that is not infinity
      if (date != std::numeric_limits<double>::infinity()) {
        std::cout << "date: " << date << std::endl;
        std::cout << "newVal: " << newVal << std::endl;
      }
      dates_sampled->set(i, newVal);
    }

    refreshTransitionsIntervals();
    return parent;
  }

  std::pair<SimpleTimedTransition *, double> getWinnerTransitionAndDelay() {
    SimpleDynamicArray<size_t> winner_indexes(10);
    double date_min = std::numeric_limits<double>::infinity();
    // print transition intervals length
    std::cout << "Transition intervals length: " << transitionIntervals.size
              << std::endl;

    for (size_t i = 0; i < transitionIntervals.size; i++) {
      auto intervals = transitionIntervals.get(i);
      std::cout << "Intervals length: " << intervals.size << std::endl;
      if (intervals.empty())
        continue;
      double date = std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < intervals.size; j++) {
        // print the word before
        std::cout << "Before " << j << ": ";
        auto interval = intervals.get(j);
        std::cout << "After " << j << ": ";
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
      std::cout << "Date: " << date << std::endl;
      // print length of dates sampled
      std::cout << "Dates sampled length: " << (*dates_sampled).size
                << std::endl;
      date = std::min(dates_sampled->get(i), date);
      if (date < date_min) {
        std::cout << "New minimum date: " << date << std::endl;
        date_min = date;
        winner_indexes.clear();
      }
      if (dates_sampled->get(i) == date_min) {
        winner_indexes.add(i);
      }
    }
    SimpleTimedTransition *winner;
    if (winner_indexes.empty()) {
      winner = nullptr;
    } else if (winner_indexes.size == 1) {
      winner = tapn.transitions[winner_indexes.get(0)];
      std::cout << "Winner indexes size: " << std::endl;
    } else {
      winner = chooseWeightedWinner(winner_indexes);
      std::cout << "Winner indexes size: " << std::endl;
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
      SimpleRealPlace place = parent->places[arc->inputPlace.index];

      std::cout << "Preset arc" << std::endl;
      std::cout << "place name: " << place.place.name << std::endl;
      std::cout << "place size: " << place.tokens->size << std::endl;
      std::cout << "place capacity: " << place.tokens->capacity << std::endl;
      if (place.isEmpty()) {
        return disabled;
      }
      std::cout << "Preset arc 2" << std::endl;
      firingIntervals = Util::setIntersection(
          firingIntervals,
          arcFiringDates(arc->interval, arc->weight, *place.tokens));
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
          arcFiringDates(arcInterval, transport->weight, *place.tokens));
      if (firingIntervals.empty())
        return firingIntervals;
    }
    std::cout << "Firing in the hole" << std::endl;

    return firingIntervals;
  }

  SimpleDynamicArray<Util::SimpleInterval>
  arcFiringDates(SimpleTimeInterval time_interval, uint32_t weight,
                 SimpleDynamicArray<SimpleRealToken *> &tokens) {

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
      // print the counts from all the tokens
      // print size of tokens
      std::cout << "Tokens z size: " << tokens.size << std::endl;
      std::cout << "Tokens get z count: " << tokens.get(i)->count << std::endl;
    }
    for (size_t i = 0; i < tokens.size; i++) {
      for (int j = 0; j < tokens.get(i)->count; j++) {
        // print tokens get i count
        std::cout << "Tokens get i count: " << tokens.get(i)->count
                  << std::endl;
        std::cout << "Tokens get i age: " << tokens.get(i)->age << std::endl;
        std::cout << "Before push_back" << std::endl;
        selected.push_back(tokens.get(i)->age);
        if (selected.size > weight) {
          selected.pop_front();
        }
        if (selected.size == weight) {
          Util::SimpleInterval tokenSetInterval(
              0, std::numeric_limits<double>::infinity());
          for (size_t k = 0; k < selected.size; k++) {
            Util::SimpleInterval shifted = arcInterval;
            shifted.delta(-selected.at(k));
            tokenSetInterval = Util::intersect(tokenSetInterval, shifted);
          }
          Util::setAdd(firingIntervals, tokenSetInterval);
        }
      }
    }
    return firingIntervals;
  }

  SimpleDynamicArray<SimpleRealToken *>
  removeRandom(SimpleDynamicArray<SimpleRealToken *> tokenList,
               const SimpleTimeInterval &interval, const int weight) {
    std::cout << "Remove random method is being called" << std::endl;
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
        tested++;
      }
    }
    assert(remaning == 0);
    return res;
  }

  SimpleDynamicArray<SimpleRealToken *>
  removeYoungest(SimpleDynamicArray<SimpleRealToken *> &tokenList,
                 const SimpleTimeInterval &interval, const int weight) {
    std::cout << "Remove youngest method is being called" << std::endl;

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
      SimpleRealPlace place =
          placeList[transition->preset[i]->inputPlace.index];
      SimpleDynamicArray<SimpleRealToken *> *&tokenList = place.tokens;
      switch (transition->_firingMode) {
      case SimpleSMC::FiringMode::Random:
        removeRandom(*tokenList, input->interval, input->weight);
        break;
      case SimpleSMC::FiringMode::Oldest:
        removeOldest(*tokenList, input->interval, input->weight);
        break;
      case SimpleSMC::FiringMode::Youngest:
        removeYoungest(*tokenList, input->interval, input->weight);
        break;
      default:
        removeOldest(*tokenList, input->interval, input->weight);
        break;
      }

      auto toCreate =
          SimpleDynamicArray<std::pair<SimpleTimedPlace *, SimpleRealToken *>>(
              10);
      for (size_t i = 0; i < transition->transportArcsLength; i++) {
        auto transport = transition->transportArcs[i];
        int destInv = transport->destination.timeInvariant.bound;
        SimpleRealPlace place = placeList[transport->source.index];
        SimpleDynamicArray<SimpleRealToken *> *&tokenList = place.tokens;
        SimpleDynamicArray<SimpleRealToken *> consumed(10);
        SimpleTimeInterval &arcInterval = transport->interval;
        if (destInv < arcInterval.upperBound)
          arcInterval.setUpperBound(destInv, false);
        switch (transition->_firingMode) {
        case SimpleSMC::FiringMode::Random:
          consumed = removeRandom(*tokenList, arcInterval, transport->weight);
          break;
        case SimpleSMC::FiringMode::Oldest:
          consumed = removeOldest(*tokenList, arcInterval, transport->weight);
          break;
        case SimpleSMC::FiringMode::Youngest:
          consumed = removeYoungest(*tokenList, arcInterval, transport->weight);
          break;
        default:
          consumed = removeOldest(*tokenList, arcInterval, transport->weight);
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
        auto token = new SimpleRealToken{.age = 0.0,
                                     .count = static_cast<int>(post->weight)};
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
} // namespace VerifyTAPN::Atler

#endif
