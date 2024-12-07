#ifndef VERIFYTAPN_ATLER_RUNGEN_HPP_
#define VERIFYTAPN_ATLER_RUNGEN_HPP_

// #include "Core/TAPN/TimedTransition.hpp"
#include "DiscreteVerification/Atler/SimpleDeque.hpp"
#include "DiscreteVerification/Atler/SimpleDynamicArray.hpp"
#include "DiscreteVerification/Atler/SimpleInterval.hpp"
#include "DiscreteVerification/Atler/SimpleRealMarking.hpp"
#include "DiscreteVerification/Atler/SimpleStochasticStructure.hpp"
#include "DiscreteVerification/Atler/SimpleTimeInterval.hpp"
#include "DiscreteVerification/Atler/SimpleTimeInvariant.hpp"
#include "DiscreteVerification/Atler/SimpleTimedArcPetriNet.hpp"
#include "DiscreteVerification/Atler/SimpleTimedInputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedPlace.hpp"
#include "DiscreteVerification/Atler/SimpleTimedTransition.hpp"
#include "DiscreteVerification/Atler/SimpleTimedTransportArc.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>

namespace VerifyTAPN::Atler {

struct AtlerRunResult {
  bool maximal = false;
  SimpleTimedArcPetriNet tapn;
  SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval>>
      *transitionIntervals;
  SimpleDynamicArray<double> *dates_sampled;
  // SimpleRealMarking *origin = nullptr;
  SimpleRealMarking *parent;
  double totalTime = 0;
  int totalSteps = 0;

  uint numericPrecision;
  std::ranlux48 rng;

  // add default constructor
  AtlerRunResult() {}

  AtlerRunResult(SimpleTimedArcPetriNet tapn, SimpleRealMarking *srm,
                 const unsigned int numericPrecision = 0)
      : tapn(tapn), parent(srm), numericPrecision(numericPrecision) {
    std::random_device rd;
    rng = std::ranlux48(rd());
    transitionIntervals =
        new SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval>>(
            tapn.transitionsLength);
  }

  AtlerRunResult(const AtlerRunResult &other) {
    std::cout << "AtlerRunResult copy constructor" << std::endl;
    tapn = other.tapn;

    transitionIntervals =
        new SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval>>(
            other.transitionIntervals->size);
    for (size_t i = 0; i < other.transitionIntervals->size; i++) {
      auto intervals = SimpleDynamicArray<Util::SimpleInterval>(
          other.transitionIntervals->get(i).size);
      intervals.ownsArray = false;
      for (size_t j = 0; j < other.transitionIntervals->get(i).size; j++) {
        intervals.add(other.transitionIntervals->get(i).get(j));
      }
      transitionIntervals->add(intervals);
    }

    parent = new SimpleRealMarking(*other.parent);
    numericPrecision = other.numericPrecision;
    std::random_device rd;
    rng = std::ranlux48(rd());
    reset();
  }

  ~AtlerRunResult() {
    // delete origin;
    delete parent;

    for (size_t i = 0; i < transitionIntervals->size; i++) {
      delete[] transitionIntervals->get(i).arr;
    }
  }

  void prepare() {
    double originMaxDelay = parent->availableDelay();

    auto invIntervals =
        SimpleDynamicArray<Util::SimpleInterval>(tapn.transitionsLength);
    invIntervals.ownsArray = false;
    invIntervals.add(Util::SimpleInterval(0, originMaxDelay));

    for (size_t i = 0; i < tapn.transitionsLength; i++) {
      SimpleTimedTransition *transition = tapn.transitions[i];
      if (transition->getPresetSize() == 0 && transition->inhibitorArcs == 0) {
        transitionIntervals->add(invIntervals);
      } else {
        SimpleDynamicArray<Util::SimpleInterval> firingDates =
            transitionFiringDates(*transition);

        transitionIntervals->add(
            Util::setIntersection(firingDates, invIntervals));

        delete[] firingDates.arr;
      }
    }

    // print default transition intervals
    // for (size_t i = 0; i < tapn.transitionsLength; i++) {
    //   std::cout << "Default intervals for transition "
    //             << tapn.transitions[i]->name << std::endl;
    //   for (size_t j = 0; j < defaultTransitionIntervals.get(i).size; j++) {
    //     std::cout << "(" << defaultTransitionIntervals.get(i).get(j).lower()
    //               << ", " << defaultTransitionIntervals.get(i).get(j).upper()
    //               << ")" << std::endl;
    //   }
    // }
    reset();
  }

  void reset() {
    maximal = false;
    totalTime = 0.0;
    totalSteps = 0;

    dates_sampled = new SimpleDynamicArray<double>(transitionIntervals->size);
    for (size_t i = 0; i < transitionIntervals->size; i++) {
      dates_sampled->add(std::numeric_limits<double>::infinity());
    }

    bool deadlock = true;
    for (size_t i = 0; i < dates_sampled->size; i++) {
      auto intervals = transitionIntervals->get(i);
      if (!intervals.empty() && intervals.get(0).lower() == 0) {
        const SimpleSMC::Distribution distribution =
            tapn.transitions[i]->distribution;
        dates_sampled->set(i, distribution.sample(rng, numericPrecision));
      }
      deadlock &= transitionIntervals->get(i).empty() ||
                  (transitionIntervals->get(i).size == 0 &&
                   transitionIntervals->get(i).get(0).lower() == 0 &&
                   dates_sampled->get(i) == 0);
    }

    parent->deadlocked = deadlock;
  }

  void refreshTransitionsIntervals() {
    double max_delay = parent->availableDelay();
    auto invIntervals = SimpleDynamicArray<Util::SimpleInterval>(10);
    invIntervals.add(Util::SimpleInterval(0, max_delay));
    bool deadlocked = true;

    for (size_t i = 0; i < tapn.transitionsLength; i++) {
      auto transition = tapn.transitions[i];
      int index = transition->index;
      if (transition->getPresetSize() == 0 &&
          transition->inhibitorArcsLength == 0) {
        delete[] transitionIntervals->get(index).arr;
        transitionIntervals->set(index, invIntervals);
      } else {
        SimpleDynamicArray<Util::SimpleInterval> firingDates =
            transitionFiringDates(*transition);

        SimpleDynamicArray<Util::SimpleInterval> arco =
            Util::setIntersection(firingDates, invIntervals);

        delete[] transitionIntervals->get(i).arr;
        transitionIntervals->set(i, arco);
        delete[] firingDates.arr;
        delete[] arco.arr;
      }
      bool enabled = (!transitionIntervals->get(i).empty()) &&
                     (transitionIntervals->get(i).get(0).lower() == 0);
      bool newlyEnabled = enabled && (dates_sampled->get(i) ==
                                      std::numeric_limits<double>::infinity());
      bool reachedUpper = enabled && !newlyEnabled &&
                          (transitionIntervals->get(i).get(0).upper() == 0) &&
                          (dates_sampled->get(i) > 0);
      if (!enabled || reachedUpper) {
        dates_sampled->set(i, std::numeric_limits<double>::infinity());
      } else if (newlyEnabled) {
        const auto distribution = tapn.transitions[i]->distribution;
        double date = distribution.sample(rng, numericPrecision);
        // print transitionIntervals.get(i).get(0)
        if (transitionIntervals->get(i).get(0).upper() > 0 || date == 0) {
          dates_sampled->set(i, date);
        }
      }
      deadlocked &= transitionIntervals->get(i).empty() ||
                    (transitionIntervals->get(i).size == 1 &&
                     transitionIntervals->get(i).get(0).lower() == 0 &&
                     dates_sampled->get(i) > 0);
    }
    parent->deadlocked = deadlocked;
  }

  bool next() {
    auto [winner, delay] = getWinnerTransitionAndDelay();

    if (delay == std::numeric_limits<double>::infinity()) {
      // print delay is infinity
      // std::cout << "Delay is infinity/Deadlocked" << std::endl;
      maximal = true;
      return false;
    }

    parent->deltaAge(delay);
    totalTime += delay;

    if (winner != nullptr) {
      // std::cout << "Winner: " << winner->name << std::endl;
      // print winnning delay
      // std::cout << "Delay: " << delay << std::endl;
      totalSteps++;
      dates_sampled->set(winner->index,
                         std::numeric_limits<double>::infinity());
      fire(winner);

      // std::cout << "New marking:" << std::endl;
      // for (size_t i = 0; i < parent->placesLength; i++) {
      //   std::cout << "Place " << i << ": ";
      //   for (size_t j = 0; j < parent->places[i].tokens->size; j++) {
      //     auto token = parent->places[i].tokens->get(j);
      //     std::cout << "(" << parent->places[i].place.name << ","
      //               << token->age << ") ";
      //   }
      //   std::cout << std::endl;
      // }
    }

    for (size_t i = 0; i < transitionIntervals->size; i++) {
      double date = dates_sampled->get(i);
      double newVal = (date == std::numeric_limits<double>::infinity())
                          ? std::numeric_limits<double>::infinity()
                          : date - delay;
      dates_sampled->set(i, newVal);
    }

    refreshTransitionsIntervals();
    return true;
  }

  std::pair<SimpleTimedTransition *, double> getWinnerTransitionAndDelay() {
    SimpleDynamicArray<size_t> winner_indexes(10);
    double date_min = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < transitionIntervals->size; i++) {
      auto intervals = transitionIntervals->get(i);
      if (intervals.empty())
        continue;
      double date = std::numeric_limits<double>::infinity();
      for (size_t j = 0; j < intervals.size; j++) {
        // print the word before
        auto interval = intervals.get(j);
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
      date = std::min(dates_sampled->get(i), date);
      if (date < date_min) {
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
        std::uniform_real_distribution<>(0.0, total_weight)(rng);
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

  // NOTE: Remember to delete the interval array within the returned
  // SimpleDynamicArray when it is no longer needed
  SimpleDynamicArray<Util::SimpleInterval>
  transitionFiringDates(const SimpleTimedTransition &transition) {
    auto firingIntervals = SimpleDynamicArray<Util::SimpleInterval>(10);

    firingIntervals.add(
        Util::SimpleInterval(0, std::numeric_limits<double>::infinity()));
    auto disabled = SimpleDynamicArray<Util::SimpleInterval>();

    // for each inhibitor arc
    for (size_t i = 0; i < transition.inhibitorArcsLength; i++) {
      // std::cout << "Inhibitor arc length: " << transition.inhibitorArcsLength
      //           << std::endl;
      // std::cout << "Preset arc length: " << transition.presetLength
      //           << std::endl;
      auto inhib = transition.inhibitorArcs[i];
      if (parent->numberOfTokensInPlace(inhib->inputPlace.index) >=
          inhib->weight) {
        disabled.ownsArray = false;
        return disabled;
      }
    }

    for (size_t i = 0; i < transition.presetLength; i++) {
      auto arc = transition.preset[i];
      SimpleRealPlace *place = parent->places[arc->inputPlace.index];

      // std::cout << "Preset arc" << std::endl;
      // std::cout << "place name: " << place.place.name << std::endl;
      // std::cout << "place size: " << place.tokens->size << std::endl;
      // std::cout << "place capacity: " << place.tokens->capacity << std::endl;
      if (place->isEmpty()) {
        disabled.ownsArray = false;
        return disabled;
      }

      auto arcFire = arcFiringDates(arc->interval, arc->weight, *place->tokens);

      auto newFiringIntervals = Util::setIntersection(firingIntervals, arcFire);
      delete[] firingIntervals.arr;
      delete[] arcFire.arr;
      firingIntervals = newFiringIntervals;

      if (firingIntervals.empty()) {
        firingIntervals.ownsArray = false;
        return firingIntervals;
      }
    }

    for (size_t i = 0; i < transition.transportArcsLength; i++) {
      // std::cout << "Transport arc" << std::endl;
      SimpleTimedTransportArc *transport = transition.transportArcs[i];
      auto place = parent->places[transport->source.index];

      // std::cout << "Preset arc" << std::endl;
      // std::cout << "place name: " << place.place.name << std::endl;
      // std::cout << "place size: " << place.tokens->size << std::endl;
      // std::cout << "place capacity: " << place.tokens->capacity << std::endl;

      if (place->isEmpty()) {
        disabled.ownsArray = false;
        return disabled;
      }

      SimpleTimeInvariant targetInvariant =
          transport->destination.timeInvariant;
      SimpleTimeInterval arcInterval = transport->interval;
      if (targetInvariant.bound < arcInterval.upperBound) {
        arcInterval.setUpperBound(targetInvariant.bound,
                                  targetInvariant.isBoundStrict);
      }

      auto arcFire =
          arcFiringDates(arcInterval, transport->weight, *place->tokens);

      auto newFiringIntervals = Util::setIntersection(firingIntervals, arcFire);
      delete[] firingIntervals.arr;
      delete[] arcFire.arr;
      firingIntervals = newFiringIntervals;

      if (firingIntervals.empty()) {
        firingIntervals.ownsArray = false;
        return firingIntervals;
      }
    }

    firingIntervals.ownsArray = false;
    return firingIntervals;
  }

  // NOTE: Remember to delete the interval array within the returned
  // SimpleDynamicArray when it is no longer needed
  SimpleDynamicArray<Util::SimpleInterval>
  arcFiringDates(SimpleTimeInterval time_interval, uint32_t weight,
                 SimpleDynamicArray<SimpleRealToken *> &tokens) {

    // std::cout << "Arc firing dates" << std::endl;
    Util::SimpleInterval arcInterval(time_interval.lowerBound,
                                     time_interval.upperBound);
    size_t total_tokens = 0;
    // std::cout << "tokens size: " << tokens.size << std::endl;
    for (size_t i = 0; i < tokens.size; i++) {
      total_tokens += tokens.get(i)->count;
    }
    // std::cout << "Arc firing dates" << std::endl;
    if (total_tokens < weight) {
      auto disabled = SimpleDynamicArray<Util::SimpleInterval>();
      disabled.ownsArray = false;
      return disabled;
    }

    SimpleDynamicArray<Util::SimpleInterval> firingIntervals =
        SimpleDynamicArray<Util::SimpleInterval>(10);

    SimpleDeque<double> selected = SimpleDeque<double>();
    for (size_t i = 0; i < tokens.size; i++) {
      for (int j = 0; j < tokens.get(i)->count; j++) {
        // print tokens get i count
        // std::cout << "Tokens get i count: " << tokens.get(i)->count
        //           << std::endl;
        // std::cout << "Tokens get i age: " << tokens.get(i)->age << std::endl;
        // std::cout << "Before push_back" << std::endl;
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

    firingIntervals.ownsArray = false;
    return firingIntervals;
  }

  SimpleDynamicArray<SimpleRealToken *> *
  removeRandom(SimpleDynamicArray<SimpleRealToken *> &tokenList,
               const SimpleTimeInterval &interval, const int weight) {
    // std::cout << "Remove random method is being called" << std::endl;
    auto res = new SimpleDynamicArray<SimpleRealToken *>(10);
    int remaning = weight;
    std::uniform_int_distribution<> randomTokenIndex(0, tokenList.size - 1);
    size_t tok_index = randomTokenIndex(rng);
    size_t tested = 0;

    while (remaning > 0 && tested < tokenList.size) {
      SimpleRealToken *token = tokenList.get(tok_index);
      if (interval.contains(token->age)) {
        res->add(new SimpleRealToken{.age = token->age, .count = 1});
        remaning--;
        tokenList.get(tok_index)->remove(1);
        if (tokenList.get(tok_index)->count == 0) {
          delete tokenList.get(tok_index);
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

  SimpleDynamicArray<SimpleRealToken *> *
  removeYoungest(SimpleDynamicArray<SimpleRealToken *> &tokenList,
                 const SimpleTimeInterval &interval, const int weight) {
    // std::cout << "Remove youngest method is being called" << std::endl;

    auto res = new SimpleDynamicArray<SimpleRealToken *>(10);
    int remaining = weight;
    // print initial remaining
    // std::cout << "Initial remaining: " << remaining << std::endl;
    // print token list size
    // std::cout << "Token list size: " << tokenList.size << std::endl;
    for (size_t i = 0; i < tokenList.size; i++) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!interval.contains(age)) {
        // print interval
        // std::cout << "Interval: " << interval.lowerBound << ", " <<
        // interval.upperBound << std::endl; print !interval.contains(age)
        // std::cout << "!interval.contains(age): " << !interval.contains(age)
        // << std::endl; print age std::cout << "Age: " << age << std::endl;
        continue;
      }
      int count = token->count;
      // std::cout << "Count: " << count << std::endl;
      if (count >= remaining) {
        // std::cout << "Count is greater or equal to remaining" << std::endl;
        res->add(new SimpleRealToken{.age = age, .count = remaining});
        token->remove(remaining);
        if (token->count == 0) {
          delete token;
          tokenList.remove(i);
        }
        remaining = 0;
        break;
      } else {
        // std::cout << "Count is less than remaining" << std::endl;
        res->add(new SimpleRealToken{.age = age, .count = count});
        remaining -= count;
        delete token;
        tokenList.remove(i);
      }
    }

    // print remaining
    //  std::cout << "Remaining: " << remaining << std::endl;
    assert(remaining == 0);
    return res;
  }

  // NOTE: Double check this method to ensure it is correct
  SimpleDynamicArray<SimpleRealToken *> *
  removeOldest(SimpleDynamicArray<SimpleRealToken *> &tokenList,
               const SimpleTimeInterval &timeInterval, const int weight) {

    // std::cout << "Remove oldest method is being called" << std::endl;
    auto res = new SimpleDynamicArray<SimpleRealToken *>(10);
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
        res->add(new SimpleRealToken{.age = age, .count = remaining});
        token->remove(remaining);
        if (token->count == 0) {
          delete token;
          tokenList.remove(i);
        }
        remaining = 0;
        break;
      } else {
        res->add(new SimpleRealToken{.age = age, .count = count});
        remaining -= count;
        delete token;
        tokenList.remove(i);
      }
    }
    assert(remaining == 0);
    return res;
  }

  bool fire(SimpleTimedTransition *transition) {
    if (transition == nullptr) {
      return false;
    }

    SimpleRealPlace **placeList = parent->places;

    for (size_t i = 0; i < transition->presetLength; i++) {
      SimpleTimedInputArc *input = transition->preset[i];
      SimpleRealPlace *place = placeList[input->inputPlace.index];
      SimpleDynamicArray<SimpleRealToken *> *tokenList = place->tokens;
      switch (transition->_firingMode) {
      case SimpleSMC::FiringMode::Random:
        removeRandom(*tokenList, input->interval, input->weight);
        break;
      case SimpleSMC::FiringMode::Oldest:
        removeOldest(*tokenList, input->interval, input->weight);
        break;
      case SimpleSMC::FiringMode::Youngest:
        // std::cout << "Preset youngest was called" << std::endl;
        removeYoungest(*tokenList, input->interval, input->weight);
        break;
      default:
        removeOldest(*tokenList, input->interval, input->weight);
        break;
      }
    }

    auto toCreate =
        SimpleDynamicArray<std::pair<SimpleTimedPlace *, SimpleRealToken *>>(
            10);
    for (size_t i = 0; i < transition->transportArcsLength; i++) {
      auto transport = transition->transportArcs[i];
      int destInv = transport->destination.timeInvariant.bound;
      SimpleRealPlace *place = placeList[transport->source.index];
      SimpleDynamicArray<SimpleRealToken *> *tokenList = place->tokens;
      SimpleDynamicArray<SimpleRealToken *> *consumed;
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
        // std::cout << "Transport youngest was called" << std::endl;
        consumed = removeYoungest(*tokenList, arcInterval, transport->weight);
        break;
      default:
        consumed = removeOldest(*tokenList, arcInterval, transport->weight);
        break;
      }
      // print consumed
      // std::cout << "Consumed size: " << consumed.size << std::endl;
      // std::cout << "Consumed capacity: " << consumed.capacity << std::endl;
      for (size_t j = 0; j < consumed->size; j++) {
        toCreate.add(
            std::make_pair(&(transport->destination), consumed->get(j)));
      }

      delete consumed;
    }

    for (size_t i = 0; i < transition->postsetLength; i++) {
      SimpleTimedPlace *place = transition->postset[i]->outputPlace;
      SimpleTimedOutputArc *post = transition->postset[i];
      auto token = new SimpleRealToken{.age = 0.0,
                                       .count = static_cast<int>(post->weight)};
      parent->addTokenInPlace(*place, *token);
    }
    for (size_t i = 0; i < toCreate.size; i++) {
      auto [place, token] = toCreate.get(i);
      parent->addTokenInPlace(*place, *token);
    }

    return true;
  }
};

struct AtlerRunResult2 {
  bool maximal = false;
  SimpleTimedArcPetriNet* stapn;
  SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval> *>
      *transitionIntervals;
  SimpleDynamicArray<double> *datesSampled;
  SimpleRealMarking *realMarking;
  double totalTime = 0;
  int totalSteps = 0;

  uint numericPrecision = 0;
  std::ranlux48 rng;

  AtlerRunResult2(SimpleTimedArcPetriNet *stapn, SimpleRealMarking* srm,
                  const unsigned int numericPrecision = 0)
      : stapn(stapn), realMarking(new SimpleRealMarking(*srm)), numericPrecision(numericPrecision) {

    // RNG setup
    std::random_device rd;
    rng = std::ranlux48(rd());

    // Initialize transition intervals
    transitionIntervals =
        new SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval> *>(
            stapn->transitionsLength);

    prepare();
  }

  AtlerRunResult2(const AtlerRunResult2 &other) {
      // set base properties
      stapn = new SimpleTimedArcPetriNet(*other.stapn);
      numericPrecision = other.numericPrecision;

      transitionIntervals = new SimpleDynamicArray<SimpleDynamicArray<Util::SimpleInterval> *>(other.transitionIntervals->size);
      for (size_t i = 0; i < other.transitionIntervals->size; i++) {
        transitionIntervals->add(new SimpleDynamicArray<Util::SimpleInterval>(*other.transitionIntervals->get(i)));
      }
      realMarking = new SimpleRealMarking(*other.realMarking);

      // random number generator
      std::random_device rd;
      rng = std::ranlux48(rd());

      // run reset
      reset();
  }

  ~AtlerRunResult2() {
    for (size_t i = 0; i < transitionIntervals->size; i++) {
      delete transitionIntervals->get(i);
    }
    delete transitionIntervals;
    delete datesSampled;
    delete realMarking;
    delete stapn;
  }

  void run() {}

  void prepare() {
    double originMaxDelay = realMarking->availableDelay();

    auto invIntervals = SimpleDynamicArray<Util::SimpleInterval>(10);
    invIntervals.add(Util::SimpleInterval(0, originMaxDelay));

    for (size_t i = 0; i < stapn->transitionsLength; i++) {
      auto *transition = stapn->transitions[i];
      if (transition->getPresetSize() == 0 && transition->inhibitorArcs == 0) {
        transitionIntervals->add(
            new SimpleDynamicArray<Util::SimpleInterval>(invIntervals));
      } else {
        SimpleDynamicArray<Util::SimpleInterval> firingDates =
            transitionFiringDates(*transition);

        auto intersection = Util::setIntersection(firingDates, invIntervals);

        transitionIntervals->add(
            new SimpleDynamicArray<Util::SimpleInterval>(intersection));
      }
    }
    reset();
  }

  void reset() {
    datesSampled = new SimpleDynamicArray<double>(transitionIntervals->size);
    for (size_t i = 0; i < transitionIntervals->size; i++) {
      datesSampled->add(std::numeric_limits<double>::infinity());
    }

    bool deadlock = true;
    for (size_t i = 0; i < datesSampled->size; i++) {
      auto *intervals = transitionIntervals->get(i);
      if (!intervals->empty() && intervals->get(0).lower() == 0) {
        const SimpleSMC::Distribution distribution =
            stapn->transitions[i]->distribution;
        datesSampled->set(i, distribution.sample(rng, numericPrecision));
      }
      deadlock &= transitionIntervals->get(i)->empty() ||
                  (transitionIntervals->get(i)->size == 1 &&
                   transitionIntervals->get(i)->get(0).lower() == 0 &&
                   datesSampled->get(i) == 0);
    }

    realMarking->deadlocked = deadlock;
  }

  // next function
  bool next() {
    auto [winner, delay] = getWinnerTransitionAndDelay();

    if (delay == std::numeric_limits<double>::infinity()) {
      // print delay is infinity
      // std::cout << "Delay is infinity/Deadlocked" << std::endl;
      maximal = true;
      return false;
    }

    realMarking->deltaAge(delay);
    totalTime += delay;

    if (winner != nullptr) {
      totalSteps++;
      datesSampled->set(winner->index, std::numeric_limits<double>::infinity());
      fire(winner);
    }

    for (size_t i = 0; i < transitionIntervals->size; i++) {
      double date = datesSampled->get(i);
      double newVal = (date == std::numeric_limits<double>::infinity())
                          ? std::numeric_limits<double>::infinity()
                          : date - delay;
      datesSampled->set(i, newVal);
    }

    refreshTransitionsIntervals();
    return true;
  }

  // refresh intervals
  void refreshTransitionsIntervals() {
    double max_delay = realMarking->availableDelay();
    auto invIntervals = SimpleDynamicArray<Util::SimpleInterval>(10);
    invIntervals.add(Util::SimpleInterval(0, max_delay));
    bool deadlocked = true;

    for (size_t i = 0; i < stapn->transitionsLength; i++) {
      auto *transition = stapn->transitions[i];
      int index = transition->index;
      if (transition->getPresetSize() == 0 &&
          transition->inhibitorArcsLength == 0) {
        delete[] transitionIntervals->get(i);
        transitionIntervals->set(
            i, new SimpleDynamicArray<Util::SimpleInterval>(invIntervals));
      } else {
        // print i and the length of transitionIntervals
        SimpleDynamicArray<Util::SimpleInterval> firingDates =
            transitionFiringDates(*transition);

        auto intersection = Util::setIntersection(firingDates, invIntervals);

        delete transitionIntervals->get(i);
        transitionIntervals->set(
            i, new SimpleDynamicArray<Util::SimpleInterval>(intersection));
      }

      bool enabled = (!transitionIntervals->get(i)->empty()) &&
                     (transitionIntervals->get(i)->get(0).lower() == 0);
      bool newlyEnabled = enabled && (datesSampled->get(i) ==
                                      std::numeric_limits<double>::infinity());
      bool reachedUpper = enabled && !newlyEnabled &&
                          (transitionIntervals->get(i)->get(0).upper() == 0) &&
                          (datesSampled->get(i) > 0);

      if (!enabled || reachedUpper) {
        datesSampled->set(i, std::numeric_limits<double>::infinity());
      } else if (newlyEnabled) {
        const auto distribution = stapn->transitions[i]->distribution;
        double date = distribution.sample(rng, numericPrecision);
        // print transitionIntervals.get(i).get(0)
        if (transitionIntervals->get(i)->get(0).upper() > 0 || date == 0) {
          datesSampled->set(i, date);
        }
      }
      deadlocked &= transitionIntervals->get(i)->empty() ||
                    (transitionIntervals->get(i)->size == 1 &&
                     transitionIntervals->get(i)->get(0).lower() == 0 &&
                     datesSampled->get(i) > 0);
    }

    realMarking->deadlocked = deadlocked;
  }

  // get winner transtion
  std::pair<SimpleTimedTransition *, double> getWinnerTransitionAndDelay() {
    SimpleDynamicArray<size_t> winner_indexes(10);
    double date_min = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < transitionIntervals->size; i++) {
      auto intervals = transitionIntervals->get(i);
      if (intervals->empty())
        continue;
      double date = std::numeric_limits<double>::infinity();
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
      date = std::min(datesSampled->get(i), date);
      if (date < date_min) {
        date_min = date;
        winner_indexes.clear();
      }
      if (datesSampled->get(i) == date_min) {
        winner_indexes.add(i);
      }
    }
    SimpleTimedTransition *winner;
    if (winner_indexes.empty()) {
      winner = nullptr;
    } else if (winner_indexes.size == 1) {
      winner = stapn->transitions[winner_indexes.get(0)];
    } else {
      winner = chooseWeightedWinner(winner_indexes);
    }
    return std::make_pair(winner, date_min);
  }
  // choose weightedWinner

  SimpleTimedTransition *
  chooseWeightedWinner(const SimpleDynamicArray<size_t> winner_indexes) {
    double total_weight = 0;
    SimpleDynamicArray<size_t> infinite_weights(10);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      double priority = stapn->transitions[candidate]->_weight;
      if (priority == std::numeric_limits<double>::infinity()) {
        infinite_weights.add(candidate);
      } else {
        total_weight += priority;
      }
    }

    if (!infinite_weights.empty()) {
      int winner_index =
          std::uniform_int_distribution<>(0, infinite_weights.size - 1)(rng);
      return stapn->transitions[infinite_weights.get(winner_index)];
    }
    if (total_weight == 0) {
      int winner_index =
          std::uniform_int_distribution<>(0, winner_indexes.size - 1)(rng);
      return stapn->transitions[winner_indexes.get(winner_index)];
    }
    double winning_weight =
        std::uniform_real_distribution<>(0.0, total_weight)(rng);
    for (size_t i = 0; i < winner_indexes.size; i++) {
      auto candidate = winner_indexes.get(i);
      SimpleTimedTransition *transition = stapn->transitions[candidate];
      winning_weight -= transition->_weight;
      if (winning_weight <= 0) {
        return transition;
      }
    }

    return stapn->transitions[winner_indexes.get(0)];
  }

  SimpleDynamicArray<Util::SimpleInterval>
  transitionFiringDates(const SimpleTimedTransition &transition) {
    auto initialFiringIntervals = SimpleDynamicArray<Util::SimpleInterval>(10);

    initialFiringIntervals.add(
        Util::SimpleInterval(0, std::numeric_limits<double>::infinity()));

    auto firingIntervalsStack =
        SimpleDeque<SimpleDynamicArray<Util::SimpleInterval>>();
    firingIntervalsStack.push_front(initialFiringIntervals);

    auto disabled = SimpleDynamicArray<Util::SimpleInterval>();

    // for each inhibitor arc
    for (size_t i = 0; i < transition.inhibitorArcsLength; i++) {
      auto inhib = transition.inhibitorArcs[i];
      if (realMarking->numberOfTokensInPlace(inhib->inputPlace.index) >=
          inhib->weight) {
        return SimpleDynamicArray(disabled);
      }
    }

    for (size_t i = 0; i < transition.presetLength; i++) {
      auto arc = transition.preset[i];
      SimpleRealPlace *place = realMarking->places[arc->inputPlace.index];
      if (place->isEmpty()) {
        return SimpleDynamicArray(disabled);
      }

      auto newFiringIntevals = Util::setIntersection(
          firingIntervalsStack.front->data,
          arcFiringDates(arc->interval, arc->weight, *place->tokens));
      firingIntervalsStack.push_front(newFiringIntevals);

      if (newFiringIntevals.empty()) {
        return SimpleDynamicArray(newFiringIntevals);
      }
    }

    for (size_t i = 0; i < transition.transportArcsLength; i++) {
      SimpleTimedTransportArc *transport = transition.transportArcs[i];
      auto place = realMarking->places[transport->source.index];

      if (place->isEmpty()) {
        return SimpleDynamicArray(disabled);
      }

      SimpleTimeInvariant targetInvariant =
          transport->destination.timeInvariant;
      SimpleTimeInterval arcInterval = transport->interval;
      if (targetInvariant.bound < arcInterval.upperBound) {
        arcInterval.setUpperBound(targetInvariant.bound,
                                  targetInvariant.isBoundStrict);
      }

      auto newFiringIntevals = Util::setIntersection(
          firingIntervalsStack.front->data,
          arcFiringDates(arcInterval, transport->weight, *place->tokens));
      firingIntervalsStack.push_front(newFiringIntevals);

      if (newFiringIntevals.empty()) {
        return SimpleDynamicArray(newFiringIntevals);
      }
    }

    return SimpleDynamicArray<Util::SimpleInterval>(
        firingIntervalsStack.front->data);
  }

  SimpleDynamicArray<Util::SimpleInterval>
  arcFiringDates(SimpleTimeInterval time_interval, uint32_t weight,
                 SimpleDynamicArray<SimpleRealToken *> &tokens) {
    Util::SimpleInterval arcInterval(time_interval.lowerBound,
                                     time_interval.upperBound);

    size_t total_tokens = 0;
    for (size_t i = 0; i < tokens.size; i++) {
      total_tokens += tokens.get(i)->count;
    }

    if (total_tokens < weight) {
      return SimpleDynamicArray<Util::SimpleInterval>();
    }

    SimpleDynamicArray<Util::SimpleInterval> firingIntervals =
        SimpleDynamicArray<Util::SimpleInterval>(10);

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
            Util::SimpleInterval shifted = arcInterval;
            shifted.delta(-selected.at(k));
            tokenSetInterval = Util::intersect(tokenSetInterval, shifted);
          }
          Util::setAdd(firingIntervals, tokenSetInterval);
        }
      }
    }

    return SimpleDynamicArray<Util::SimpleInterval>(firingIntervals);
  }

  SimpleDynamicArray<SimpleRealToken>
  removeRandom(SimpleDynamicArray<SimpleRealToken *> &tokenList,
               const SimpleTimeInterval &timeInterval, const int weight) {
    auto res = SimpleDynamicArray<SimpleRealToken>(10);
    int remaining = weight;
    std::uniform_int_distribution<> randomTokenIndex(0, tokenList.size - 1);
    size_t tok_index = randomTokenIndex(rng);
    size_t tested = 0;

    while (remaining > 0 && tested < tokenList.size) {
      SimpleRealToken *token = tokenList.get(tok_index);
      if (timeInterval.contains(token->age)) {
        res.add(SimpleRealToken{.age = token->age, .count = 1});
        remaining--;
        tokenList.get(tok_index)->remove(1);
        if (tokenList.get(tok_index)->count == 0) {
          delete tokenList.get(tok_index);
          tokenList.remove(tok_index);
          randomTokenIndex =
              std::uniform_int_distribution<>(0, tokenList.size - 1);
        }
        if (remaining > 0) {
          tok_index = randomTokenIndex(rng);
          tested = 0;
        }
      } else {
        tok_index = (tok_index + 1) % tokenList.size;
        tested++;
      }
    }
    assert(remaining == 0);
    return SimpleDynamicArray<SimpleRealToken>(res);
  }

  SimpleDynamicArray<SimpleRealToken>
  removeYoungest(SimpleDynamicArray<SimpleRealToken *> &tokenList,
                 const SimpleTimeInterval &timeInterval, const int weight) {
    auto res = SimpleDynamicArray<SimpleRealToken>(10);
    int remaining = weight;
    for (size_t i = 0; i < tokenList.size; i++) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!timeInterval.contains(age)) {
        continue;
      }
      int count = token->count;
      if (count >= remaining) {
        res.add(SimpleRealToken{.age = age, .count = remaining});
        token->remove(remaining);
        if (token->count == 0) {
          delete token;
          tokenList.remove(i);
        }
        remaining = 0;
        break;
      } else {
        res.add(SimpleRealToken{.age = age, .count = count});
        remaining -= count;
        delete token;
        tokenList.remove(i);
      }
    }
    assert(remaining == 0);
    return SimpleDynamicArray<SimpleRealToken>(res);
  }

  SimpleDynamicArray<SimpleRealToken>
  removeOldest(SimpleDynamicArray<SimpleRealToken *> &tokenList,
               const SimpleTimeInterval &timeInterval, const int weight) {
    auto res = SimpleDynamicArray<SimpleRealToken>(10);
    int remaining = weight;
    for (size_t i = 0; i < tokenList.size; i++) {
      auto token = tokenList.get(i);
      double age = token->age;
      if (!timeInterval.contains(age)) {
        continue;
      }
      int count = token->count;
      if (count >= remaining) {
        res.add(SimpleRealToken{.age = age, .count = remaining});
        token->remove(remaining);
        if (token->count == 0) {
          delete token;
          tokenList.remove(i);
        }
        remaining = 0;
        break;
      } else {
        res.add(SimpleRealToken{.age = age, .count = count});
        remaining -= count;
        delete token;
        tokenList.remove(i);
      }
    }

    return SimpleDynamicArray<SimpleRealToken>(res);
  }

  bool fire(SimpleTimedTransition *transition) {
    if (transition == nullptr) {
      return false;
    }

    SimpleRealPlace **placeList = realMarking->places;

    for (size_t i = 0; i < transition->presetLength; i++) {
      SimpleTimedInputArc *input = transition->preset[i];
      SimpleRealPlace *place = placeList[input->inputPlace.index];
      SimpleDynamicArray<SimpleRealToken *> *tokenList = place->tokens;
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
    }

    auto toCreate =
        SimpleDynamicArray<std::pair<SimpleTimedPlace *, SimpleRealToken>>(10);

    for (size_t i = 0; i < transition->transportArcsLength; i++) {
      auto transport = transition->transportArcs[i];
      int destInv = transport->destination.timeInvariant.bound;
      SimpleRealPlace *place = placeList[transport->source.index];
      SimpleDynamicArray<SimpleRealToken *> *tokenList = place->tokens;
      SimpleTimeInterval &arcInterval = transport->interval;
      SimpleDynamicArray<SimpleRealToken> consumed =
          switcher(transition, tokenList, arcInterval, transport->weight);

      if (destInv < arcInterval.upperBound) {
        arcInterval.setUpperBound(destInv, false);
      }

      // switch (transition->_firingMode) {
      // case SimpleSMC::FiringMode::Random: {
      //   consumed = removeRandom(*tokenList, arcInterval, transport->weight);
      //   break;
      // }
      // case SimpleSMC::FiringMode::Oldest: {
      //   consumed = removeOldest(*tokenList, arcInterval, transport->weight);
      //   break;
      // }
      // case SimpleSMC::FiringMode::Youngest: {
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
        toCreate.add(std::make_pair(&(transport->destination),
                                    SimpleRealToken{con.age, con.count}));
      }
    }

    for (size_t i = 0; i < transition->postsetLength; i++) {
      SimpleTimedPlace *place = transition->postset[i]->outputPlace;
      SimpleTimedOutputArc *post = transition->postset[i];
      auto token = SimpleRealToken{.age = 0.0,
                                       .count = static_cast<int>(post->weight)};
      realMarking->addTokenInPlace(*place, token);
    }
    for (size_t i = 0; i < toCreate.size; i++) {
      auto [place, token] = toCreate.get(i);
      realMarking->addTokenInPlace(*place, token);
    }

    return true;
  }

  SimpleDynamicArray<SimpleRealToken>
  switcher(SimpleTimedTransition *transition,
           SimpleDynamicArray<SimpleRealToken *> *tokens,
           SimpleTimeInterval &interval, int weight) {

    switch (transition->_firingMode) {
    case SimpleSMC::FiringMode::Random:
      return removeRandom(*tokens, interval, weight);
    case SimpleSMC::FiringMode::Oldest:
      return removeOldest(*tokens, interval, weight);
    case SimpleSMC::FiringMode::Youngest:
      return removeYoungest(*tokens, interval, weight);
    default:
      return removeOldest(*tokens, interval, weight);
    }
  }
};
} // namespace VerifyTAPN::Atler

#endif
