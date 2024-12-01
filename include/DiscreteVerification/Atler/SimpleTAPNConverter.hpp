#ifndef VERIFYYAPN_ATLER_SIMPLETIMEDARCPETRINET_CONVERTER_HPP_
#define VERIFYYAPN_ATLER_SIMPLETIMEDARCPETRINET_CONVERTER_HPP_

// TODO: Move logic to SimpleTAPNConverter.cpp
// Only keep definition of SimpleTAPNConverter

#include "Core/TAPN/TimedArcPetriNet.hpp"
#include "DiscreteVerification/Atler/SimpleRealMarking.hpp"
#include "DiscreteVerification/Atler/SimpleTimedArcPetriNet.hpp"
#include "DiscreteVerification/Atler/SimpleTimedInputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedOutputArc.hpp"
#include "DiscreteVerification/Atler/SimpleTimedTransition.hpp"
#include "DiscreteVerification/DataStructures/RealMarking.hpp"
#include <cstddef>
#include <unordered_map>

namespace VerifyTAPN {
namespace Atler {

class SimpleTAPNConverter {
public:
  static std::pair<SimpleTimedArcPetriNet, SimpleRealMarking> *
  convert(const TAPN::TimedArcPetriNet &tapn,
          DiscreteVerification::RealMarking &marking) {
    std::unordered_map<const TAPN::TimedPlace *, SimpleTimedPlace *> placeMap;
    std::unordered_map<const TAPN::TimedTransition *, SimpleTimedTransition *>
        transitionMap;
    std::unordered_map<const TAPN::TimedInputArc *, SimpleTimedInputArc *>
        inputArcMap;
    std::unordered_map<const TAPN::TransportArc *, SimpleTimedTransportArc *>
        transportArcMap;
    std::unordered_map<const TAPN::InhibitorArc *, SimpleTimedInhibitorArc *>
        inhibitorArcMap;
    std::unordered_map<const TAPN::OutputArc *, SimpleTimedOutputArc *>
        outputArcMap;

    auto places = new SimpleTimedPlace *[tapn.getPlaces().size()];
    for (size_t i = 0; i < tapn.getPlaces().size(); i++) {
      auto originalPlace = tapn.getPlaces()[i];
      places[i] = new SimpleTimedPlace(convertPlace(*originalPlace));
      placeMap[originalPlace] = places[i];
    }

    auto transitions =
        new SimpleTimedTransition *[tapn.getTransitions().size()];
    for (size_t i = 0; i < tapn.getTransitions().size(); i++) {
      auto originalTransition = tapn.getTransitions()[i];
      transitions[i] =
          new SimpleTimedTransition(convertTransition(*originalTransition));
      transitionMap[originalTransition] = transitions[i];
    }

    auto inputArcs = new SimpleTimedInputArc *[tapn.getInputArcs().size()];
    for (size_t i = 0; i < tapn.getInputArcs().size(); i++) {
      auto originalArc = tapn.getInputArcs()[i];
      inputArcs[i] = new SimpleTimedInputArc(
          convertInputArc(*originalArc, placeMap, transitionMap));
      inputArcMap[originalArc] = inputArcs[i];
    }

    auto outputArcs = new SimpleTimedOutputArc *[tapn.getOutputArcs().size()];
    for (size_t i = 0; i < tapn.getOutputArcs().size(); i++) {
      auto originalArc = tapn.getOutputArcs()[i];
      outputArcs[i] = new SimpleTimedOutputArc(
          convertOutputArc(*originalArc, placeMap, transitionMap));
      outputArcMap[originalArc] = outputArcs[i];
    }

    auto transportArcs =
        new SimpleTimedTransportArc *[tapn.getTransportArcs().size()];
    for (size_t i = 0; i < tapn.getTransportArcs().size(); i++) {
      auto originalArc = tapn.getTransportArcs()[i];
      transportArcs[i] = new SimpleTimedTransportArc(
          convertTransportArc(*originalArc, placeMap, transitionMap));
      transportArcMap[originalArc] = transportArcs[i];
    }

    auto inhibitorArcs =
        new SimpleTimedInhibitorArc *[tapn.getInhibitorArcs().size()];
    for (size_t i = 0; i < tapn.getInhibitorArcs().size(); i++) {
      auto originalArc = tapn.getInhibitorArcs()[i];
      inhibitorArcs[i] = new SimpleTimedInhibitorArc(
          convertInhibitorArc(*originalArc, placeMap, transitionMap));
      inhibitorArcMap[originalArc] = inhibitorArcs[i];
    }

    // Add the arcs to the places
    for (size_t i = 0; i < tapn.getPlaces().size(); i++) {
      auto originalPlace = tapn.getPlaces()[i];
      auto simplePlace = placeMap.at(originalPlace);

      auto inputArcs =
          new SimpleTimedInputArc *[originalPlace->getInputArcs().size()];
      for (size_t j = 0; j < originalPlace->getInputArcs().size(); j++) {
        auto originalArc = originalPlace->getInputArcs()[j];
        inputArcs[j] = inputArcMap.at(originalArc);
      }

      simplePlace->inputArcs = inputArcs;
      simplePlace->inputArcsLength = originalPlace->getInputArcs().size();

      auto transportArcs = new SimpleTimedTransportArc
          *[originalPlace->getTransportArcs().size()];
      for (size_t j = 0; j < originalPlace->getTransportArcs().size(); j++) {
        auto originalArc = originalPlace->getTransportArcs()[j];
        transportArcs[j] = transportArcMap.at(originalArc);
      }

      simplePlace->transportArcs = transportArcs;
      simplePlace->transportArcsLength =
          originalPlace->getTransportArcs().size();

      auto prodTransportArcs = new SimpleTimedTransportArc
          *[originalPlace->getProdTransportArcs().size()];
      for (size_t j = 0; j < originalPlace->getProdTransportArcs().size();
           j++) {
        auto originalArc = originalPlace->getProdTransportArcs()[j];
        prodTransportArcs[j] = transportArcMap.at(originalArc);
      }

      simplePlace->prodTransportArcs = prodTransportArcs;
      simplePlace->prodTransportArcsLength =
          originalPlace->getProdTransportArcs().size();

      auto inhibitorArcs = new SimpleTimedInhibitorArc
          *[originalPlace->getInhibitorArcs().size()];
      for (size_t j = 0; j < originalPlace->getInhibitorArcs().size(); j++) {
        auto originalArc = originalPlace->getInhibitorArcs()[j];
        inhibitorArcs[j] = inhibitorArcMap.at(originalArc);
      }

      simplePlace->inhibitorArcs = inhibitorArcs;
      simplePlace->inhibitorArcsLength =
          originalPlace->getInhibitorArcs().size();

      auto outputArcs =
          new SimpleTimedOutputArc *[originalPlace->getOutputArcs().size()];
      for (size_t j = 0; j < originalPlace->getOutputArcs().size(); j++) {
        auto originalArc = originalPlace->getOutputArcs()[j];
        outputArcs[j] = outputArcMap.at(originalArc);
      }

      simplePlace->outputArcs = outputArcs;
      simplePlace->outputArcsLength = originalPlace->getOutputArcs().size();
    }

    // Add the arcs to the transitions
    for (size_t i = 0; i < tapn.getTransitions().size(); i++) {
      auto originalTransition = tapn.getTransitions()[i];
      auto simpleTransition = transitionMap.at(originalTransition);

      auto inputArcs =
          new SimpleTimedInputArc *[originalTransition->getPreset().size()];
      for (size_t j = 0; j < originalTransition->getPreset().size(); j++) {
        auto originalArc = originalTransition->getPreset()[j];
        inputArcs[j] = inputArcMap.at(originalArc);
      }

      simpleTransition->preset = inputArcs;
      simpleTransition->presetLength = originalTransition->getPreset().size();

      auto outputArcs =
          new SimpleTimedOutputArc *[originalTransition->getPostset().size()];
      for (size_t j = 0; j < originalTransition->getPostset().size(); j++) {
        auto originalArc = originalTransition->getPostset()[j];
        outputArcs[j] = outputArcMap.at(originalArc);
      }

      simpleTransition->postset = outputArcs;
      simpleTransition->postsetLength = originalTransition->getPostset().size();

      auto transportArcs = new SimpleTimedTransportArc
          *[originalTransition->getTransportArcs().size()];
      for (size_t j = 0; j < originalTransition->getTransportArcs().size();
           j++) {
        auto originalArc = originalTransition->getTransportArcs()[j];
        transportArcs[j] = transportArcMap.at(originalArc);
      }

      simpleTransition->transportArcs = transportArcs;
      simpleTransition->transportArcsLength =
          originalTransition->getTransportArcs().size();

      auto inhibitorArcs = new SimpleTimedInhibitorArc
          *[originalTransition->getInhibitorArcs().size()];
      int inhibitorArcsLength = 0;
      for (size_t j = 0; j < originalTransition->getInhibitorArcs().size();
           j++) {
        auto originalArc = originalTransition->getInhibitorArcs()[j];
        inhibitorArcs[j] = inhibitorArcMap.at(originalArc);
      }

      simpleTransition->inhibitorArcs = inhibitorArcs;
      simpleTransition->inhibitorArcsLength =
          originalTransition->getInhibitorArcs().size();
    }

    // Create and return new SimpleTimedArcPetriNet
    auto simpleTapn = new SimpleTimedArcPetriNet();
    simpleTapn->places = places;
    simpleTapn->placesLength = tapn.getPlaces().size();
    simpleTapn->transitions = transitions;
    simpleTapn->transitionsLength = tapn.getTransitions().size();
    simpleTapn->inputArcs = inputArcs;
    simpleTapn->inputArcsLength = tapn.getInputArcs().size();
    simpleTapn->outputArcs = outputArcs;
    simpleTapn->outputArcsLength = tapn.getOutputArcs().size();
    simpleTapn->transportArcs = transportArcs;
    simpleTapn->transportArcsLength = tapn.getTransportArcs().size();
    simpleTapn->inhibitorArcs = inhibitorArcs;
    simpleTapn->inhibitorArcsLength = tapn.getTransportArcs().size();
    simpleTapn->maxConstant = tapn.getMaxConstant();
    simpleTapn->gcd = tapn.getGCD();

    // Create and return new SimpleRealMarking
    std::cout << "Converting Marking..." << std::endl;


    SimpleRealMarking srm;
    srm.placesLength = marking.getPlaceList().size();
    srm.places = new SimpleRealPlace[srm.placesLength];
    srm.deadlocked = marking.canDeadlock(tapn, false);
    srm.fromDelay = marking.getPreviousDelay();
    srm.generatedBy = marking.getGeneratedBy() != nullptr
                          ? transitionMap.at(marking.getGeneratedBy())
                          : nullptr;

    // Initialize SimpleRealPlace array with mapped places
    std::cout << "Converting Places..." << std::endl;
    auto placeLength = marking.getPlaceList().size();
    srm.places = new SimpleRealPlace[placeLength];
    for (size_t i = 0; i < placeLength; i++) {
      DiscreteVerification::RealPlace &realPlace = marking.getPlaceList()[i];

      // Get mapped place
      auto simplePlace = placeMap.at(realPlace.place);

      // Create and initialize SimpleRealToken array with converted tokens
      size_t tokenLength = realPlace.tokens.size();
      for (size_t j = 0; j < tokenLength; j++) {
          //print token count
          std::cout << "Token count: " << realPlace.tokens[j].getCount() << std::endl;
        srm.places[i].tokens->add(new SimpleRealToken{
            realPlace.tokens[j].getAge(), realPlace.tokens[j].getCount()});
      }

      // Set SimpleRealPlace fields
      srm.places[i].place = *simplePlace;
    }

    auto result = new std::pair<SimpleTimedArcPetriNet, SimpleRealMarking>(
        *simpleTapn, srm);
    return result;
  }

private:
  static SimpleTimedPlace convertPlace(const TAPN::TimedPlace &place) {
    SimpleTimedPlace simplePlace;
    simplePlace.index = place.getIndex();
    simplePlace.name = place.getName().c_str();
    simplePlace.id = place.getId().c_str();
    simplePlace.type = convertPlaceType(place.getType());
    simplePlace.timeInvariant = convertTimeInvariant(place.getInvariant());
    simplePlace.untimed = place.isUntimed();
    simplePlace.maxConstant = place.getMaxConstant();
    simplePlace.containsInhibitorArcs = place.hasInhibitorArcs();
    simplePlace._position = place.getPosition();
    return simplePlace;
  }

  static SimpleTimedTransition
  convertTransition(const TAPN::TimedTransition &transition) {
    SimpleTimedTransition simpleTransition;
    simpleTransition.index = transition.getIndex();
    simpleTransition.name = transition.getName().c_str();
    simpleTransition.id = transition.getId().c_str();
    simpleTransition.untimedPostset = transition.hasUntimedPostset();
    simpleTransition.urgent = transition.isUrgent();
    simpleTransition.controllable = transition.isControllable();
    simpleTransition._position = transition.getPosition();
    simpleTransition.distribution =
        convertDistribution(transition.getDistribution());
    simpleTransition._weight = transition.getWeight();
    simpleTransition._firingMode =
        convertFiringMode(transition.getFiringMode());
    return simpleTransition;
  }

  static SimpleTimedInputArc convertInputArc(
      const TAPN::TimedInputArc &arc,
      const std::unordered_map<const TAPN::TimedPlace *, SimpleTimedPlace *>
          &placeMap,
      const std::unordered_map<const TAPN::TimedTransition *,
                               SimpleTimedTransition *> &transitionMap) {
    return {convertTimeInterval(arc.getInterval()),
            *placeMap.at(&arc.getInputPlace()),
            *transitionMap.at(&arc.getOutputTransition()), arc.getWeight()};
  }

  static SimpleTimedOutputArc convertOutputArc(
      const TAPN::OutputArc &arc,
      const std::unordered_map<const TAPN::TimedPlace *, SimpleTimedPlace *>
          &placeMap,
      const std::unordered_map<const TAPN::TimedTransition *,
                               SimpleTimedTransition *> &transitionMap) {
    return {transitionMap.at(&arc.getInputTransition()),
            placeMap.at(&arc.getOutputPlace()), arc.getWeight()};
  }

  static SimpleTimedTransportArc convertTransportArc(
      const TAPN::TransportArc &arc,
      const std::unordered_map<const TAPN::TimedPlace *, SimpleTimedPlace *>
          &placeMap,
      const std::unordered_map<const TAPN::TimedTransition *,
                               SimpleTimedTransition *> &transitionMap) {
    return {
        convertTimeInterval(arc.getInterval()), *placeMap.at(&arc.getSource()),
        *transitionMap.at(&arc.getTransition()),
        *placeMap.at(&arc.getDestination()), static_cast<int>(arc.getWeight())};
  }

  static SimpleTimedInhibitorArc convertInhibitorArc(
      const TAPN::InhibitorArc &arc,
      const std::unordered_map<const TAPN::TimedPlace *, SimpleTimedPlace *>
          &placeMap,
      const std::unordered_map<const TAPN::TimedTransition *,
                               SimpleTimedTransition *> &transitionMap) {
    return {*placeMap.at(&arc.getInputPlace()),
            *transitionMap.at(&arc.getOutputTransition()), arc.getWeight()};
  }

  static PlaceType convertPlaceType(TAPN::PlaceType type) {
    switch (type) {
    case TAPN::PlaceType::Inv:
      return PlaceType::Inv;
    case TAPN::PlaceType::Dead:
      return PlaceType::Dead;
    case TAPN::PlaceType::Std:
      return PlaceType::Std;
    default:
      return PlaceType::Std;
    }
  }

  static SimpleTimeInterval
  convertTimeInterval(const TAPN::TimeInterval &interval) {
    return {interval.isLowerBoundStrict(), interval.getLowerBound(),
            interval.getUpperBound(), interval.isUpperBoundStrict()};
  }

  static SimpleTimeInvariant
  convertTimeInvariant(const TAPN::TimeInvariant &invariant) {
      //print the invariant
    return {invariant.isBoundStrict(), invariant.getBound()};
  }

  static SimpleSMC::Distribution
  convertDistribution(const SMC::Distribution &distribution) {
    SimpleSMC::Distribution simpleDistribution;
    simpleDistribution.type =
        static_cast<SimpleSMC::DistributionType>(distribution.type);
    simpleDistribution.parameters =
        *reinterpret_cast<const SimpleSMC::DistributionParameters *>(
            &distribution.parameters);
    return simpleDistribution;
  }

  static SimpleSMC::FiringMode convertFiringMode(const SMC::FiringMode &mode) {
    return static_cast<SimpleSMC::FiringMode>(mode);
  }
};

} // namespace Atler
} // namespace VerifyTAPN

#endif
