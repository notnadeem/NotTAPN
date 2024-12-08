#ifndef VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CONVERTER_CUH_
#define VERIFYYAPN_ATLER_CUDATIMEDARCPETRINET_CONVERTER_CUH_

// TODO: Move logic to CudaTAPNConverter.cu
// Only keep definition of CudaTAPNConverter

#include "Core/TAPN/TimedArcPetriNet.hpp"
#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"
#include "DiscreteVerification/Cuda/CudaTimedArcPetriNet.cuh"
#include "DiscreteVerification/Cuda/CudaTimedInputArc.cuh"
#include "DiscreteVerification/Cuda/CudaTimedOutputArc.cuh"
#include "DiscreteVerification/DataStructures/RealMarking.hpp"
#include "cuda_runtime.h"

namespace VerifyTAPN {
namespace Cuda {

class CudaTAPNConverter {
public:
  static std::pair<CudaTimedArcPetriNet, CudaRealMarking *> *convert(const TAPN::TimedArcPetriNet &tapn,
                                                                     DiscreteVerification::RealMarking &marking) {
    std::unordered_map<const TAPN::TimedPlace *, CudaTimedPlace *> placeMap;
    std::unordered_map<const TAPN::TimedTransition *, CudaTimedTransition *> transitionMap;
    std::unordered_map<const TAPN::TimedInputArc *, CudaTimedInputArc *> inputArcMap;
    std::unordered_map<const TAPN::TransportArc *, CudaTimedTransportArc *> transportArcMap;
    std::unordered_map<const TAPN::InhibitorArc *, CudaTimedInhibitorArc *> inhibitorArcMap;
    std::unordered_map<const TAPN::OutputArc *, CudaTimedOutputArc *> outputArcMap;

    auto places = new CudaTimedPlace *[tapn.getPlaces().size()];
    for (size_t i = 0; i < tapn.getPlaces().size(); i++) {
      auto originalPlace = tapn.getPlaces()[i];
      places[i] = new CudaTimedPlace(convertPlace(*originalPlace));
      placeMap[originalPlace] = places[i];
    }

    auto transitions = new CudaTimedTransition *[tapn.getTransitions().size()];
    for (size_t i = 0; i < tapn.getTransitions().size(); i++) {
      auto originalTransition = tapn.getTransitions()[i];
      transitions[i] = new CudaTimedTransition(convertTransition(*originalTransition));
      transitionMap[originalTransition] = transitions[i];
    }

    auto inputArcs = new CudaTimedInputArc *[tapn.getInputArcs().size()];
    for (size_t i = 0; i < tapn.getInputArcs().size(); i++) {
      auto originalArc = tapn.getInputArcs()[i];
      inputArcs[i] = new CudaTimedInputArc(convertInputArc(*originalArc, placeMap, transitionMap));
      inputArcMap[originalArc] = inputArcs[i];
    }

    auto outputArcs = new CudaTimedOutputArc *[tapn.getOutputArcs().size()];
    for (size_t i = 0; i < tapn.getOutputArcs().size(); i++) {
      auto originalArc = tapn.getOutputArcs()[i];
      outputArcs[i] = new CudaTimedOutputArc(convertOutputArc(*originalArc, placeMap, transitionMap));
      outputArcMap[originalArc] = outputArcs[i];
    }

    auto transportArcs = new CudaTimedTransportArc *[tapn.getTransportArcs().size()];
    for (size_t i = 0; i < tapn.getTransportArcs().size(); i++) {
      auto originalArc = tapn.getTransportArcs()[i];
      transportArcs[i] = new CudaTimedTransportArc(convertTransportArc(*originalArc, placeMap, transitionMap));
      transportArcMap[originalArc] = transportArcs[i];
    }

    auto inhibitorArcs = new CudaTimedInhibitorArc *[tapn.getInhibitorArcs().size()];
    for (size_t i = 0; i < tapn.getInhibitorArcs().size(); i++) {
      auto originalArc = tapn.getInhibitorArcs()[i];
      inhibitorArcs[i] = new CudaTimedInhibitorArc(convertInhibitorArc(*originalArc, placeMap, transitionMap));
      inhibitorArcMap[originalArc] = inhibitorArcs[i];
    }

    // Add the arcs to the places
    for (size_t i = 0; i < tapn.getPlaces().size(); i++) {
      auto originalPlace = tapn.getPlaces()[i];
      auto cudaPlace = placeMap.at(originalPlace);

      auto inputArcs = new CudaTimedInputArc *[originalPlace->getInputArcs().size()];
      for (size_t j = 0; j < originalPlace->getInputArcs().size(); j++) {
        auto originalArc = originalPlace->getInputArcs()[j];
        inputArcs[j] = inputArcMap.at(originalArc);
      }

      cudaPlace->inputArcs = inputArcs;
      cudaPlace->inputArcsLength = originalPlace->getInputArcs().size();

      auto transportArcs = new CudaTimedTransportArc *[originalPlace->getTransportArcs().size()];
      for (size_t j = 0; j < originalPlace->getTransportArcs().size(); j++) {
        auto originalArc = originalPlace->getTransportArcs()[j];
        transportArcs[j] = transportArcMap.at(originalArc);
      }

      cudaPlace->transportArcs = transportArcs;
      cudaPlace->transportArcsLength = originalPlace->getTransportArcs().size();

      auto prodTransportArcs = new CudaTimedTransportArc *[originalPlace->getProdTransportArcs().size()];
      for (size_t j = 0; j < originalPlace->getProdTransportArcs().size(); j++) {
        auto originalArc = originalPlace->getProdTransportArcs()[j];
        prodTransportArcs[j] = transportArcMap.at(originalArc);
      }

      cudaPlace->prodTransportArcs = prodTransportArcs;
      cudaPlace->prodTransportArcsLength = originalPlace->getProdTransportArcs().size();

      auto inhibitorArcs = new CudaTimedInhibitorArc *[originalPlace->getInhibitorArcs().size()];
      for (size_t j = 0; j < originalPlace->getInhibitorArcs().size(); j++) {
        auto originalArc = originalPlace->getInhibitorArcs()[j];
        inhibitorArcs[j] = inhibitorArcMap.at(originalArc);
      }

      cudaPlace->inhibitorArcs = inhibitorArcs;
      cudaPlace->inhibitorArcsLength = originalPlace->getInhibitorArcs().size();

      auto outputArcs = new CudaTimedOutputArc *[originalPlace->getOutputArcs().size()];
      for (size_t j = 0; j < originalPlace->getOutputArcs().size(); j++) {
        auto originalArc = originalPlace->getOutputArcs()[j];
        outputArcs[j] = outputArcMap.at(originalArc);
      }

      cudaPlace->outputArcs = outputArcs;
      cudaPlace->outputArcsLength = originalPlace->getOutputArcs().size();
    }

    // Add the arcs to the transitions
    for (size_t i = 0; i < tapn.getTransitions().size(); i++) {
      auto originalTransition = tapn.getTransitions()[i];
      auto cudaTransition = transitionMap.at(originalTransition);
      printf("Transition: %s\n", originalTransition->getName().c_str());
      auto inputArcs = new CudaTimedInputArc *[originalTransition->getPreset().size()];
      for (size_t j = 0; j < originalTransition->getPreset().size(); j++) {
        auto originalArc = originalTransition->getPreset()[j];
        inputArcs[j] = inputArcMap.at(originalArc);
      }

      cudaTransition->preset = inputArcs;
      cudaTransition->presetLength = originalTransition->getPreset().size();
      for (size_t i = 0; i < cudaTransition->presetLength; i++) {
        printf("Preset: %s\n", cudaTransition->preset[i]->inputPlace->name);
      }
      printf("Preset length: %d\n", cudaTransition->presetLength);

      auto outputArcs = new CudaTimedOutputArc *[originalTransition->getPostset().size()];
      for (size_t j = 0; j < originalTransition->getPostset().size(); j++) {
        auto originalArc = originalTransition->getPostset()[j];
        outputArcs[j] = outputArcMap.at(originalArc);
      }

      cudaTransition->postset = outputArcs;
      cudaTransition->postsetLength = originalTransition->getPostset().size();

      auto transportArcs = new CudaTimedTransportArc *[originalTransition->getTransportArcs().size()];
      for (size_t j = 0; j < originalTransition->getTransportArcs().size(); j++) {
        auto originalArc = originalTransition->getTransportArcs()[j];
        transportArcs[j] = transportArcMap.at(originalArc);
      }

      cudaTransition->transportArcs = transportArcs;
      cudaTransition->transportArcsLength = originalTransition->getTransportArcs().size();

      auto inhibitorArcs = new CudaTimedInhibitorArc *[originalTransition->getInhibitorArcs().size()];
      int inhibitorArcsLength = 0;
      for (size_t j = 0; j < originalTransition->getInhibitorArcs().size(); j++) {
        auto originalArc = originalTransition->getInhibitorArcs()[j];
        inhibitorArcs[j] = inhibitorArcMap.at(originalArc);
      }

      cudaTransition->inhibitorArcs = inhibitorArcs;
      cudaTransition->inhibitorArcsLength = originalTransition->getInhibitorArcs().size();
    }

    // Create and return new CudaTimedArcPetriNet
    auto cudaTapn = new CudaTimedArcPetriNet();
    cudaTapn->places = places;
    cudaTapn->placesLength = tapn.getPlaces().size();
    cudaTapn->transitions = transitions;
    cudaTapn->transitionsLength = tapn.getTransitions().size();
    cudaTapn->inputArcs = inputArcs;
    cudaTapn->inputArcsLength = tapn.getInputArcs().size();
    cudaTapn->outputArcs = outputArcs;
    cudaTapn->outputArcsLength = tapn.getOutputArcs().size();
    cudaTapn->transportArcs = transportArcs;
    cudaTapn->transportArcsLength = tapn.getTransportArcs().size();
    cudaTapn->inhibitorArcs = inhibitorArcs;
    cudaTapn->inhibitorArcsLength = tapn.getTransportArcs().size();
    cudaTapn->maxConstant = tapn.getMaxConstant();
    cudaTapn->gcd = tapn.getGCD();

    // Create and return new CudaRealMarking
    std::cout << "Converting Marking..." << std::endl;

    auto srm = new CudaRealMarking();
    srm->placesLength = marking.getPlaceList().size();
    srm->places = new CudaRealPlace *[srm->placesLength];
    srm->deadlocked = marking.canDeadlock(tapn, false);
    // srm.fromDelay = marking.getPreviousDelay();
    // srm.generatedBy = marking.getGeneratedBy() != nullptr
    //                       ? transitionMap.at(marking.getGeneratedBy())
    // : nullptr;

    // Initialize CudaRealPlace array with mapped places
    std::cout << "Converting Places..." << std::endl;
    auto placeLength = marking.getPlaceList().size();
    for (size_t i = 0; i < placeLength; i++) {
      DiscreteVerification::RealPlace &realPlace = marking.getPlaceList()[i];

      // Get mapped place
      auto cudaPlace = placeMap.at(realPlace.place);

      // Set new place
      srm->places[i] = new CudaRealPlace();

      // Create and initialize CudaRealToken array with converted tokens
      size_t tokenLength = realPlace.tokens.size();
      for (size_t j = 0; j < tokenLength; j++) {
        // print token count
        std::cout << "Token count: " << realPlace.tokens[j].getCount() << std::endl;
        srm->places[i]->tokens->add(new CudaRealToken{realPlace.tokens[j].getAge(), realPlace.tokens[j].getCount()});
      }

      // Set CudaRealPlace fields
      srm->places[i]->place = cudaPlace;
    }

    auto result = new std::pair<CudaTimedArcPetriNet, CudaRealMarking *>(*cudaTapn, srm);
    return result;
  }

private:
  static CudaTimedPlace convertPlace(const TAPN::TimedPlace &place) {
    CudaTimedPlace cudaPlace;
    cudaPlace.index = place.getIndex();
    cudaPlace.name = place.getName().c_str();
    cudaPlace.nameLength = place.getName().size();
    cudaPlace.id = place.getId().c_str();
    cudaPlace.idLength = place.getId().size();
    cudaPlace.type = convertPlaceType(place.getType());
    cudaPlace.timeInvariant = convertTimeInvariant(place.getInvariant());
    cudaPlace.untimed = place.isUntimed();
    cudaPlace.maxConstant = place.getMaxConstant();
    cudaPlace.containsInhibitorArcs = place.hasInhibitorArcs();
    cudaPlace._position = place.getPosition();
    return cudaPlace;
  }

  static CudaTimedTransition convertTransition(const TAPN::TimedTransition &transition) {
    CudaTimedTransition cudaTransition;
    cudaTransition.index = transition.getIndex();
    cudaTransition.name = transition.getName().c_str();
    cudaTransition.nameLength = transition.getName().size();
    cudaTransition.id = transition.getId().c_str();
    cudaTransition.idLength = transition.getId().size();
    cudaTransition.untimedPostset = transition.hasUntimedPostset();
    cudaTransition.urgent = transition.isUrgent();
    cudaTransition.controllable = transition.isControllable();
    cudaTransition._position = transition.getPosition();
    cudaTransition.distribution = convertDistribution(transition.getDistribution());
    cudaTransition._weight = transition.getWeight();
    cudaTransition._firingMode = convertFiringMode(transition.getFiringMode());
    return cudaTransition;
  }

  static CudaTimedInputArc
  convertInputArc(const TAPN::TimedInputArc &arc,
                  const std::unordered_map<const TAPN::TimedPlace *, CudaTimedPlace *> &placeMap,
                  const std::unordered_map<const TAPN::TimedTransition *, CudaTimedTransition *> &transitionMap) {
    return {convertTimeInterval(arc.getInterval()), placeMap.at(&arc.getInputPlace()),
            transitionMap.at(&arc.getOutputTransition()), arc.getWeight()};
  }

  static CudaTimedOutputArc
  convertOutputArc(const TAPN::OutputArc &arc,
                   const std::unordered_map<const TAPN::TimedPlace *, CudaTimedPlace *> &placeMap,
                   const std::unordered_map<const TAPN::TimedTransition *, CudaTimedTransition *> &transitionMap) {
    return {transitionMap.at(&arc.getInputTransition()), placeMap.at(&arc.getOutputPlace()), arc.getWeight()};
  }

  static CudaTimedTransportArc
  convertTransportArc(const TAPN::TransportArc &arc,
                      const std::unordered_map<const TAPN::TimedPlace *, CudaTimedPlace *> &placeMap,
                      const std::unordered_map<const TAPN::TimedTransition *, CudaTimedTransition *> &transitionMap) {
    return {convertTimeInterval(arc.getInterval()), placeMap.at(&arc.getSource()),
            transitionMap.at(&arc.getTransition()), placeMap.at(&arc.getDestination()),
            static_cast<int>(arc.getWeight())};
  }

  static CudaTimedInhibitorArc
  convertInhibitorArc(const TAPN::InhibitorArc &arc,
                      const std::unordered_map<const TAPN::TimedPlace *, CudaTimedPlace *> &placeMap,
                      const std::unordered_map<const TAPN::TimedTransition *, CudaTimedTransition *> &transitionMap) {
    return {placeMap.at(&arc.getInputPlace()), transitionMap.at(&arc.getOutputTransition()), arc.getWeight()};
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

  static CudaTimeInterval convertTimeInterval(const TAPN::TimeInterval &interval) {
    return {interval.isLowerBoundStrict(), interval.getLowerBound(), interval.getUpperBound(),
            interval.isUpperBoundStrict()};
  }

  static Atler::SimpleTimeInvariant convertTimeInvariant(const TAPN::TimeInvariant &invariant) {
    // print the invariant
    return {invariant.isBoundStrict(), invariant.getBound()};
  }

  static CudaSMC::Distribution convertDistribution(const SMC::Distribution &distribution) {
    CudaSMC::Distribution cudaDistribution;
    cudaDistribution.type = static_cast<CudaSMC::DistributionType>(distribution.type);
    cudaDistribution.parameters = *reinterpret_cast<const CudaSMC::DistributionParameters *>(&distribution.parameters);
    return cudaDistribution;
  }

  static CudaSMC::FiringMode convertFiringMode(const SMC::FiringMode &mode) {
    return static_cast<CudaSMC::FiringMode>(mode);
  }
};

} // namespace Cuda
} // namespace VerifyTAPN

#endif
