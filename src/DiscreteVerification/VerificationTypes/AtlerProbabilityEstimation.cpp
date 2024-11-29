#include "DiscreteVerification/VerificationTypes/AtlerProbabilityEstimation.hpp"
#include "DiscreteVerification/Atler/AtlerRunResult.hpp"
#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Atler/SimpleDynamicArray.hpp"
#include "DiscreteVerification/Atler/SimpleInterval.hpp"
#include "DiscreteVerification/Atler/SimpleOptionsConverter.hpp"
#include "DiscreteVerification/Atler/SimpleQueryVisitor.hpp"
#include "DiscreteVerification/Atler/SimpleRealMarking.hpp"
#include "DiscreteVerification/Atler/SimpleSMCQuery.hpp"
#include "DiscreteVerification/Atler/SimpleSMCQueryConverter.hpp"
#include "DiscreteVerification/Atler/SimpleTAPNConverter.hpp"
#include "DiscreteVerification/Atler/SimpleTimedArcPetriNet.hpp"
#include "DiscreteVerification/Atler/SimpleVerificationOptions.hpp"
#include <iomanip>
#include <vector>

std::string printDoubleo(double value, unsigned int precision) {
  std::ostringstream oss;
  if (precision == 0)
    precision = std::numeric_limits<double>::max_digits10;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

namespace VerifyTAPN::DiscreteVerification {

AtlerProbabilityEstimation::AtlerProbabilityEstimation(
    TAPN::TimedArcPetriNet &tapn, RealMarking &initialMarking,
    AST::SMCQuery *query, VerificationOptions options)
    : tapn(tapn), initialMarking(initialMarking), query(query),
      options(options), numberOfRuns(0), maxTokensSeen(0),
      smcSettings(query->getSmcSettings()), validRuns(0), runsNeeded(0) {
  computeChernoffHoeffdingBound(smcSettings.estimationIntervalWidth,
                                smcSettings.confidence);
}

bool AtlerProbabilityEstimation::run() {
  std::cout << "Converting TAPN and marking..." << std::endl;
  auto result = Atler::SimpleTAPNConverter::convert(tapn, initialMarking);
  Atler::SimpleTimedArcPetriNet stapn = result->first;
  Atler::SimpleRealMarking siMarking = result->second;

  std::cout << "Converting Query..." << std::endl;
  SMCQuery *currentSMCQuery = static_cast<SMCQuery *>(query);
  Atler::AST::SimpleSMCQuery *simpleSMCQuery =
      Atler::AST::SimpleSMCQueryConverter::convert(currentSMCQuery);

  std::cout << "Converting Options..." << std::endl;
  Atler::SimpleVerificationOptions simpleOptions =
      Atler::SimpleOptionsConverter::convert(options);

  // TODO: Convert the PlaceVisitor to a simple representation
  // NOTE: Also find a way to simplify the representation of the PlaceVisitor

  // Simulate prepare func
  // setup the run generator

  std::cout << "Creating run generator..." << std::endl;
  auto runres = Atler::AtlerRunResult(stapn);
  std::cout << "Run prepare " << std::endl;
  runres.prepare(siMarking);

  Atler::SimpleDynamicArray<Atler::AtlerRunResult *> clones(runsNeeded);
  for (int i = 0; i < runsNeeded; i++) {
    clones.add(new Atler::AtlerRunResult(runres));
  }

  for (int i = 0; i < 1; i++) {
    Atler::AtlerRunResult *runner = clones.get(i);
    bool runRes = false;
    Atler::SimpleRealMarking *newMarking = runner->parent;
    int count = 0;
    while (!runner->maximal && !(runner->totalTime >= smcSettings.timeBound ||
                                 runner->totalSteps >= smcSettings.stepBound)) {
      Atler::SimpleQueryVisitor checker(*newMarking, stapn);
      Atler::AST::BoolResult result;

      simpleSMCQuery->accept(checker, result);
      runRes =  result.value;
      if (runRes) {
          break;
      }
      newMarking = runner->next();

      std::cout << "Checking: " << count << "/" << runsNeeded << std::endl;
      std::cout << "Time bound: " << smcSettings.timeBound << std::endl;
      std::cout << "Steps bound: " << smcSettings.stepBound << std::endl;
      std::cout << "Total time: " << runner->totalTime << std::endl;
      std::cout << "Total steps: " << runner->totalSteps << std::endl;
      count++;
    }
  }

  // Create clones of the run generator

  // print all the transition intervals
  // for (size_t i = 0; i < runres.transitionIntervals.size; i++) {
  //   std::cout << "Transition " << i << ": ";
  //   for (size_t j = 0; j < runres.transitionIntervals.get(i).size; j++) {
  //     std::cout << "(" << runres.transitionIntervals.get(i).get(j).lower()
  //               << ", " << runres.transitionIntervals.get(i).get(j).upper()
  //               << ") ";
  //   }
  //   std::cout << std::endl;
  // }

  // End prepare
  // std::cout << "Weight: " << stapn.places[0].inputArcs[0].weight <<
  // std::endl; std::cout << "Number of places in simple tapn: " <<
  // stapn.placesLength
  //           << std::endl;
  // std::cout << stapn.maxConstant << std::endl;
  // std::cout << "Magic number: " << simpleSMCQuery->quantifier << std::endl;
  // std::cout << "input length: " << simpleOptions.inputFile << std::endl;
  return false;
}

bool AtlerProbabilityEstimation::parallel_run() { return false; }

// void AtlerProbabilityEstimation::prepare() { return; }
// bool AtlerProbabilityEstimation::executeRun(SMCRunGenerator *generator) {
// }

unsigned int AtlerProbabilityEstimation::maxUsedTokens() {
  return maxTokensSeen;
}

void AtlerProbabilityEstimation::setMaxTokensIfGreater(unsigned int i) {
  if (i > maxTokensSeen) {
    maxTokensSeen = i;
  }
}

// NOTE: This should not be necessary in the new implementation
bool AtlerProbabilityEstimation::mustDoAnotherRun() {
  return numberOfRuns < runsNeeded;
}

// bool AtlerProbabilityEstimation::handleSuccessor(
//     Atler::SimpleRealMarking *marking) {
//         return false;
// }

float AtlerProbabilityEstimation::getEstimation() {
  float proba = ((float)validRuns) / numberOfRuns;
  return (query->getQuantifier() == PG) ? 1 - proba : proba;
}

void AtlerProbabilityEstimation::computeChernoffHoeffdingBound(
    const float intervalWidth, const float confidence) {
  // https://link.springer.com/content/pdf/10.1007/b94790.pdf p.78-79
  float bound = log(2.0 / (1 - confidence)) / (2.0 * pow(intervalWidth, 2));
  runsNeeded = (unsigned int)ceil(bound);
}

void AtlerProbabilityEstimation::printResult() {
  /*if (options.getXmlTrace()) {
      printXMLTrace(m, printStack, query, tapn);
  } else {
      printHumanTrace(m, printStack, query->getQuantifier());
  }*/
  float result = getEstimation();
  float width = smcSettings.estimationIntervalWidth;
  std::cout << "Probability estimation:" << std::endl;
  std::cout << "\tConfidence: " << smcSettings.confidence * 100 << "%"
            << std::endl;
  std::cout << "\tP = " << result << " ± " << width << std::endl;
}

void AtlerProbabilityEstimation::printStats() {
  std::cout << "  runs executed:\t" << numberOfRuns << std::endl;
  std::cout << "  average run length:\t" << (totalSteps / (double)numberOfRuns)
            << std::endl;
  std::cout << "  average run duration:\t" << (totalTime / (double)numberOfRuns)
            << std::endl;
  std::cout << "  verification time:\t" << ((double)durationNs / 1.0E9) << "s"
            << std::endl;
  printGlobalRunsStats();
  printValidRunsStats();
  printViolatingRunsStats();
  if (options.mustPrintCumulative())
    printCumulativeStats();
}

void AtlerProbabilityEstimation::printTransitionStatistics() const {
  // runGenerator.printTransitionStatistics(std::cout);
}

void AtlerProbabilityEstimation::printHumanTrace(
    std::stack<RealMarking *> &stack, const std::string &name) {
  bool isFirst = true;
  std::cout << "Name: " << name << std::endl;
  while (!stack.empty()) {
    if (isFirst) {
      isFirst = false;
    } else {
      RealMarking *marking = stack.top();
      if (marking->getPreviousDelay() > 0) {
        std::cout << "\tDelay: " << marking->getPreviousDelay() << std::endl;
      }
      if (marking->getGeneratedBy() != nullptr) {
        std::cout << "\tTransition:" << marking->getGeneratedBy()->getName()
                  << std::endl;
      }
      if (marking->canDeadlock(tapn, 0)) {
        std::cout << "\tDeadlock: " << std::endl;
      }
    }
    std::cout << "Marking: ";
    for (auto &token_list : stack.top()->getPlaceList()) {
      for (auto &token : token_list.tokens) {
        for (int i = 0; i < token.getCount(); i++) {
          std::cout << "(" << token_list.place->getName() << ","
                    << token.getAge() << ") ";
        }
      }
    }
    stack.pop();
  }
}

void AtlerProbabilityEstimation::printXMLTrace(
    std::stack<RealMarking *> &stack, const std::string &name,
    rapidxml::xml_document<> &doc, rapidxml::xml_node<> *list_node) {
  using namespace rapidxml;
  bool isFirst = true;
  RealMarking *old = nullptr;
  xml_node<> *root = doc.allocate_node(node_element, "trace");
  xml_attribute<> *name_attr =
      doc.allocate_attribute("name", doc.allocate_string(name.c_str()));
  root->append_attribute(name_attr);
  list_node->append_node(root);
  while (!stack.empty()) {
    if (isFirst) {
      isFirst = false;
    } else {
      RealMarking *marking = stack.top();
      if (marking->getPreviousDelay() > 0) {
        std::string str = printDoubleo(marking->getPreviousDelay(),
                                       options.getSMCNumericPrecision());
        xml_node<> *node = doc.allocate_node(node_element, "delay",
                                             doc.allocate_string(str.c_str()));
        root->append_node(node);
      }
      if (marking->getGeneratedBy() != nullptr) {
        root->append_node(createTransitionNode(old, marking, doc));
      }
      if (marking->canDeadlock(tapn, 0)) {
        root->append_node(doc.allocate_node(node_element, "deadlock"));
      }
    }
    old = stack.top();
    stack.pop();
  }
}

void AtlerProbabilityEstimation::printValidRunsStats() {
  std::string category = "valid";
  if (query->getQuantifier() == PF) {
    printRunsStats(category, validRuns, validRunsSteps, validRunsTime,
                   validPerStep, validPerDelay);
  } else {
    printRunsStats(category, numberOfRuns - validRuns, violatingRunSteps,
                   violatingRunTime, violatingPerStep, violatingPerDelay);
  }
}

void AtlerProbabilityEstimation::printViolatingRunsStats() {
  std::string category = "violating";
  if (query->getQuantifier() == PG) {
    printRunsStats(category, validRuns, validRunsSteps, validRunsTime,
                   validPerStep, validPerDelay);
  } else {
    printRunsStats(category, numberOfRuns - validRuns, violatingRunSteps,
                   violatingRunTime, violatingPerStep, violatingPerDelay);
  }
}

void AtlerProbabilityEstimation::printGlobalRunsStats() {
  double stepsMean = (totalSteps / (double)numberOfRuns);
  double timeMean = (totalTime / numberOfRuns);
  double stepsAcc = 0;
  double delayAcc = 0;
  for (int i = 0; i < validPerStep.size(); i++) {
    stepsAcc += pow(i - stepsMean, 2.0) * validPerStep[i];
  }
  for (int i = 0; i < violatingPerStep.size(); i++) {
    stepsAcc += pow(i - stepsMean, 2.0) * violatingPerStep[i];
  }
  for (int i = 0; i < validPerDelay.size(); i++) {
    delayAcc += pow(validPerDelay[i] - timeMean, 2.0);
  }
  for (int i = 0; i < violatingPerDelay.size(); i++) {
    delayAcc += pow(violatingPerDelay[i] - timeMean, 2.0);
  }
  double stepsStdDev = sqrt(stepsAcc / numberOfRuns);
  double delayStdDev = sqrt(delayAcc / numberOfRuns);
  std::cout << "  run duration (std. dev.):\t" << delayStdDev << std::endl;
  std::cout << "  run length (std. dev.):\t" << stepsStdDev << std::endl;
}

void AtlerProbabilityEstimation::printCumulativeStats() {
  unsigned int digits = options.getCumulativeRoundingDigits();
  unsigned int stepScale = options.getStepsStatsScale();
  unsigned int timeScale = options.getTimeStatsScale();
  double mult = pow(10.0f, digits);

  double fact = (query->getQuantifier() == PF) ? 1 : -1;
  double initial = (query->getQuantifier() == PF) ? 0 : 1;

  std::cout << "  cumulative probability / step :" << std::endl;
  double acc = initial;
  double binSize = stepScale == 0 ? 1 : validPerStep.size() / (double)stepScale;
  double bin = 0;
  double lastAcc = acc;
  std::cout << 0 << ":" << acc << ";";
  for (int i = 0; i < validPerStep.size(); i++) {
    double toPrint = round(acc * mult) / mult;
    if (i >= bin + binSize) {
      if (toPrint != lastAcc) {
        std::cout << bin << ":" << lastAcc << ";";
        std::cout << bin << ":" << toPrint << ";";
        lastAcc = toPrint;
      }
      bin = round(i / binSize) * binSize;
    }
    acc += fact * (validPerStep[i] / (double)numberOfRuns);
  }
  if (!validPerStep.empty()) {
    std::cout << (validPerStep.size() - 1) << ":" << lastAcc << ";";
    std::cout << (validPerStep.size() - 1) << ":" << getEstimation() << ";";
  }
  std::cout << std::endl;

  std::cout << "  cumulative probability / delay :" << std::endl;
  acc = initial;
  binSize = timeScale == 0 ? 1 : (maxValidDuration / (double)timeScale);
  std::vector<double> bins(
      binSize > 0 ? (size_t)round(maxValidDuration / binSize) : 1, 0.0f);
  lastAcc = acc;
  for (int i = 0; i < validPerDelay.size(); i++) {
    double delay = validPerDelay[i];
    int binIndex = std::min((size_t)round(delay / binSize), bins.size() - 1);
    bins[binIndex] += 1;
  }
  std::cout << 0 << ":" << acc << ";";
  for (int i = 0; i < bins.size(); i++) {
    acc += fact * (bins[i] / (double)numberOfRuns);
    double toPrint = round(acc * mult) / mult;
    if (toPrint != lastAcc) {
      double binIndex = (i)*binSize;
      std::cout << binIndex << ":" << lastAcc << ";";
      std::cout << binIndex << ":" << toPrint << ";";
      lastAcc = toPrint;
    }
  }
  if (validRuns > 0) {
    std::cout << maxValidDuration << ":" << lastAcc << ";";
    std::cout << maxValidDuration << ":" << getEstimation() << ";";
  }
  std::cout << std::endl;
}

void AtlerProbabilityEstimation::printRunsStats(
    const std::string category, unsigned long n, unsigned long totalSteps,
    double totalDelay, std::vector<int> perStep, std::vector<float> perDelay) {
  if (n == 0) {
    std::cout << "  no " + category +
                     " runs, unable to compute specific statistics"
              << std::endl;
    return;
  }
  double stepsMean = (totalSteps / (double)n);
  double timeMean = (totalDelay / n);
  double stepsAcc = 0;
  double delayAcc = 0;
  for (int i = 0; i < perStep.size(); i++) {
    stepsAcc += pow(i - stepsMean, 2.0) * perStep[i];
  }
  for (int i = 0; i < perDelay.size(); i++) {
    delayAcc += pow(perDelay[i] - timeMean, 2.0);
  }
  double stepsStdDev = sqrt(stepsAcc / n);
  double delayStdDev = sqrt(delayAcc / n);
  std::cout << "  statistics of " + category + " runs:" << std::endl;
  std::cout << "    number of " << category << " runs: " << n << std::endl;
  std::cout << "    duration of a " + category + " run (average):\t" << timeMean
            << std::endl;
  std::cout << "    duration of a " + category + " run (std. dev.):\t"
            << delayStdDev << std::endl;
  std::cout << "    length of a " + category + " run (average):\t" << stepsMean
            << std::endl;
  std::cout << "    length of a " + category + " run (std. dev.):\t"
            << stepsStdDev << std::endl;
}

// XML node creation methods

rapidxml::xml_node<> *AtlerProbabilityEstimation::createTransitionNode(
    RealMarking *old, RealMarking *current, rapidxml::xml_document<> &doc) {
  using namespace rapidxml;
  xml_node<> *transitionNode = doc.allocate_node(node_element, "transition");
  xml_attribute<> *id =
      doc.allocate_attribute("id", current->getGeneratedBy()->getId().c_str());
  transitionNode->append_attribute(id);

  for (auto *arc : current->getGeneratedBy()->getPreset()) {
    createTransitionSubNodes(old, current, doc, transitionNode,
                             arc->getInputPlace(), arc->getInterval(),
                             arc->getWeight());
  }

  for (auto *arc : current->getGeneratedBy()->getTransportArcs()) {
    createTransitionSubNodes(old, current, doc, transitionNode,
                             arc->getSource(), arc->getInterval(),
                             arc->getWeight());
  }

  return transitionNode;
}

void AtlerProbabilityEstimation::createTransitionSubNodes(
    RealMarking *old, RealMarking *current, rapidxml::xml_document<> &doc,
    rapidxml::xml_node<> *transitionNode, const TAPN::TimedPlace &place,
    const TAPN::TimeInterval &interval, int weight) {
  RealTokenList current_tokens = current->getTokenList(place.getIndex());
  RealTokenList old_tokens = old->getTokenList(place.getIndex());
  int tokensFound = 0;
  RealTokenList::const_iterator n_iter = current_tokens.begin();
  RealTokenList::const_iterator o_iter = old_tokens.begin();
  while (n_iter != current_tokens.end() && o_iter != old_tokens.end()) {
    if (n_iter->getAge() == o_iter->getAge()) {
      for (int i = 0; i < o_iter->getCount() - n_iter->getCount(); i++) {
        transitionNode->append_node(createTokenNode(doc, place, *n_iter));
        tokensFound++;
      }
      n_iter++;
      o_iter++;
    } else {
      if (n_iter->getAge() > o_iter->getAge()) {
        transitionNode->append_node(createTokenNode(doc, place, *o_iter));
        tokensFound++;
        o_iter++;
      } else {
        n_iter++;
      }
    }
  }
  for (RealTokenList::const_iterator iter = n_iter;
       iter != current_tokens.end(); iter++) {
    for (int i = 0; i < iter->getCount(); i++) {
      transitionNode->append_node(createTokenNode(doc, place, *iter));
      tokensFound++;
    }
  }
  for (auto &token : old_tokens) {
    if (tokensFound >= weight)
      break;
    if (token.getAge() >= interval.getLowerBound()) {
      for (int i = 0; i < token.getCount() && tokensFound < weight; i++) {
        transitionNode->append_node(createTokenNode(doc, place, token));
        tokensFound++;
      }
    }
  }
}

rapidxml::xml_node<> *
AtlerProbabilityEstimation::createTokenNode(rapidxml::xml_document<> &doc,
                                            const TAPN::TimedPlace &place,
                                            const RealToken &token) {
  using namespace rapidxml;
  xml_node<> *tokenNode = doc.allocate_node(node_element, "token");
  xml_attribute<> *placeAttribute = doc.allocate_attribute(
      "place", doc.allocate_string(place.getName().c_str()));
  tokenNode->append_attribute(placeAttribute);
  auto str = printDoubleo(token.getAge(), options.getSMCNumericPrecision());
  xml_attribute<> *ageAttribute =
      doc.allocate_attribute("age", doc.allocate_string(str.c_str()));
  tokenNode->append_attribute(ageAttribute);
  return tokenNode;
}
} // namespace VerifyTAPN::DiscreteVerification
