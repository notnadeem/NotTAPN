#ifndef ATLERPROBABILITYESTIMATION_HPP
#define ATLERPROBABILITYESTIMATION_HPP

// #include "../../Core/TAPN/TimedArcPetriNet.hpp"
// #include "../../Core/Query/SMCQuery.hpp"
// #include "../VerificationTypes/Verification.hpp"

#include "Core/Query/SMCQuery.hpp"
#include "DiscreteVerification/DataStructures/RealMarking.hpp"
#include "DiscreteVerification/VerificationTypes/Verification.hpp"

namespace VerifyTAPN::DiscreteVerification {

class AtlerProbabilityEstimation : public Verification<RealMarking> {

    public:
      AtlerProbabilityEstimation(TAPN::TimedArcPetriNet &tapn,
                                 RealMarking &initialMarking, AST::SMCQuery *query,
                                 VerificationOptions options);

      AtlerProbabilityEstimation(TAPN::TimedArcPetriNet &tapn,
                                 RealMarking &initialMarking, AST::SMCQuery *query,
                                 VerificationOptions options, unsigned int runs)
          : Verification(tapn, initialMarking, query, options), numberOfRuns(0),
            maxTokensSeen(0), smcSettings(query->getSmcSettings()), validRuns(0),
            runsNeeded(runs) {}

      // AtlerProbabilityEstimation(TAPN::TimedArcPetriNet &tapn,
      //                            RealMarking &initialMarking, AST::SMCQuery
      //                            *query, VerificationOptions options, unsigned
      //                            int runs)
      //     : Verification(tapn, initialMarking, query, options),
      //       runGenerator(tapn, options.getSMCNumericPrecision()),
      //       numberOfRuns(0), maxTokensSeen(0),
      //       smcSettings(query->getSmcSettings()), validRuns(0), runsNeeded(runs)
      //       {}

      // Main execution methods
      bool run() override;
      bool parallel_run();

      // Core functionality
      // void prepare();
      // bool executeRun(SMCRunGenerator *generator);
      unsigned int maxUsedTokens() override;
      void setMaxTokensIfGreater(unsigned int i);
      bool mustDoAnotherRun();
      bool handleSuccessor(RealMarking *marking) override;
      float getEstimation();
      void computeChernoffHoeffdingBound(const float intervalWidth,
                                         const float confidence);
      // bool reachedRunBound(SMCRunGenerator *generator);
      // void handleRunResult(const bool res, int steps, double delay);

      // Printing and output methods
      void printResult();
      void printStats() override;
      void printTransitionStatistics() const override;
      void printHumanTrace(std::stack<RealMarking *> &stack,
                       const std::string &name);
  void printXMLTrace(std::stack<RealMarking *> &stack, const std::string &name,
                     rapidxml::xml_document<> &doc,
                     rapidxml::xml_node<> *list_node);
  void printValidRunsStats();
  void printViolatingRunsStats();
  void printGlobalRunsStats();
  static void printRunsStats(const std::string category, unsigned long n,
                             unsigned long totalSteps, double totalDelay,
                             std::vector<int> perStep,
                             std::vector<float> perDelay);
  void printCumulativeStats();

  // XML node creation methods
  rapidxml::xml_node<> *createTransitionNode(RealMarking *old,
                                             RealMarking *current,
                                             rapidxml::xml_document<> &doc);
  void createTransitionSubNodes(RealMarking *old, RealMarking *current,
                                rapidxml::xml_document<> &doc,
                                rapidxml::xml_node<> *transitionNode,
                                const TAPN::TimedPlace &place,
                                const TAPN::TimeInterval &interval, int weight);
  rapidxml::xml_node<> *createTokenNode(rapidxml::xml_document<> &doc,
                                        const TAPN::TimedPlace &place,
                                        const RealToken &token);

protected: // 1
  // TODO: Create alternative run generator for this class
  // RandomRunGenerator runGenerator;
  SMCSettings smcSettings;
  size_t numberOfRuns;
  uint maxTokensSeen;
  double totalTime = 0;
  unsigned long totalSteps = 0;
  int64_t durationNs = 0;

  unsigned int runsNeeded;
  unsigned int validRuns;
  double validRunsTime = 0;
  unsigned long validRunsSteps = 0;
  double violatingRunTime = 0;
  unsigned long violatingRunSteps = 0;

  std::vector<int> validPerStep;
  std::vector<float> validPerDelay;
  std::vector<int> violatingPerStep;
  std::vector<float> violatingPerDelay;
  float maxValidDuration = 0.0f;

  // std::mutex run_res_mutex;
  // std::vector<std::stack<RealMarking *>> traces; // NOTE: not used or
  // necessary
};

} // namespace VerifyTAPN::DiscreteVerification

#endif
