#ifndef SIMPLEVERIFICATIONOPTIONS_HPP_
#define SIMPLEVERIFICATIONOPTIONS_HPP_

#include <cstddef>
namespace VerifyTAPN::Atler {

struct SimpleVerificationOptions {
  enum Trace { NO_TRACE, SOME_TRACE, FASTEST_TRACE };

  enum SearchType {
    BREADTHFIRST,
    DEPTHFIRST,
    RANDOM,
    COVERMOST,
    DEFAULT,
    MINDELAYFIRST,
    OverApprox
  };

  enum VerificationType { DISCRETE, TIMEDART };

  enum MemoryOptimization { NO_MEMORY_OPTIMIZATION, PTRIE };

  enum WorkflowMode {
    NOT_WORKFLOW,
    WORKFLOW_SOUNDNESS,
    WORKFLOW_STRONG_SOUNDNESS
  };

  enum SMCTracesType { ANY_TRACE, SATISFYING_TRACES, UNSATISFYING_TRACES };

  const char *inputFile;
  size_t inputFileLength;

  const char *queryFile;
  size_t queryFileLength;

  const SearchType searchType = DEFAULT;
  VerificationType verificationType = DISCRETE;
  MemoryOptimization memOptimization = NO_MEMORY_OPTIMIZATION;
  unsigned int k_bound = 0;
  Trace trace = NO_TRACE;
  bool xml_trace = true;
  bool useGlobalMaxConstants = false;
  bool keepDeadTokens = false;
  bool enableGCDLowerGuards = false;
  bool printBindings = false;
  WorkflowMode workflow = NOT_WORKFLOW;
  long long workflowBound = 0;
  bool calculateCmax = false;
  bool partialOrder{};

  const char *outputFile;
  size_t outputFileLength;

  const char *outputQuery;
  size_t outputQueryLength;

  const char *strategy_output;
  size_t strategy_outputLength;

  bool benchmark = false;
  unsigned int benchmarkRuns = 100;
  bool parallel = false;
  bool printCumulative = false;
  unsigned int cumulativeRoundingDigits = 2;
  unsigned int stepsStatsScale = 2000;
  unsigned int timeStatsScale = 2000;
  unsigned int smcTraces = 0;
  SMCTracesType smcTracesType = ANY_TRACE;
  unsigned int smcNumericPrecision = 5;
};

} // namespace VerifyDTAPN::Atler

#endif
