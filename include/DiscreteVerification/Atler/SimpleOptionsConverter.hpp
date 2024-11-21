#ifndef VERIFYYAPN_ATLER_SIMPLEOPTIONS_CONVERTER_HPP_
#define VERIFYYAPN_ATLER_SIMPLEOPTIONS_CONVERTER_HPP_

#include "Core/VerificationOptions.hpp"
#include "DiscreteVerification/Atler/SimpleVerificationOptions.hpp"

namespace VerifyTAPN::Atler {

class SimpleOptionsConverter {
public:
  static Atler::SimpleVerificationOptions
  convert(VerifyTAPN::VerificationOptions &options) {
    Atler::SimpleVerificationOptions simpleOptions{
        .inputFile = options.getInputFile().c_str(),
        .inputFileLength = options.getInputFile().size(),
        .queryFile = options.getQueryFile().c_str(),
        .queryFileLength = options.getQueryFile().size(),
        .searchType = convertSearchType(options.getSearchType()),
        .verificationType =
            convertVerificationType(options.getVerificationType()),
        .memOptimization =
            convertMemoryOptimization(options.getMemoryOptimization()),
        .k_bound = options.getKBound(),
        .trace = convertTrace(options.getTrace()),
        .xml_trace = options.getXmlTrace(),
        .useGlobalMaxConstants = options.getGlobalMaxConstantsEnabled(),
        .keepDeadTokens = options.getKeepDeadTokens(),
        .enableGCDLowerGuards = options.getGCDLowerGuardsEnabled(),
        .printBindings = options.getPrintBindings(),
        .workflow = convertWorkflowMode(options.getWorkflowMode()),
        .workflowBound = options.getWorkflowBound(),
        .calculateCmax = options.getCalculateCmax(),
        .partialOrder = options.getPartialOrderReduction(),
        .outputFile = options.getOutputModelFile().c_str(),
        .outputFileLength = options.getOutputModelFile().size(),
        .outputQuery = options.getOutputModelFile().c_str(),
        .outputQueryLength = options.getOutputModelFile().size(),
        .strategy_output = options.getStrategyFile().c_str(),
        .strategy_outputLength = options.getStrategyFile().size(),
        .benchmark = options.isBenchmarkMode(),
        .benchmarkRuns = options.getBenchmarkRuns(),
        .parallel = options.isParallel(),
        .printCumulative = options.mustPrintCumulative(),
        .cumulativeRoundingDigits = options.getCumulativeRoundingDigits(),
        .stepsStatsScale = options.getStepsStatsScale(),
        .timeStatsScale = options.getTimeStatsScale(),
        .smcTraces = options.getSmcTraces(),
        .smcTracesType = convertSMCTracesType(options.getSMCTracesType()),
        .smcNumericPrecision = options.getSMCNumericPrecision()};

    return simpleOptions;
  }

private:
  static Atler::SimpleVerificationOptions::SearchType
  convertSearchType(VerifyTAPN::VerificationOptions::SearchType searchType) {
    switch (searchType) {
    case VerifyTAPN::VerificationOptions::BREADTHFIRST:
      return Atler::SimpleVerificationOptions::SearchType::BREADTHFIRST;
    case VerifyTAPN::VerificationOptions::DEPTHFIRST:
      return Atler::SimpleVerificationOptions::SearchType::DEPTHFIRST;
    case VerifyTAPN::VerificationOptions::RANDOM:
      return Atler::SimpleVerificationOptions::SearchType::RANDOM;
    case VerifyTAPN::VerificationOptions::COVERMOST:
      return Atler::SimpleVerificationOptions::SearchType::COVERMOST;
    case VerifyTAPN::VerificationOptions::DEFAULT:
      return Atler::SimpleVerificationOptions::SearchType::DEFAULT;
    case VerifyTAPN::VerificationOptions::MINDELAYFIRST:
      return Atler::SimpleVerificationOptions::SearchType::MINDELAYFIRST;
    case VerifyTAPN::VerificationOptions::OverApprox:
      return Atler::SimpleVerificationOptions::SearchType::OverApprox;
    }
  }

  static Atler::SimpleVerificationOptions::VerificationType
  convertVerificationType(
      VerifyTAPN::VerificationOptions::VerificationType verificationType) {
    switch (verificationType) {
    case VerifyTAPN::VerificationOptions::DISCRETE:
      return Atler::SimpleVerificationOptions::VerificationType::DISCRETE;
    case VerifyTAPN::VerificationOptions::TIMEDART:
      return Atler::SimpleVerificationOptions::VerificationType::TIMEDART;
    }
  }

  static Atler::SimpleVerificationOptions::MemoryOptimization
  convertMemoryOptimization(
      VerifyTAPN::VerificationOptions::MemoryOptimization memOptimization) {
    switch (memOptimization) {
    case VerifyTAPN::VerificationOptions::NO_MEMORY_OPTIMIZATION:
      return Atler::SimpleVerificationOptions::MemoryOptimization::
          NO_MEMORY_OPTIMIZATION;
    case VerifyTAPN::VerificationOptions::PTRIE:
      return Atler::SimpleVerificationOptions::MemoryOptimization::PTRIE;
    }
  }

  static Atler::SimpleVerificationOptions::Trace
  convertTrace(VerifyTAPN::VerificationOptions::Trace trace) {
    switch (trace) {
    case VerifyTAPN::VerificationOptions::NO_TRACE:
      return Atler::SimpleVerificationOptions::Trace::NO_TRACE;
    case VerifyTAPN::VerificationOptions::SOME_TRACE:
      return Atler::SimpleVerificationOptions::Trace::SOME_TRACE;
    case VerifyTAPN::VerificationOptions::FASTEST_TRACE:
      return Atler::SimpleVerificationOptions::Trace::FASTEST_TRACE;
    }
  }

  static Atler::SimpleVerificationOptions::WorkflowMode
  convertWorkflowMode(VerifyTAPN::VerificationOptions::WorkflowMode workflow) {
    switch (workflow) {
    case VerifyTAPN::VerificationOptions::NOT_WORKFLOW:
      return Atler::SimpleVerificationOptions::WorkflowMode::NOT_WORKFLOW;
    case VerifyTAPN::VerificationOptions::WORKFLOW_SOUNDNESS:
      return Atler::SimpleVerificationOptions::WorkflowMode::WORKFLOW_SOUNDNESS;
    case VerifyTAPN::VerificationOptions::WORKFLOW_STRONG_SOUNDNESS:
      return Atler::SimpleVerificationOptions::WorkflowMode::
          WORKFLOW_STRONG_SOUNDNESS;
    }
  }

  static Atler::SimpleVerificationOptions::SMCTracesType
  convertSMCTracesType(VerifyTAPN::VerificationOptions::SMCTracesType toSave) {
    switch (toSave) {
    case VerifyTAPN::VerificationOptions::ANY_TRACE:
      return Atler::SimpleVerificationOptions::SMCTracesType::ANY_TRACE;
    case VerifyTAPN::VerificationOptions::SATISFYING_TRACES:
      return Atler::SimpleVerificationOptions::SMCTracesType::SATISFYING_TRACES;
    case VerifyTAPN::VerificationOptions::UNSATISFYING_TRACES:
      return Atler::SimpleVerificationOptions::SMCTracesType::UNSATISFYING_TRACES;
    }
  }

};

} // namespace VerifyTAPN::Atler

#endif
