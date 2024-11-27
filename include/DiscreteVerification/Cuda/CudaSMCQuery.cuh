#ifndef SIMPLESMCQUERIES_CUH
#define SIMPLESMCQUERIES_CUH

#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda::AST {

struct SimpleSMCSettings {
  int timeBound;
  int stepBound;
  float falsePositives;
  float falseNegatives;
  float indifferenceRegionUp;
  float indifferenceRegionDown;
  float confidence;
  float estimationIntervalWidth;
  bool compareToFloat;
  float geqThan;
};

class SimpleSMCQuery : public Atler::AST::SimpleQuery {
public:
  __host__ __device__ SimpleSMCQuery(Atler::AST::SimpleQuantifier quantifier, SimpleSMCSettings settings,
                 Atler::AST::SimpleExpression *expr)
      : SimpleQuery(quantifier, expr), smcSettings(settings) {};

  __host__ __device__ SimpleSMCQuery(const SimpleSMCQuery &other)
      : SimpleQuery(other.quantifier, other.expr->clone()),
        smcSettings(other.smcSettings) {};

  __host__ __device__ SimpleSMCQuery &operator=(const SimpleSMCQuery &other) {
    if (&other != this) {
      delete expr;
      expr = other.expr->clone();
      smcSettings = other.smcSettings;
    }
    return *this;
  }

  __host__ __device__ virtual SimpleSMCQuery *clone() const override;

  __host__ __device__ void accept(Atler::AST::SimpleVisitor &visitor, Atler::AST::Result &context) override;

  __host__ __device__ void setSMCSettings(SimpleSMCSettings newSettings) {
    smcSettings = newSettings;
  }

  __host__ __device__ SimpleSMCSettings &getSmcSettings() { return smcSettings; }

public:
  Atler::AST::SimpleQuantifier quantifier;
  Atler::AST::SimpleExpression *expr;
  SimpleSMCSettings smcSettings;
};

} // namespace VerifyTAPN::Cuda::AST

#endif
