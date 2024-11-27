#ifndef CUDASMCQUERIES_CUH
#define CUDASMCQUERIES_CUH

#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda::AST {

struct CudaSMCSettings {
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

class CudaSMCQuery : public Atler::AST::SimpleQuery {
public:
  __host__ __device__ CudaSMCQuery(Atler::AST::SimpleQuantifier quantifier, CudaSMCSettings settings,
                 Atler::AST::SimpleExpression *expr)
      : SimpleQuery(quantifier, expr), smcSettings(settings) {};

  __host__ __device__ CudaSMCQuery(const CudaSMCQuery &other)
      : SimpleQuery(other.quantifier, other.expr->clone()),
        smcSettings(other.smcSettings) {};

  __host__ __device__ CudaSMCQuery &operator=(const CudaSMCQuery &other) {
    if (&other != this) {
      delete expr;
      expr = other.expr->clone();
      smcSettings = other.smcSettings;
    }
    return *this;
  }

  __host__ __device__ virtual CudaSMCQuery *clone() const override;

  __host__ __device__ void accept(Atler::AST::SimpleVisitor &visitor, Atler::AST::Result &context) override;

  __host__ __device__ void setSMCSettings(CudaSMCSettings newSettings) {
    smcSettings = newSettings;
  }

  __host__ __device__ CudaSMCSettings &getSmcSettings() { return smcSettings; }

public:
  Atler::AST::SimpleQuantifier quantifier;
  Atler::AST::SimpleExpression *expr;
  CudaSMCSettings smcSettings;
};

} // namespace VerifyTAPN::Cuda::AST

#endif
