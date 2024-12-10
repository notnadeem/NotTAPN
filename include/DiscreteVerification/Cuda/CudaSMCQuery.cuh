#ifndef CUDASMCQUERIES_CUH
#define CUDASMCQUERIES_CUH

#include "DiscreteVerification/Cuda/CudaAST.cuh"
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

class CudaSMCQuery : public AST::CudaQuery {
public:
  __host__ __device__ CudaSMCQuery(AST::CudaQuantifier quantifier, CudaSMCSettings settings,
                 AST::CudaExpression *expr)
      : AST::CudaQuery(quantifier, expr), smcSettings(settings) {
        this->expr = expr;
        this->quantifier = quantifier;
        this->smcSettings = settings;
      };

  __host__ __device__ CudaSMCQuery(const CudaSMCQuery &other)
      : AST::CudaQuery(other.quantifier, other.expr->clone()),
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

  __host__ __device__ void accept(AST::CudaVisitor &visitor, AST::Result &context) override;

  __host__ __device__ void setSMCSettings(CudaSMCSettings newSettings) {
    smcSettings = newSettings;
  }

  __host__ __device__ CudaSMCSettings &getSmcSettings() { return smcSettings; }

public:
  AST::CudaQuantifier quantifier;
  AST::CudaExpression *expr;
  CudaSMCSettings smcSettings;
};

} // namespace VerifyTAPN::Cuda::AST

#endif
