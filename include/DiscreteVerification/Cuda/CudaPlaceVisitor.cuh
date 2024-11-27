#ifndef SIMPLEPLACEVISITOR_CUH_
#define SIMPLEPLACEVISITOR_CUH_

#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Atler/SimpleVisitor.hpp"
#include <exception>
#include <vector>
#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {

using namespace Atler::AST;

class SimplePlaceVisitor : public SimpleVisitor {
public:
  __host__ __device__ SimplePlaceVisitor() = default;

  ~SimplePlaceVisitor() override = default;

public: // visitor methods
  __host__ __device__ void visit(NotExpression &expr, Result &context) override;

  __host__ __device__ void visit(OrExpression &expr, Result &context) override;

  __host__ __device__ void visit(AndExpression &expr, Result &context) override;

  __host__ __device__ void visit(AtomicProposition &expr, Result &context) override;

  __host__ __device__ void visit(BoolExpression &expr, Result &context) override;

  __host__ __device__ void visit(SimpleQuery &query, Result &context) override;

  __host__ __device__ void visit(DeadlockExpression &expr, Result &context) override;

  __host__ __device__ void visit(NumberExpression &expr, Result &context) override;

  __host__ __device__ void visit(IdentifierExpression &expr, Result &context) override;

  __host__ __device__ void visit(MinusExpression &expr, Result &context) override;

  __host__ __device__ virtual void visit(OperationExpression &expr, Result &context);

  __host__ __device__ void visit(MultiplyExpression &expr, Result &context) override {
    visit((OperationExpression &)expr, context);
  };

  __host__ __device__ void visit(SubtractExpression &expr, Result &context) override {
    visit((OperationExpression &)expr, context);
  };

  __host__ __device__ void visit(PlusExpression &expr, Result &context) override {
    visit((OperationExpression &)expr, context);
  };
};

} // namespace VerifyTAPN::Atler
#endif /* PLACEVISITOR_CUH_ */
