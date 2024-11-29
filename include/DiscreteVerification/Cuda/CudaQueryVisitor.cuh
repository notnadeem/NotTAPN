#ifndef CUDAQUERYVISITOR_CUH_
#define CUDAQUERYVISITOR_CUH_

#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"
#include "DiscreteVerification/Cuda/CudaTimedArcPetriNet.cuh"
#include "SimpleVisitor.hpp"

#include <cassert>
#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

using namespace Atler::AST;

class CudaQueryVisitor : public SimpleVisitor {
public:
  __host__ __device__ CudaQueryVisitor(CudaRealMarking &marking, const CudaTimedArcPetriNet &tapn, int maxDelay)
      : marking(marking), tapn(tapn), maxDelay(maxDelay) {
    deadlockChecked = false;
    deadlocked = false;
  };

  __host__ __device__ CudaQueryVisitor(CudaRealMarking &marking, const CudaTimedArcPetriNet &tapn)
      : marking(marking), tapn(tapn), maxDelay(0) {
    deadlockChecked = false;
    deadlocked = false;
  }

  __host__ __device__ ~CudaQueryVisitor() override = default;

public: // visitor methods
  __host__ __device__ void visit(NotExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(OrExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(AndExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(AtomicProposition &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(BoolExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(SimpleQuery &query, Atler::AST::Result &context) override;

  __host__ __device__ void visit(DeadlockExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(NumberExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(IdentifierExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(MultiplyExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(MinusExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(SubtractExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(PlusExpression &expr, Atler::AST::Result &context) override;

private:
  __host__ __device__ bool compare(int numberOfTokensInPlace, AtomicProposition::op_e op, int n) const;

private:
  const CudaRealMarking &marking;
  const CudaTimedArcPetriNet &tapn;
  bool deadlockChecked;
  bool deadlocked;
  const int maxDelay;
};

__host__ __device__ void CudaQueryVisitor::visit(NotExpression &expr, Atler::AST::Result &context) {
  BoolResult c;
  expr.getChild().accept(*this, c);
  expr.eval = !c.value;
  static_cast<BoolResult &>(context).value = expr.eval;
}

__host__ __device__ void CudaQueryVisitor::visit(OrExpression &expr, Atler::AST::Result &context) {
  BoolResult left, right;
  expr.getLeft().accept(*this, left);
  // use lazy evaluation
  if (left.value) {
    static_cast<BoolResult &>(context).value = true;
  } else {
    expr.getRight().accept(*this, right);
    static_cast<BoolResult &>(context).value = right.value;
  }
  expr.eval = static_cast<BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AndExpression &expr, Atler::AST::Result &context) {
  BoolResult left, right;
  expr.getLeft().accept(*this, left);

  // use lazy evaluation
  if (!left.value) {
    static_cast<BoolResult &>(context).value = false;
  } else {
    expr.getRight().accept(*this, right);
    static_cast<BoolResult &>(context).value = right.value;
  }
  expr.eval = static_cast<BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AtomicProposition &expr, Atler::AST::Result &context) {
  IntResult left;
  expr.getLeft().accept(*this, left);
  IntResult right;
  expr.getRight().accept(*this, right);

  static_cast<BoolResult &>(context).value = compare(left.value, expr.getOperator(), right.value);
  expr.eval = static_cast<BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(BoolExpression &expr, Atler::AST::Result &context) {
  static_cast<BoolResult &>(context).value = expr.getValue();
  expr.eval = expr.getValue();
}

__host__ __device__ void CudaQueryVisitor::visit(NumberExpression &expr, Atler::AST::Result &context) {
  ((IntResult &)context).value = expr.getValue();
  expr.eval = static_cast<IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(IdentifierExpression &expr, Atler::AST::Result &context) {
  ((IntResult &)context).value = marking.numberOfTokensInPlace(expr.getPlace());
  expr.eval = static_cast<IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(MultiplyExpression &expr, Atler::AST::Result &context) {
  IntResult left;
  expr.getLeft().accept(*this, left);
  IntResult right;
  expr.getRight().accept(*this, right);
  ((IntResult &)context).value = left.value * right.value;
  expr.eval = static_cast<IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(MinusExpression &expr, Atler::AST::Result &context) {
  IntResult value;
  expr.getValue().accept(*this, value);
  ((IntResult &)context).value = -value.value;
  expr.eval = static_cast<IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(SubtractExpression &expr, Atler::AST::Result &context) {
  IntResult left;
  expr.getLeft().accept(*this, left);
  IntResult right;
  expr.getRight().accept(*this, right);
  ((IntResult &)context).value = left.value - right.value;
  expr.eval = static_cast<IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(PlusExpression &expr, Atler::AST::Result &context) {
  IntResult left;
  expr.getLeft().accept(*this, left);
  IntResult right;
  expr.getRight().accept(*this, right);
  ((IntResult &)context).value = left.value + right.value;
  expr.eval = static_cast<IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(SimpleQuery &query, Atler::AST::Result &context) {
  query.getChild()->accept(*this, context);
  if (query.getQuantifier() == AG || query.getQuantifier() == AF || query.getQuantifier() == PG) {
    static_cast<BoolResult &>(context).value = !static_cast<BoolResult &>(context).value;
  }
  query.eval = static_cast<IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(DeadlockExpression &expr, Atler::AST::Result &context) {
  if (!deadlockChecked) {
    deadlockChecked = true;
    deadlocked = marking.canDeadlock(tapn, maxDelay);
  }
  static_cast<BoolResult &>(context).value = deadlocked;
  expr.eval = static_cast<BoolResult &>(context).value;
}

__host__ __device__ bool CudaQueryVisitor::compare(int numberOfTokensInPlace, AtomicProposition::op_e op,
                                                     int n) const {

  switch (op) {
  case AtomicProposition::LT:
    return numberOfTokensInPlace < n;
  case AtomicProposition::LE:
    return numberOfTokensInPlace <= n;
  case AtomicProposition::EQ:
    return numberOfTokensInPlace == n;
  case AtomicProposition::NE:
    return numberOfTokensInPlace != n;
  default:
    assert(false);
  }
  return false;
}

} // namespace Cuda
} // namespace VerifyTAPN
#endif /* CUDAQUERYVISITOR_CUH_ */
