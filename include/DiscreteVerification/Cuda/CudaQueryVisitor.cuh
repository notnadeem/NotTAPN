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
  __host__ __device__ void visit(Atler::AST::NotExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::OrExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::AndExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::AtomicProposition &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::BoolExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(SimpleQuery &query, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::DeadlockExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::NumberExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::IdentifierExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::MultiplyExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::MinusExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::SubtractExpression &expr, Atler::AST::Result &context) override;

  __host__ __device__ void visit(Atler::AST::PlusExpression &expr, Atler::AST::Result &context) override;

private:
  __host__ __device__ bool compare(int numberOfTokensInPlace, Atler::AST::AtomicProposition::op_e op, int n) const;

private:
  const CudaRealMarking &marking;
  const CudaTimedArcPetriNet &tapn;
  bool deadlockChecked;
  bool deadlocked;
  const int maxDelay;
};

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::NotExpression &expr, Atler::AST::Result &context) {
  Atler::AST::BoolResult c;
  expr.getChild().accept(*this, c);
  expr.eval = !c.value;
  static_cast<Atler::AST::BoolResult &>(context).value = expr.eval;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::OrExpression &expr, Atler::AST::Result &context) {
  Atler::AST::BoolResult left, right;
  expr.getLeft().accept(*this, left);
  // use lazy evaluation
  if (left.value) {
    static_cast<Atler::AST::BoolResult &>(context).value = true;
  } else {
    expr.getRight().accept(*this, right);
    static_cast<Atler::AST::BoolResult &>(context).value = right.value;
  }
  expr.eval = static_cast<Atler::AST::BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::AndExpression &expr, Atler::AST::Result &context) {
  Atler::AST::BoolResult left, right;
  expr.getLeft().accept(*this, left);

  // use lazy evaluation
  if (!left.value) {
    static_cast<Atler::AST::BoolResult &>(context).value = false;
  } else {
    expr.getRight().accept(*this, right);
    static_cast<Atler::AST::BoolResult &>(context).value = right.value;
  }
  expr.eval = static_cast<Atler::AST::BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::AtomicProposition &expr, Atler::AST::Result &context) {
  Atler::AST::IntResult left;
  expr.getLeft().accept(*this, left);
  Atler::AST::IntResult right;
  expr.getRight().accept(*this, right);

  static_cast<Atler::AST::BoolResult &>(context).value = compare(left.value, expr.getOperator(), right.value);
  expr.eval = static_cast<Atler::AST::BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::BoolExpression &expr, Atler::AST::Result &context) {
  static_cast<Atler::AST::BoolResult &>(context).value = expr.getValue();
  expr.eval = expr.getValue();
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::NumberExpression &expr, Atler::AST::Result &context) {
  ((Atler::AST::IntResult &)context).value = expr.getValue();
  expr.eval = static_cast<Atler::AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::IdentifierExpression &expr, Atler::AST::Result &context) {
  ((Atler::AST::IntResult &)context).value = marking.numberOfTokensInPlace(expr.getPlace());
  expr.eval = static_cast<Atler::AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::MultiplyExpression &expr, Atler::AST::Result &context) {
  Atler::AST::IntResult left;
  expr.getLeft().accept(*this, left);
  Atler::AST::IntResult right;
  expr.getRight().accept(*this, right);
  ((Atler::AST::IntResult &)context).value = left.value * right.value;
  expr.eval = static_cast<Atler::AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::MinusExpression &expr, Atler::AST::Result &context) {
  Atler::AST::IntResult value;
  expr.getValue().accept(*this, value);
  ((Atler::AST::IntResult &)context).value = -value.value;
  expr.eval = static_cast<Atler::AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::SubtractExpression &expr, Atler::AST::Result &context) {
  Atler::AST::IntResult left;
  expr.getLeft().accept(*this, left);
  Atler::AST::IntResult right;
  expr.getRight().accept(*this, right);
  ((Atler::AST::IntResult &)context).value = left.value - right.value;
  expr.eval = static_cast<Atler::AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::PlusExpression &expr, Atler::AST::Result &context) {
  Atler::AST::IntResult left;
  expr.getLeft().accept(*this, left);
  Atler::AST::IntResult right;
  expr.getRight().accept(*this, right);
  ((Atler::AST::IntResult &)context).value = left.value + right.value;
  expr.eval = static_cast<Atler::AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(SimpleQuery &query, Atler::AST::Result &context) {
  query.getChild()->accept(*this, context);
  if (query.getQuantifier() == Atler::AST::AG || query.getQuantifier() == Atler::AST::AF || query.getQuantifier() == Atler::AST::PG) {
    static_cast<Atler::AST::BoolResult &>(context).value = !static_cast<Atler::AST::BoolResult &>(context).value;
  }
  query.eval = static_cast<Atler::AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(Atler::AST::DeadlockExpression &expr, Atler::AST::Result &context) {
  if (!deadlockChecked) {
    deadlockChecked = true;
    deadlocked = marking.canDeadlock(tapn, maxDelay);
  }
  static_cast<Atler::AST::BoolResult &>(context).value = deadlocked;
  expr.eval = static_cast<Atler::AST::BoolResult &>(context).value;
}

__host__ __device__ bool CudaQueryVisitor::compare(int numberOfTokensInPlace, Atler::AST::AtomicProposition::op_e op,
                                                     int n) const {

  switch (op) {
  case Atler::AST::AtomicProposition::LT:
    return numberOfTokensInPlace < n;
  case Atler::AST::AtomicProposition::LE:
    return numberOfTokensInPlace <= n;
  case Atler::AST::AtomicProposition::EQ:
    return numberOfTokensInPlace == n;
  case Atler::AST::AtomicProposition::NE:
    return numberOfTokensInPlace != n;
  default:
    assert(false);
  }
  return false;
}

} // namespace Cuda
} // namespace VerifyTAPN
#endif /* CUDAQUERYVISITOR_CUH_ */
