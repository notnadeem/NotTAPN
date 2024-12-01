#ifndef CUDAQUERYVISITOR_CUH_
#define CUDAQUERYVISITOR_CUH_

#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include "DiscreteVerification/Cuda/CudaVisitor.cuh"
#include "DiscreteVerification/Cuda/CudaRealMarking.cuh"
#include "DiscreteVerification/Cuda/CudaTimedArcPetriNet.cuh"

#include <cuda_runtime.h>

namespace VerifyTAPN {
namespace Cuda {

using namespace AST;

class CudaQueryVisitor : public CudaVisitor {
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
  __host__ __device__ void visit(AST::NotExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::OrExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::AndExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::AtomicProposition &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::BoolExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::CudaQuery &query, AST::Result &context) override;

  __host__ __device__ void visit(AST::DeadlockExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::NumberExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::IdentifierExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::MultiplyExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::MinusExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::SubtractExpression &expr, AST::Result &context) override;

  __host__ __device__ void visit(AST::PlusExpression &expr, AST::Result &context) override;

private:
  __host__ __device__ bool compare(int numberOfTokensInPlace, AST::AtomicProposition::op_e op, int n) const;

private:
  const CudaRealMarking &marking;
  const CudaTimedArcPetriNet &tapn;
  bool deadlockChecked;
  bool deadlocked;
  const int maxDelay;
};

__host__ __device__ void CudaQueryVisitor::visit(AST::NotExpression &expr, AST::Result &context) {
  AST::BoolResult c;
  expr.getChild().accept(*this, c);
  expr.eval = !c.value;
  static_cast<AST::BoolResult &>(context).value = expr.eval;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::OrExpression &expr, AST::Result &context) {
  AST::BoolResult left, right;
  expr.getLeft().accept(*this, left);
  // use lazy evaluation
  if (left.value) {
    static_cast<AST::BoolResult &>(context).value = true;
  } else {
    expr.getRight().accept(*this, right);
    static_cast<AST::BoolResult &>(context).value = right.value;
  }
  expr.eval = static_cast<AST::BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::AndExpression &expr, AST::Result &context) {
  AST::BoolResult left, right;
  expr.getLeft().accept(*this, left);

  // use lazy evaluation
  if (!left.value) {
    static_cast<AST::BoolResult &>(context).value = false;
  } else {
    expr.getRight().accept(*this, right);
    static_cast<AST::BoolResult &>(context).value = right.value;
  }
  expr.eval = static_cast<AST::BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::AtomicProposition &expr, AST::Result &context) {
  AST::IntResult left;
  expr.getLeft().accept(*this, left);
  AST::IntResult right;
  expr.getRight().accept(*this, right);

  static_cast<AST::BoolResult &>(context).value = compare(left.value, expr.getOperator(), right.value);
  expr.eval = static_cast<AST::BoolResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::BoolExpression &expr, AST::Result &context) {
  static_cast<AST::BoolResult &>(context).value = expr.getValue();
  expr.eval = expr.getValue();
}

__host__ __device__ void CudaQueryVisitor::visit(AST::NumberExpression &expr, AST::Result &context) {
  ((AST::IntResult &)context).value = expr.getValue();
  expr.eval = static_cast<AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::IdentifierExpression &expr, AST::Result &context) {
  ((AST::IntResult &)context).value = marking.numberOfTokensInPlace(expr.getPlace());
  expr.eval = static_cast<AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::MultiplyExpression &expr, AST::Result &context) {
  AST::IntResult left;
  expr.getLeft().accept(*this, left);
  AST::IntResult right;
  expr.getRight().accept(*this, right);
  ((AST::IntResult &)context).value = left.value * right.value;
  expr.eval = static_cast<AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::MinusExpression &expr, AST::Result &context) {
  AST::IntResult value;
  expr.getValue().accept(*this, value);
  ((AST::IntResult &)context).value = -value.value;
  expr.eval = static_cast<AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::SubtractExpression &expr, AST::Result &context) {
  AST::IntResult left;
  expr.getLeft().accept(*this, left);
  AST::IntResult right;
  expr.getRight().accept(*this, right);
  ((AST::IntResult &)context).value = left.value - right.value;
  expr.eval = static_cast<AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::PlusExpression &expr, AST::Result &context) {
  AST::IntResult left;
  expr.getLeft().accept(*this, left);
  AST::IntResult right;
  expr.getRight().accept(*this, right);
  ((AST::IntResult &)context).value = left.value + right.value;
  expr.eval = static_cast<AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::CudaQuery &query, AST::Result &context) {
  query.getChild()->accept(*this, context);
  if (query.getQuantifier() == AST::AG || query.getQuantifier() == AST::AF ||
      query.getQuantifier() == AST::PG) {
    static_cast<AST::BoolResult &>(context).value = !static_cast<AST::BoolResult &>(context).value;
  }
  query.eval = static_cast<AST::IntResult &>(context).value;
}

__host__ __device__ void CudaQueryVisitor::visit(AST::DeadlockExpression &expr, AST::Result &context) {
  if (!deadlockChecked) {
    deadlockChecked = true;
    deadlocked = marking.canDeadlock(tapn, maxDelay);
  }
  static_cast<AST::BoolResult &>(context).value = deadlocked;
  expr.eval = static_cast<AST::BoolResult &>(context).value;
}

__host__ __device__ bool CudaQueryVisitor::compare(int numberOfTokensInPlace, AST::AtomicProposition::op_e op,
                                                   int n) const {

  switch (op) {
  case AST::AtomicProposition::LT:
    return numberOfTokensInPlace < n;
  case AST::AtomicProposition::LE:
    return numberOfTokensInPlace <= n;
  case AST::AtomicProposition::EQ:
    return numberOfTokensInPlace == n;
  case AST::AtomicProposition::NE:
    return numberOfTokensInPlace != n;
  default:
    printf("Unknown operator\n");
  }
  return false;
}

} // namespace Cuda
} // namespace VerifyTAPN
#endif /* CUDAQUERYVISITOR_CUH_ */
