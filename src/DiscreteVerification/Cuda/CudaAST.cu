#include "DiscreteVerification/Cuda/CudaAST.cuh"

namespace VerifyTAPN::Cuda {
namespace AST {

__host__ __device__ NotExpression *NotExpression::clone() const { return new NotExpression(*this); }

__host__ __device__ void NotExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ void BoolExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ BoolExpression *BoolExpression::clone() const {
  return new BoolExpression(*this);
}

__host__ __device__ void DeadlockExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ DeadlockExpression *DeadlockExpression::clone() const {
  return new DeadlockExpression(*this);
}

__host__ __device__ void AtomicProposition::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ AtomicProposition *AtomicProposition::clone() const {
  return new AtomicProposition(*this);
}

__host__ __device__ AndExpression *AndExpression::clone() const { return new AndExpression(*this); }

__host__ __device__ void AndExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ OrExpression *OrExpression::clone() const { return new OrExpression(*this); }

__host__ __device__ void OrExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ void PlusExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ PlusExpression *PlusExpression::clone() const {
  return new PlusExpression(*this);
}

__host__ __device__ void SubtractExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ SubtractExpression *SubtractExpression::clone() const {
  return new SubtractExpression(*this);
}

__host__ __device__ void MinusExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ MinusExpression *MinusExpression::clone() const {
  return new MinusExpression(*this);
}

__host__ __device__ void MultiplyExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ MultiplyExpression *MultiplyExpression::clone() const {
  return new MultiplyExpression(*this);
}

__host__ __device__ void NumberExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ NumberExpression *NumberExpression::clone() const {
  return new NumberExpression(*this);
}

__host__ __device__ void IdentifierExpression::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

__host__ __device__ IdentifierExpression *IdentifierExpression::clone() const {
  return new IdentifierExpression(*this);
}

__host__ __device__ CudaQuery *CudaQuery::clone() const { return new CudaQuery(*this); }

__host__ __device__ void CudaQuery::accept(CudaVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

} // namespace AST
} // namespace VerifyTAPN::Atler