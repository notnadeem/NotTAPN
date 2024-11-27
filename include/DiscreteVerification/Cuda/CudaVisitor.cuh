#ifndef SIMPLEVISITOR_CUH_
#define SIMPLEVISITOR_CUH_

#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {
namespace AST {
class NotExpression;

class OrExpression;

class AndExpression;

class AtomicProposition;

class DeadlockExpression;

class BoolExpression;

class NumberExpression;

class IdentifierExpression;

class MultiplyExpression;

class MinusExpression;

class SubtractExpression;

class PlusExpression;

class ArithmeticExpression;

class SimpleQuery;

class Result {};

template <typename T> class SpecificResult : public Result {
public:
  T value;
};

typedef SpecificResult<int> IntResult;
typedef SpecificResult<bool> BoolResult;
/*typedef SpecificResult<std::vector<int>> IntVectorResult;*/

class SimpleVisitor {
public:
  __host__ __device__ virtual ~SimpleVisitor() = default;

  __host__ __device__ virtual void visit(NotExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(OrExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(AndExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(AtomicProposition &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(BoolExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(SimpleQuery &query, Result &context) = 0;

  __host__ __device__ virtual void visit(DeadlockExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(NumberExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(IdentifierExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(MultiplyExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(MinusExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(SubtractExpression &expr, Result &context) = 0;

  __host__ __device__ virtual void visit(PlusExpression &expr, Result &context) = 0;
};
} // namespace AST
} // namespace VerifyTAPN::Cuda

#endif /* VISITOR_CUH_ */
