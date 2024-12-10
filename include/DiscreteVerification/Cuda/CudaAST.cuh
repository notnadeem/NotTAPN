#ifndef CUDAAST_CUH_
#define CUDAAST_CUH_

#include "CudaVisitor.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN::Cuda {

namespace AST {

enum ExpressionType {
  UNKNOWN_EXPRESSION = 0,
  BOOL_EXPRESSION = 1,
  NOT_EXPRESSION = 2,
  DEADLOCK_EXPRESSION = 3,
  ATOMIC_PROPOSITION = 4,
  AND_EXPRESSION = 5,
  OR_EXPRESSION = 6,
  PLUS_EXPRESSION = 7,
  SUBTRACT_EXPRESSION = 8,
  MINUS_EXPRESSION = 9,
  MULTIPLY_EXPRESSION = 10,
  NUMBER_EXPRESSION = 11,
  IDENTIFIER_EXPRESSION = 12
};

// Forward declarations
struct NotExpression;
struct DeadlockExpression;
struct BoolExpression;
struct AtomicProposition;
struct AndExpression;
struct OrExpression;

struct PlusExpression;
struct SubtractExpression;
struct MinusExpression;
struct MultiplyExpression;
struct NumberExpression;
struct IdentifierExpression;

// Base struct for all CUDA expressions
struct CudaExpression {
    ExpressionType type;
    int32_t eval;

    union {
        NotExpression* notExpr;
        DeadlockExpression* deadlockExpr;
        BoolExpression* boolExpr;
        AtomicProposition* atomicProp;
        AndExpression* andExpr;
        OrExpression* orExpr;
    };

    __host__ __device__ CudaExpression();
    __host__ __device__ ~CudaExpression();
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
    __host__ __device__ CudaExpression* clone() const;
};

// NOT Expression
struct NotExpression {
    CudaExpression* expr;

    __host__ __device__ NotExpression(CudaExpression* expr = nullptr);
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
    __host__ __device__ NotExpression* clone() const;
};

// Deadlock Expression
struct DeadlockExpression {
    __host__ __device__ DeadlockExpression();
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
    __host__ __device__ DeadlockExpression* clone() const;
};

// Bool Expression
struct BoolExpression {
    bool value;

    __host__ __device__ BoolExpression(bool value = false);
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
    __host__ __device__ BoolExpression* clone() const;
};

// Atomic Proposition
struct AtomicProposition {
    enum op_e { LT, LE, EQ, NE };

    ArithmeticExpression* left;
    ArithmeticExpression* right;
    op_e op;

    __host__ __device__ AtomicProposition(ArithmeticExpression* left = nullptr, ArithmeticExpression* right = nullptr);
    __host__ __device__ AtomicProposition(ArithmeticExpression* l, const char* sop, ArithmeticExpression* r);
    __host__ __device__ AtomicProposition(ArithmeticExpression* left, op_e op, ArithmeticExpression* right);
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
    __host__ __device__ AtomicProposition* clone() const;
};

// AND Expression
struct AndExpression {
    CudaExpression* left;
    CudaExpression* right;

    __host__ __device__ AndExpression(CudaExpression* left = nullptr, CudaExpression* right = nullptr);
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
    __host__ __device__ AndExpression* clone() const;
};

// OR Expression
struct OrExpression {
    CudaExpression* left;
    CudaExpression* right;

    __host__ __device__ OrExpression(CudaExpression* left = nullptr, CudaExpression* right = nullptr);
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
    __host__ __device__ OrExpression* clone() const;
};

// ArithmeticExpression - Base
struct ArithmeticExpression {
    ExpressionType type;
    int32_t eval;

    union {
        PlusExpression* plusExpr;
        SubtractExpression* subtractExpr;
        MinusExpression* minusExpr;
        MultiplyExpression* multiplyExpr;
        NumberExpression* numberExpr;
        IdentifierExpression* identifierExpr;
    };

    __host__ __device__ ArithmeticExpression();
    __host__ __device__ ~ArithmeticExpression();
    __host__ __device__ ArithmeticExpression* clone() const;
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
};

// PlusExpression (Operation Expression)
struct PlusExpression {
    ArithmeticExpression *left;
    ArithmeticExpression *right;

    __host__ __device__ PlusExpression(ArithmeticExpression *left = nullptr, ArithmeticExpression *right = nullptr);
    __host__ __device__ PlusExpression* clone() const;
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
};

// SubtractExpression (Operation Expression)
struct SubtractExpression {
    ArithmeticExpression *left;
    ArithmeticExpression *right;

    __host__ __device__ SubtractExpression(ArithmeticExpression *left = nullptr, ArithmeticExpression *right = nullptr);
    __host__ __device__ SubtractExpression* clone() const;
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
};

// MinusExpression (Unary operation)
struct MinusExpression {
    ArithmeticExpression *value;

    __host__ __device__ MinusExpression(ArithmeticExpression *value = nullptr);
    __host__ __device__ MinusExpression* clone() const;
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
};

// MultiplyExpression (Operation Expression)
struct MultiplyExpression {
    ArithmeticExpression *left;
    ArithmeticExpression *right;

    __host__ __device__ MultiplyExpression(ArithmeticExpression *left = nullptr, ArithmeticExpression *right = nullptr);
    __host__ __device__ MultiplyExpression* clone() const;
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
};

// NumberExpression (Literal value)
struct NumberExpression {
    int value;

    __host__ __device__ NumberExpression(int value = 0);
    __host__ __device__ NumberExpression* clone() const;
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
};

// IdentifierExpression
struct IdentifierExpression {
    int place;

    __host__ __device__ IdentifierExpression(int place = 0);
    __host__ __device__ IdentifierExpression* clone() const;
    __host__ __device__ void accept(CudaVisitor &visitor, Result &context);
};

// EF : Reachability
// AG : Safety
// EG : Preservability
// AF : Liveness
// CF : Control liveness
// CG : Control Safety
// PF : Probability Finally
// PG : Probability Globally
enum CudaQuantifier { EF, AG, EG, AF, CF, CG, PF, PG };

struct CudaQuery {
  int32_t eval;

  __host__ __device__ CudaQuery(CudaQuantifier quantifier, CudaExpression *expr);
  __host__ __device__ CudaQuery(const CudaQuery &other);
  __host__ __device__ CudaQuery &operator=(const CudaQuery &other);
  __host__ __device__ ~CudaQuery();
  __host__ __device__ virtual CudaQuery *clone() const;
  __host__ __device__ virtual void accept(CudaVisitor &visitor, Result &context);
  __host__ __device__ CudaQuantifier getQuantifier() const;
  __host__ __device__ const CudaExpression &getConstChild() const;
  __host__ __device__ CudaExpression *getChild();
  __host__ __device__ void setChild(CudaExpression *expr);
  __host__ __device__ void setQuantifier(CudaQuantifier q);
  __host__ __device__ bool hasSMCQuantifier() const;

  CudaQuantifier quantifier;
  CudaExpression *expr;
};

} // namespace AST
} // namespace VerifyTAPN::Cuda

#endif /* CUDAAST_CUH_ */