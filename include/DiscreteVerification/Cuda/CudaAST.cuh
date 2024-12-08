#ifndef CUDAAST_CUH_
#define CUDAAST_CUH_

#include "CudaVisitor.cuh"

#include <cuda_runtime.h>
/*#include <iostream>*/
#include <string>
 
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

class Visitable {
public:
  __host__ __device__ virtual void accept(CudaVisitor &visitor, Result &context) = 0;
  int32_t eval = 0;
};

class CudaExpression : public Visitable {
public:
  __host__ __device__ virtual ~CudaExpression() = default;

  __host__ __device__ virtual CudaExpression *clone() const = 0;

  __host__ __device__ virtual ExpressionType getType() const { return ExpressionType(type); }

  __host__ __device__ virtual void setType(ExpressionType type) { this->type = type; }

protected:
  __host__ __device__ explicit CudaExpression() {}
  __host__ __device__ explicit CudaExpression(ExpressionType type) : type(type) {}
  int type = UNKNOWN_EXPRESSION;
};

class NotExpression : public CudaExpression {
public:
  __host__ __device__ explicit NotExpression(CudaExpression *expr) : CudaExpression(NOT_EXPRESSION),
                        expr(expr) {};

  __host__ __device__ NotExpression(const NotExpression &other) : CudaExpression(NOT_EXPRESSION),
                        expr(other.expr->clone()) {};

  __host__ __device__ NotExpression &operator=(const NotExpression &other) {
    if (&other != this) {
      delete expr;
      expr = other.expr->clone();
    }

    return *this;
  }

  __host__ ~NotExpression() override { delete expr; };

  __host__ __device__ NotExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

  __host__ __device__ CudaExpression &getChild() const { return *expr; }

public:
  CudaExpression *expr;
};

class DeadlockExpression : public CudaExpression {
public:
  __host__ __device__ explicit DeadlockExpression() : CudaExpression(DEADLOCK_EXPRESSION) {};

  __host__ ~DeadlockExpression() override = default;

  __host__ __device__ DeadlockExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;
};

class BoolExpression : public CudaExpression {
public:
  __host__ __device__ explicit BoolExpression(bool value) : CudaExpression(BOOL_EXPRESSION),
                            value(value) {};

  __host__ __device__ ~BoolExpression() override = default;

  __host__ __device__ BoolExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

  __host__ __device__ bool getValue() const { return value; };

private:
  bool value;
};

class AtomicProposition : public CudaExpression {
public:
  enum op_e { LT, LE, EQ, NE };

  __host__ __device__ AtomicProposition::AtomicProposition(ArithmeticExpression *l, char *sop,
                                     ArithmeticExpression *r)
    : CudaExpression(ATOMIC_PROPOSITION),
    left(l), right(r) {
    if (strcmp(sop, "=") == 0 || strcmp(sop, "==") == 0)
      op = EQ;
    else if (strcmp(sop, "!=") == 0)
      op = NE;
    else if (strcmp(sop, "<") == 0)
      op = LT;
    else if (strcmp(sop, "<=") == 0)
      op = LE;
    else if (strcmp(sop, ">=") == 0) {
      op = LE;
      std::swap(left, right);
    } else if (strcmp(sop, ">") == 0) {
      op = LT;
      std::swap(left, right);
    } else {
      printf("Unknown operator: %s\n");
    }
  }

  __host__ __device__ AtomicProposition(ArithmeticExpression *left, op_e op,
                    ArithmeticExpression *right) : CudaExpression(ATOMIC_PROPOSITION),
                    left(left), right(right), op(op) {};

  __host__ __device__ AtomicProposition &operator=(const AtomicProposition &other) {
    if (&other != this) {
      left = other.left;
      op = other.op;
      right = other.right;
    }
    return *this;
  }

  __host__ __device__ ~AtomicProposition() override = default;

  __host__ __device__ ArithmeticExpression &getLeft() const { return *left; };

  __host__ __device__ ArithmeticExpression &getRight() const { return *right; };

  __host__ __device__ op_e getOperator() const { return op; };

  __host__ __device__ AtomicProposition *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

public:
  ArithmeticExpression *left;
  ArithmeticExpression *right;
  op_e op;
};

class AndExpression : public CudaExpression {
public:
  __host__ __device__ AndExpression(CudaExpression *left, CudaExpression *right) : CudaExpression(AND_EXPRESSION),
        left(left), right(right) {};

  __host__ __device__ AndExpression(const AndExpression &other) : CudaExpression(AND_EXPRESSION),
        left(other.left->clone()), right(other.right->clone()) {};

  __host__ __device__ AndExpression &operator=(const AndExpression &other) {
    if (&other != this) {
      delete left;
      delete right;

      left = other.left->clone();
      right = other.right->clone();
    }
    return *this;
  }

  __host__ ~AndExpression() override {
    delete left;
    delete right;
  }

  __host__ __device__ AndExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

  __host__ __device__ CudaExpression &getLeft() const { return *left; }

  __host__ __device__ CudaExpression &getRight() const { return *right; }

public:
  CudaExpression *left;
  CudaExpression *right;
};

class OrExpression : public CudaExpression {
public:
  __host__ __device__ OrExpression(CudaExpression *left, CudaExpression *right)
        : CudaExpression(OR_EXPRESSION),
        left(left), right(right) {};

  __host__ __device__ OrExpression(const OrExpression &other) : CudaExpression(OR_EXPRESSION),
        left(other.left->clone()), right(other.right->clone()) {};

  __host__ __device__ OrExpression &operator=(const OrExpression &other) {
    if (&other != this) {
      delete left;
      delete right;

      left = other.left->clone();
      right = other.right->clone();
    }
    return *this;
  }

  __host__ ~OrExpression() override {
    delete left;
    delete right;
  };

  __host__ __device__ OrExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

  __host__ __device__ CudaExpression &getLeft() const { return *left; }

  __host__ __device__ CudaExpression &getRight() const { return *right; }

public:
  CudaExpression *left;
  CudaExpression *right;
};

class ArithmeticExpression : public Visitable {
public:
  __host__ __device__ virtual ~ArithmeticExpression() {};

  __host__ __device__ virtual ArithmeticExpression *clone() const = 0;

  __host__ __device__ virtual ExpressionType getType() const { return ExpressionType(type); }

  __host__ __device__ virtual void setType(ExpressionType type) { this->type = type; }

protected:
  __host__ __device__ explicit ArithmeticExpression() {}
  __host__ __device__ explicit ArithmeticExpression(ExpressionType type) : type(type) {}
  int type = UNKNOWN_EXPRESSION;
};

class OperationExpression : public ArithmeticExpression {
protected:
  __host__ __device__ OperationExpression() = default;

  __host__ __device__ OperationExpression(ArithmeticExpression *left, ArithmeticExpression *right)
      : left(left), right(right) {};

  __host__ __device__ OperationExpression(ArithmeticExpression *left, ArithmeticExpression *right, ExpressionType type)
      : ArithmeticExpression(type), left(left), right(right) {};

  __host__ __device__ OperationExpression(const OperationExpression &other, ExpressionType type)
      : ArithmeticExpression(type), left(other.left), right(other.right) {};

  __host__ __device__ OperationExpression &operator=(const OperationExpression &other) {
    if (&other != this) {
      delete left;
      left = other.left;
      delete right;
      right = other.right;
    }
    return *this;
  }

  __host__ __device__ ~OperationExpression() override = default;

public:
  __host__ __device__ ArithmeticExpression &getLeft() { return *left; };

  __host__ __device__ ArithmeticExpression &getRight() { return *right; };

public:
  ArithmeticExpression *left;
  ArithmeticExpression *right;
};

class PlusExpression : public OperationExpression {
public:
  __host__ __device__ PlusExpression(ArithmeticExpression *left, ArithmeticExpression *right)
      : OperationExpression(left, right, PLUS_EXPRESSION) {};

  __host__ __device__ PlusExpression(const PlusExpression &other) : OperationExpression(other, PLUS_EXPRESSION) {};

  __host__ __device__ PlusExpression &operator=(const PlusExpression &other) {
    if (&other != this) {
      left = other.left;
      right = other.right;
    }
    return *this;
  }

  __host__ __device__ ~PlusExpression() override = default;

  __host__ __device__ PlusExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;
};

class SubtractExpression : public OperationExpression {
public:
  __host__ __device__ SubtractExpression(ArithmeticExpression *left, ArithmeticExpression *right)
      : OperationExpression(left, right, SUBTRACT_EXPRESSION) {};

  __host__ __device__ SubtractExpression(const SubtractExpression &other)
      : OperationExpression(other, SUBTRACT_EXPRESSION) {};

  __host__ __device__ SubtractExpression &operator=(const SubtractExpression &other) {
    if (&other != this) {
      left = other.left;
      right = other.right;
    }
    return *this;
  }

  __host__ __device__ ~SubtractExpression() override = default;

  __host__ __device__ SubtractExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;
};

class MinusExpression : public ArithmeticExpression {
public:
  __host__ __device__ explicit MinusExpression(ArithmeticExpression *value)
                    : ArithmeticExpression(MINUS_EXPRESSION),
                    value(value) {};

  __host__ __device__ MinusExpression(const MinusExpression &other)
                    : ArithmeticExpression(MINUS_EXPRESSION),
                    value(other.value) {};

  __host__ __device__ MinusExpression &operator=(const MinusExpression &other) {
    if (&other != this) {
      value = other.value;
    }
    return *this;
  }

  __host__ __device__ ArithmeticExpression &getValue() const { return *value; };

  __host__ __device__ ~MinusExpression() override = default;

  __host__ __device__ MinusExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

public:
  ArithmeticExpression *value;
};

class MultiplyExpression : public OperationExpression {
public:
  __host__ __device__ MultiplyExpression(ArithmeticExpression *left, ArithmeticExpression *right)
      : OperationExpression(left, right, MULTIPLY_EXPRESSION) {};

  __host__ __device__ MultiplyExpression(const MultiplyExpression &other)
      : OperationExpression(other, MULTIPLY_EXPRESSION) {};

  __host__ __device__ MultiplyExpression &operator=(const MultiplyExpression &other) {
    if (&other != this) {
      left = other.left;
      right = other.right;
    }
    return *this;
  }

  __host__ __device__ ~MultiplyExpression() override = default;

  __host__ __device__ MultiplyExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;
};

class NumberExpression : public ArithmeticExpression {
public:
  __host__ __device__ explicit NumberExpression(int i)
                : ArithmeticExpression(NUMBER_EXPRESSION),
                value(i) {};

  __host__ __device__ NumberExpression(const NumberExpression &other)
                : ArithmeticExpression(NUMBER_EXPRESSION),
                value(other.value) {};

  __host__ __device__ NumberExpression &operator=(const NumberExpression &other) {
    value = other.value;
    return *this;
  };

  __host__ __device__ int getValue() const { return value; };

  __host__ __device__ ~NumberExpression() override = default;

  __host__ __device__ NumberExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

private:
  int value;
};

class IdentifierExpression : public ArithmeticExpression {
public:
  __host__ __device__ explicit IdentifierExpression(int placeIndex)
              : ArithmeticExpression(IDENTIFIER_EXPRESSION),
              place(placeIndex) {}

  __host__ __device__ IdentifierExpression(const IdentifierExpression &other)
      : ArithmeticExpression(IDENTIFIER_EXPRESSION),
      place(other.place) {};

  __host__ __device__ IdentifierExpression &operator=(const IdentifierExpression &other) {
    place = other.place;
    return *this;
  };

  __host__ __device__ int getPlace() const { return place; };

  __host__ __device__ ~IdentifierExpression() override = default;

  __host__ __device__ IdentifierExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

public:
  int place;
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

class CudaQuery : public Visitable {
public:
  __host__ __device__ CudaQuery(CudaQuantifier quantifier, CudaExpression *expr)
      : quantifier(quantifier), expr(expr) {
        this->expr = expr;
      };

  __host__ __device__ CudaQuery(const CudaQuery &other)
      : quantifier(other.quantifier), expr(other.expr->clone()) {};

  __host__ __device__ CudaQuery &operator=(const CudaQuery &other) {
    if (&other != this) {
      delete expr;
      expr = other.expr->clone();
    }
    return *this;
  }

  __host__ __device__ virtual ~CudaQuery() { delete expr; }

  __host__ __device__ virtual CudaQuery *clone() const;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

  __host__ __device__ CudaQuantifier getQuantifier() const { return quantifier; }

  __host__ __device__ const CudaExpression &getConstChild() const { return *expr; }

  __host__ __device__ CudaExpression *getChild() { return expr; }

  __host__ __device__ void setChild(CudaExpression *expr) { this->expr = expr; }

  __host__ __device__ void setQuantifier(CudaQuantifier q) { quantifier = q; }

  __host__ __device__ bool hasSMCQuantifier() const { return quantifier == PF || quantifier == PG; }

private:
  CudaQuantifier quantifier;
  CudaExpression *expr;
};

} // namespace AST
} // namespace VerifyTAPN::Cuda

#endif /* AST_CUH_ */