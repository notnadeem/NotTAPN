#ifndef CUDAAST_HPP_
#define CUDAAST_HPP_

#include "CudaVisitor.cuh"

/*#include <iostream>*/
#include <string>
 
namespace VerifyTAPN::Cuda {

namespace AST {

class Visitable {
public:
  __host__ __device__ virtual void accept(CudaVisitor &visitor, Result &context) = 0;
  int32_t eval = 0;
};

class CudaExpression : public Visitable {
public:
  __host__ __device__ virtual ~CudaExpression() = default;

  __host__ __device__ virtual CudaExpression *clone() const = 0;
};

class NotExpression : public CudaExpression {
public:
  __host__ __device__ explicit NotExpression(CudaExpression *expr) : expr(expr) {};

  __host__ __device__ NotExpression(const NotExpression &other) : expr(other.expr->clone()) {};

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

private:
  CudaExpression *expr;
};

class DeadlockExpression : public CudaExpression {
public:
  __host__ __device__ explicit DeadlockExpression() = default;

  __host__ ~DeadlockExpression() override = default;

  __host__ __device__ DeadlockExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;
};

class BoolExpression : public CudaExpression {
public:
  __host__ __device__ explicit BoolExpression(bool value) : value(value) {};

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

  __host__ __device__ AtomicProposition(ArithmeticExpression *left, std::string *op,
                    ArithmeticExpression *right);

  __host__ __device__ AtomicProposition(ArithmeticExpression *left, op_e op,
                    ArithmeticExpression *right)
      : left(left), right(right), op(op) {};

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

private:
  ArithmeticExpression *left;
  ArithmeticExpression *right;
  op_e op;
};

class AndExpression : public CudaExpression {
public:
  __host__ __device__ AndExpression(CudaExpression *left, CudaExpression *right)
      : left(left), right(right) {};

  __host__ __device__ AndExpression(const AndExpression &other)
      : left(other.left->clone()), right(other.right->clone()) {};

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

private:
  CudaExpression *left;
  CudaExpression *right;
};

class OrExpression : public CudaExpression {
public:
  __host__ __device__ OrExpression(CudaExpression *left, CudaExpression *right)
      : left(left), right(right) {};

  __host__ __device__ OrExpression(const OrExpression &other)
      : left(other.left->clone()), right(other.right->clone()) {};

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

private:
  CudaExpression *left;
  CudaExpression *right;
};

class ArithmeticExpression : public Visitable {
public:
  __host__ __device__ virtual ~ArithmeticExpression() = default;

  __host__ __device__ virtual ArithmeticExpression *clone() const = 0;
};

class OperationExpression : public ArithmeticExpression {
protected:
  __host__ __device__ OperationExpression(ArithmeticExpression *left, ArithmeticExpression *right)
      : left(left), right(right) {};

  __host__ __device__ OperationExpression(const OperationExpression &other)
      : left(other.left), right(other.right) {};

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

protected:
  ArithmeticExpression *left;
  ArithmeticExpression *right;
};

class PlusExpression : public OperationExpression {
public:
  __host__ __device__ PlusExpression(ArithmeticExpression *left, ArithmeticExpression *right)
      : OperationExpression(left, right) {};

  __host__ __device__ PlusExpression(const PlusExpression &other) = default;

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
      : OperationExpression(left, right) {};

  __host__ __device__ SubtractExpression(const SubtractExpression &other) = default;

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
  __host__ __device__ explicit MinusExpression(ArithmeticExpression *value) : value(value) {};

  __host__ __device__ MinusExpression(const MinusExpression &other) : value(other.value) {};

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

private:
  ArithmeticExpression *value;
};

class MultiplyExpression : public OperationExpression {
public:
  __host__ __device__ MultiplyExpression(ArithmeticExpression *left, ArithmeticExpression *right)
      : OperationExpression(left, right) {};

  __host__ __device__ MultiplyExpression(const MultiplyExpression &other) = default;

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
  __host__ __device__ explicit NumberExpression(int i) : value(i) {}

  __host__ __device__ NumberExpression(const NumberExpression &other) : value(other.value) {};

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
  __host__ __device__ explicit IdentifierExpression(int placeIndex) : place(placeIndex) {}

  __host__ __device__ IdentifierExpression(const IdentifierExpression &other)
      : place(other.place) {};

  __host__ __device__ IdentifierExpression &operator=(const IdentifierExpression &other) {
    place = other.place;
    return *this;
  };

  __host__ __device__ int getPlace() const { return place; };

  __host__ __device__ ~IdentifierExpression() override = default;

  __host__ __device__ IdentifierExpression *clone() const override;

  __host__ __device__ void accept(CudaVisitor &visitor, Result &context) override;

private:
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
      : quantifier(quantifier), expr(expr) {};

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
} // namespace VerifyTAPN::Atler

#endif /* AST_HPP_ */