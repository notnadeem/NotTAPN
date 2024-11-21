#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include <cassert>

namespace VerifyTAPN::Atler {
namespace AST {

NotExpression *NotExpression::clone() const { return new NotExpression(*this); }

void NotExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

void BoolExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

BoolExpression *BoolExpression::clone() const {
  return new BoolExpression(*this);
}

void DeadlockExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

DeadlockExpression *DeadlockExpression::clone() const {
  return new DeadlockExpression(*this);
}

void AtomicProposition::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

AtomicProposition *AtomicProposition::clone() const {
  return new AtomicProposition(*this);
}

AndExpression *AndExpression::clone() const { return new AndExpression(*this); }

void AndExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

OrExpression *OrExpression::clone() const { return new OrExpression(*this); }

void OrExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

void PlusExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

PlusExpression *PlusExpression::clone() const {
  return new PlusExpression(*this);
}

void SubtractExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

SubtractExpression *SubtractExpression::clone() const {
  return new SubtractExpression(*this);
}

void MinusExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

MinusExpression *MinusExpression::clone() const {
  return new MinusExpression(*this);
}

void MultiplyExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

MultiplyExpression *MultiplyExpression::clone() const {
  return new MultiplyExpression(*this);
}

void NumberExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

NumberExpression *NumberExpression::clone() const {
  return new NumberExpression(*this);
}

void IdentifierExpression::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

IdentifierExpression *IdentifierExpression::clone() const {
  return new IdentifierExpression(*this);
}

SimpleQuery *SimpleQuery::clone() const { return new SimpleQuery(*this); }

void SimpleQuery::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

AtomicProposition::AtomicProposition(ArithmeticExpression *l, std::string *sop,
                                     ArithmeticExpression *r)
    : left(l), right(r) {
  if (*sop == "=" || *sop == "==")
    op = EQ;
  else if (*sop == "!=")
    op = NE;
  else if (*sop == "<")
    op = LT;
  else if (*sop == "<=")
    op = LE;
  else if (*sop == ">=") {
    op = LE;
    std::swap(left, right);
  } else if (*sop == ">") {
    op = LT;
    std::swap(left, right);
  } else {
    assert(false);
    throw std::exception();
  }
}

} // namespace AST
} // namespace VerifyTAPN::Atler
