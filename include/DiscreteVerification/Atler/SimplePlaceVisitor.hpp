#ifndef SIMPLEPLACEVISITOR_HPP_
#define SIMPLEPLACEVISITOR_HPP_

#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Atler/SimpleVisitor.hpp"
#include <exception>
#include <vector>

namespace VerifyTAPN::Atler {

using namespace AST;

class SimplePlaceVisitor : public SimpleVisitor {
public:
  SimplePlaceVisitor() = default;

  ~SimplePlaceVisitor() override = default;

public: // visitor methods
  void visit(NotExpression &expr, Result &context) override;

  void visit(OrExpression &expr, Result &context) override;

  void visit(AndExpression &expr, Result &context) override;

  void visit(AtomicProposition &expr, Result &context) override;

  void visit(BoolExpression &expr, Result &context) override;

  void visit(SimpleQuery &query, Result &context) override;

  void visit(DeadlockExpression &expr, Result &context) override;

  void visit(NumberExpression &expr, Result &context) override;

  void visit(IdentifierExpression &expr, Result &context) override;

  void visit(MinusExpression &expr, Result &context) override;

  virtual void visit(OperationExpression &expr, Result &context);

  void visit(MultiplyExpression &expr, Result &context) override {
    visit((OperationExpression &)expr, context);
  };

  void visit(SubtractExpression &expr, Result &context) override {
    visit((OperationExpression &)expr, context);
  };

  void visit(PlusExpression &expr, Result &context) override {
    visit((OperationExpression &)expr, context);
  };
};

} // namespace VerifyTAPN::Atler
#endif /* PLACEVISITOR_HPP_ */
