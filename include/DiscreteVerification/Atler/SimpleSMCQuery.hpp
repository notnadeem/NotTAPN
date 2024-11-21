#ifndef SIMPLESMCQUERIES_H
#define SIMPLESMCQUERIES_H

#include "SimpleAST.hpp"

namespace VerifyTAPN::Atler::AST {

struct SimpleSMCSettings {
  int timeBound;
  int stepBound;
  float falsePositives;
  float falseNegatives;
  float indifferenceRegionUp;
  float indifferenceRegionDown;
  float confidence;
  float estimationIntervalWidth;
  bool compareToFloat;
  float geqThan;
};

class SimpleSMCQuery : public SimpleQuery {
public:
  SimpleSMCQuery(SimpleQuantifier quantifier, SimpleSMCSettings settings,
                 SimpleExpression *expr)
      : SimpleQuery(quantifier, expr), smcSettings(settings) {};

  SimpleSMCQuery(const SimpleSMCQuery &other)
      : SimpleQuery(other.quantifier, other.expr->clone()),
        smcSettings(other.smcSettings) {};

  SimpleSMCQuery &operator=(const SimpleSMCQuery &other) {
    if (&other != this) {
      delete expr;
      expr = other.expr->clone();
      smcSettings = other.smcSettings;
    }
    return *this;
  }

  virtual SimpleSMCQuery *clone() const override;

  void accept(SimpleVisitor &visitor, Result &context) override;

  void setSMCSettings(SimpleSMCSettings newSettings) {
    smcSettings = newSettings;
  }

  SimpleSMCSettings &getSmcSettings() { return smcSettings; }

public:
  SimpleQuantifier quantifier;
  SimpleExpression *expr;
  SimpleSMCSettings smcSettings;
};

} // namespace VerifyTAPN::Atler::AST

#endif
