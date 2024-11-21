#include "DiscreteVerification/Atler/SimpleSMCQuery.hpp"

namespace VerifyTAPN::Atler::AST {

SimpleSMCQuery *SimpleSMCQuery::clone() const {
  return new SimpleSMCQuery(*this);
}

void SimpleSMCQuery::accept(SimpleVisitor &visitor, Result &context) {
  visitor.visit(*this, context);
}

} // namespace VerifyTAPN::Atler::AST
