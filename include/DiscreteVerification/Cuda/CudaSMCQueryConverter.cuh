#ifndef VERIFYYAPN_ATLER_CUDASMCQUERY_CONVERTER_HPP_
#define VERIFYYAPN_ATLER_CUDASMCQUERY_CONVERTER_HPP_

#include "Core/Query/AST.hpp"
#include "Core/Query/SMCQuery.hpp"
#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"

namespace VerifyTAPN {
namespace Cuda {
class CudaSMCQueryConverter {
public:
  static Atler::AST::SimpleExpression *convert(const VerifyTAPN::AST::Expression *expr);
  static VerifyTAPN::Cuda::AST::CudaSMCQuery *convert(VerifyTAPN::AST::SMCQuery *SMCQuery);
  static Atler::AST::ArithmeticExpression *convert(VerifyTAPN::AST::ArithmeticExpression *expr);
  static VerifyTAPN::Cuda::AST::CudaSMCSettings convertSMCSettings(const VerifyTAPN::AST::SMCSettings &settings);

private:
  static Atler::AST::AtomicProposition::op_e convertOperator(VerifyTAPN::AST::AtomicProposition::op_e op);
  static Atler::AST::SimpleQuantifier convertQuantifier(VerifyTAPN::AST::Quantifier q);
};

// Implementation
inline Atler::AST::SimpleExpression *CudaSMCQueryConverter::convert(const VerifyTAPN::AST::Expression *expr) {
  if (const auto *notExpr = dynamic_cast<const VerifyTAPN::AST::NotExpression *>(expr)) {
    return new Atler::AST::NotExpression(convert(&notExpr->getChild()));
  } else if (const auto *deadlock = dynamic_cast<const VerifyTAPN::AST::DeadlockExpression *>(expr)) {
    return new Atler::AST::DeadlockExpression();
  } else if (const auto *boolExpr = dynamic_cast<const VerifyTAPN::AST::BoolExpression *>(expr)) {
    return new Atler::AST::BoolExpression(boolExpr->getValue());
  } else if (const auto *atomic = dynamic_cast<const VerifyTAPN::AST::AtomicProposition *>(expr)) {
    return new Atler::AST::AtomicProposition(convert(&atomic->getLeft()), convertOperator(atomic->getOperator()),
                                             convert(&atomic->getRight()));
  } else if (const auto *andExpr = dynamic_cast<const VerifyTAPN::AST::AndExpression *>(expr)) {
    return new Atler::AST::AndExpression(convert(&andExpr->getLeft()), convert(&andExpr->getRight()));
  } else if (const auto *orExpr = dynamic_cast<const VerifyTAPN::AST::OrExpression *>(expr)) {
    return new Atler::AST::OrExpression(convert(&orExpr->getLeft()), convert(&orExpr->getRight()));
  }
  return nullptr;
}

inline Atler::AST::ArithmeticExpression *CudaSMCQueryConverter::convert(VerifyTAPN::AST::ArithmeticExpression *expr) {
  if (auto *plus = dynamic_cast<VerifyTAPN::AST::PlusExpression *>(expr)) {
    return new Atler::AST::PlusExpression(convert(&plus->getLeft()), convert(&plus->getRight()));
  } else if (auto *subtract = dynamic_cast<VerifyTAPN::AST::SubtractExpression *>(expr)) {
    return new Atler::AST::SubtractExpression(convert(&subtract->getLeft()), convert(&subtract->getRight()));
  } else if (const auto *minus = dynamic_cast<const VerifyTAPN::AST::MinusExpression *>(expr)) {
    return new Atler::AST::MinusExpression(convert(&minus->getValue()));
  } else if (auto *multiply = dynamic_cast<VerifyTAPN::AST::MultiplyExpression *>(expr)) {
    return new Atler::AST::MultiplyExpression(convert(&multiply->getLeft()), convert(&multiply->getRight()));
  } else if (const auto *number = dynamic_cast<const VerifyTAPN::AST::NumberExpression *>(expr)) {
    return new Atler::AST::NumberExpression(number->getValue());
  } else if (const auto *identifier = dynamic_cast<const VerifyTAPN::AST::IdentifierExpression *>(expr)) {
    return new Atler::AST::IdentifierExpression(identifier->getPlace());
  }
  return nullptr;
}

inline VerifyTAPN::Cuda::AST::CudaSMCSettings
CudaSMCQueryConverter::convertSMCSettings(const VerifyTAPN::AST::SMCSettings &settings) {
  return VerifyTAPN::Cuda::AST::CudaSMCSettings{.timeBound = settings.timeBound,
                                                .stepBound = settings.stepBound,
                                                .falsePositives = settings.falsePositives,
                                                .falseNegatives = settings.falseNegatives,
                                                .indifferenceRegionUp = settings.indifferenceRegionUp,
                                                .indifferenceRegionDown = settings.indifferenceRegionDown,
                                                .confidence = settings.confidence,
                                                .estimationIntervalWidth = settings.estimationIntervalWidth,
                                                .compareToFloat = settings.compareToFloat,
                                                .geqThan = settings.geqThan};
}

inline VerifyTAPN::Cuda::AST::CudaSMCQuery *CudaSMCQueryConverter::convert(VerifyTAPN::AST::SMCQuery *query) {
  return new VerifyTAPN::Cuda::AST::CudaSMCQuery(convertQuantifier(query->getQuantifier()),
                                                 convertSMCSettings(query->getSmcSettings()),
                                                 convert(&query->getConstChild()));
}

inline Atler::AST::AtomicProposition::op_e
CudaSMCQueryConverter::convertOperator(VerifyTAPN::AST::AtomicProposition::op_e op) {
  switch (op) {
  case VerifyTAPN::AST::AtomicProposition::LT:
    return Atler::AST::AtomicProposition::LT;
  case VerifyTAPN::AST::AtomicProposition::LE:
    return Atler::AST::AtomicProposition::LE;
  case VerifyTAPN::AST::AtomicProposition::EQ:
    return Atler::AST::AtomicProposition::EQ;
  case VerifyTAPN::AST::AtomicProposition::NE:
    return Atler::AST::AtomicProposition::NE;
  default:
    throw std::runtime_error("Unknown operator");
  }
}

inline Atler::AST::SimpleQuantifier CudaSMCQueryConverter::convertQuantifier(VerifyTAPN::AST::Quantifier q) {
  switch (q) {
  case VerifyTAPN::AST::EF:
    return Atler::AST::EF;
  case VerifyTAPN::AST::AG:
    return Atler::AST::AG;
  case VerifyTAPN::AST::EG:
    return Atler::AST::EG;
  case VerifyTAPN::AST::AF:
    return Atler::AST::AF;
  case VerifyTAPN::AST::CF:
    return Atler::AST::CF;
  case VerifyTAPN::AST::CG:
    return Atler::AST::CG;
  case VerifyTAPN::AST::PF:
    return Atler::AST::PF;
  case VerifyTAPN::AST::PG:
    return Atler::AST::PG;
  default:
    throw std::runtime_error("Unknown quantifier");
  }
}
} // namespace Cuda
} // namespace VerifyTAPN::AST
#endif
