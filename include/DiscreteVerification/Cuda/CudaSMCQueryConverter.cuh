#ifndef VERIFYYAPN_ATLER_CUDASMCQUERY_CONVERTER_HPP_
#define VERIFYYAPN_ATLER_CUDASMCQUERY_CONVERTER_HPP_

#include "Core/Query/AST.hpp"
#include "Core/Query/SMCQuery.hpp"
#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"

namespace VerifyTAPN {
namespace Cuda {
class CudaSMCQueryConverter {
public:
  static VerifyTAPN::Cuda::AST::CudaExpression *convert(const VerifyTAPN::AST::Expression *expr);
  static VerifyTAPN::Cuda::AST::CudaSMCQuery *convert(VerifyTAPN::AST::SMCQuery *SMCQuery);
  static AST::ArithmeticExpression *convert(VerifyTAPN::AST::ArithmeticExpression *expr);
  static VerifyTAPN::Cuda::AST::CudaSMCSettings convertSMCSettings(const VerifyTAPN::AST::SMCSettings &settings);

private:
  static AST::AtomicProposition::op_e convertOperator(VerifyTAPN::AST::AtomicProposition::op_e op);
  static VerifyTAPN::Cuda::AST::CudaQuantifier convertQuantifier(VerifyTAPN::AST::Quantifier q);
};

// Implementation
inline VerifyTAPN::Cuda::AST::CudaExpression *CudaSMCQueryConverter::convert(const VerifyTAPN::AST::Expression *expr) {
  if (const auto *notExpr = dynamic_cast<const VerifyTAPN::AST::NotExpression *>(expr)) {
    AST::NotExpression* expression = new AST::NotExpression(convert(&notExpr->getChild()));
    expression->setType(AST::NOT_EXPRESSION);
    return expression;
  } else if (const auto *deadlock = dynamic_cast<const VerifyTAPN::AST::DeadlockExpression *>(expr)) {
    AST::DeadlockExpression* expression = new AST::DeadlockExpression();
    expression->setType(AST::DEADLOCK_EXPRESSION);
    return expression;
  } else if (const auto *boolExpr = dynamic_cast<const VerifyTAPN::AST::BoolExpression *>(expr)) {
    AST::BoolExpression* expression = new AST::BoolExpression(boolExpr->getValue());
    expression->setType(AST::BOOL_EXPRESSION);
    return expression;
  } else if (const auto *atomic = dynamic_cast<const VerifyTAPN::AST::AtomicProposition *>(expr)) {
    AST::AtomicProposition* expression = new AST::AtomicProposition(convert(&atomic->getLeft()), convertOperator(atomic->getOperator()),
                                             convert(&atomic->getRight()));
    expression->setType(AST::ATOMIC_PROPOSITION);
    return expression;
  } else if (const auto *andExpr = dynamic_cast<const VerifyTAPN::AST::AndExpression *>(expr)) {
    AST::AndExpression* expression = new AST::AndExpression(convert(&andExpr->getLeft()), convert(&andExpr->getRight()));
    expression->setType(AST::AND_EXPRESSION);
    return expression;
  } else if (const auto *orExpr = dynamic_cast<const VerifyTAPN::AST::OrExpression *>(expr)) {
    AST::OrExpression* expression = new AST::OrExpression(convert(&orExpr->getLeft()), convert(&orExpr->getRight()));
    expression->setType(AST::OR_EXPRESSION);
    return expression;
  }
  return nullptr;
}

inline AST::ArithmeticExpression *CudaSMCQueryConverter::convert(VerifyTAPN::AST::ArithmeticExpression *expr) {
  if (auto *plus = dynamic_cast<VerifyTAPN::AST::PlusExpression *>(expr)) {
    AST::PlusExpression* expression = new AST::PlusExpression(convert(&plus->getLeft()), convert(&plus->getRight()));
    expression->setType(AST::PLUS_EXPRESSION);
    return expression;
  } else if (auto *subtract = dynamic_cast<VerifyTAPN::AST::SubtractExpression *>(expr)) {
    AST::SubtractExpression* expression = new AST::SubtractExpression(convert(&subtract->getLeft()), convert(&subtract->getRight()));
    expression->setType(AST::SUBTRACT_EXPRESSION);
    return expression;
  } else if (const auto *minus = dynamic_cast<const VerifyTAPN::AST::MinusExpression *>(expr)) {
    AST::MinusExpression* expression = new AST::MinusExpression(convert(&minus->getValue()));
    expression->setType(AST::MINUS_EXPRESSION);
    return expression;
  } else if (auto *multiply = dynamic_cast<VerifyTAPN::AST::MultiplyExpression *>(expr)) {
    AST::MultiplyExpression* expression = new AST::MultiplyExpression(convert(&multiply->getLeft()), convert(&multiply->getRight()));
    expression->setType(AST::MULTIPLY_EXPRESSION);
    return expression;
  } else if (const auto *number = dynamic_cast<const VerifyTAPN::AST::NumberExpression *>(expr)) {
    AST::NumberExpression* expression = new AST::NumberExpression(number->getValue());
    expression->setType(AST::NUMBER_EXPRESSION);
    return expression;
  } else if (const auto *identifier = dynamic_cast<const VerifyTAPN::AST::IdentifierExpression *>(expr)) {
    AST::IdentifierExpression* expression = new AST::IdentifierExpression(identifier->getPlace());
    expression->setType(AST::IDENTIFIER_EXPRESSION);
    return expression;
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
  VerifyTAPN::Cuda::AST::CudaExpression* testExpr = convert(&query->getConstChild());
  
  VerifyTAPN::Cuda::AST::CudaSMCQuery* result = new VerifyTAPN::Cuda::AST::CudaSMCQuery(convertQuantifier(query->getQuantifier()),
                                                 convertSMCSettings(query->getSmcSettings()),
                                                 convert(&query->getConstChild()));
  return result;
}

inline AST::AtomicProposition::op_e
CudaSMCQueryConverter::convertOperator(VerifyTAPN::AST::AtomicProposition::op_e op) {
  switch (op) {
  case VerifyTAPN::AST::AtomicProposition::LT:
    return AST::AtomicProposition::LT;
  case VerifyTAPN::AST::AtomicProposition::LE:
    return AST::AtomicProposition::LE;
  case VerifyTAPN::AST::AtomicProposition::EQ:
    return AST::AtomicProposition::EQ;
  case VerifyTAPN::AST::AtomicProposition::NE:
    return AST::AtomicProposition::NE;
  default:
    throw std::runtime_error("Unknown operator");
  }
}

inline AST::CudaQuantifier CudaSMCQueryConverter::convertQuantifier(VerifyTAPN::AST::Quantifier q) {
  switch (q) {
  case VerifyTAPN::AST::EF:
    return AST::EF;
  case VerifyTAPN::AST::AG:
    return AST::AG;
  case VerifyTAPN::AST::EG:
    return AST::EG;
  case VerifyTAPN::AST::AF:
    return AST::AF;
  case VerifyTAPN::AST::CF:
    return AST::CF;
  case VerifyTAPN::AST::CG:
    return AST::CG;
  case VerifyTAPN::AST::PF:
    return AST::PF;
  case VerifyTAPN::AST::PG:
    return AST::PG;
  default:
    throw std::runtime_error("Unknown quantifier");
  }
}
} // namespace Cuda
} // namespace VerifyTAPN::AST
#endif
