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
inline VerifyTAPN::Cuda::AST::CudaExpression* CudaSMCQueryConverter::convert(const VerifyTAPN::AST::Expression* expr) {
    using namespace VerifyTAPN::Cuda::AST;

    CudaExpression* cudaExpr = new CudaExpression();

    if (const auto* notExpr = dynamic_cast<const VerifyTAPN::AST::NotExpression*>(expr)) {
        cudaExpr->type = NOT_EXPRESSION;
        cudaExpr->notExpr = new NotExpression(convert(&notExpr->getChild()));
        return cudaExpr;
    } else if (const auto* deadlock = dynamic_cast<const VerifyTAPN::AST::DeadlockExpression*>(expr)) {
        cudaExpr->type = DEADLOCK_EXPRESSION;
        cudaExpr->deadlockExpr = new DeadlockExpression();
        return cudaExpr;
    } else if (const auto* boolExpr = dynamic_cast<const VerifyTAPN::AST::BoolExpression*>(expr)) {
        cudaExpr->type = BOOL_EXPRESSION;
        cudaExpr->boolExpr = new BoolExpression(boolExpr->getValue());
        return cudaExpr;
    } else if (const auto* atomic = dynamic_cast<const VerifyTAPN::AST::AtomicProposition*>(expr)) {
        cudaExpr->type = ATOMIC_PROPOSITION;
        cudaExpr->atomicProp = new AtomicProposition(
            convert(&atomic->getLeft()), 
            convertOperator(atomic->getOperator()),
            convert(&atomic->getRight())
        );
        return cudaExpr;
    } else if (const auto* andExpr = dynamic_cast<const VerifyTAPN::AST::AndExpression*>(expr)) {
        cudaExpr->type = AND_EXPRESSION;
        cudaExpr->andExpr = new AndExpression(convert(&andExpr->getLeft()), convert(&andExpr->getRight()));
        return cudaExpr;
    } else if (const auto* orExpr = dynamic_cast<const VerifyTAPN::AST::OrExpression*>(expr)) {
        cudaExpr->type = OR_EXPRESSION;
        cudaExpr->orExpr = new OrExpression(convert(&orExpr->getLeft()), convert(&orExpr->getRight()));
        return cudaExpr;
    }

    delete cudaExpr;
    return nullptr;
}

inline VerifyTAPN::Cuda::AST::ArithmeticExpression* CudaSMCQueryConverter::convert(VerifyTAPN::AST::ArithmeticExpression* expr) {
    using namespace VerifyTAPN::Cuda::AST;

    ArithmeticExpression* arithmeticExpr = new ArithmeticExpression();

    if (auto *plus = dynamic_cast<VerifyTAPN::AST::PlusExpression*>(expr)) {
        arithmeticExpr->type = PLUS_EXPRESSION;
        arithmeticExpr->plusExpr = new PlusExpression(convert(&plus->getLeft()), convert(&plus->getRight()));
        return arithmeticExpr;
    } else if (auto *subtract = dynamic_cast<VerifyTAPN::AST::SubtractExpression*>(expr)) {
        arithmeticExpr->type = SUBTRACT_EXPRESSION;
        arithmeticExpr->subtractExpr = new SubtractExpression(convert(&subtract->getLeft()), convert(&subtract->getRight()));
        return arithmeticExpr;
    } else if (const auto *minus = dynamic_cast<const VerifyTAPN::AST::MinusExpression*>(expr)) {
        arithmeticExpr->type = MINUS_EXPRESSION;
        arithmeticExpr->minusExpr = new MinusExpression(convert(&minus->getValue()));
        return arithmeticExpr;
    } else if (auto *multiply = dynamic_cast<VerifyTAPN::AST::MultiplyExpression*>(expr)) {
        arithmeticExpr->type = MULTIPLY_EXPRESSION;
        arithmeticExpr->multiplyExpr = new MultiplyExpression(convert(&multiply->getLeft()), convert(&multiply->getRight()));
        return arithmeticExpr;
    } else if (const auto *number = dynamic_cast<const VerifyTAPN::AST::NumberExpression*>(expr)) {
        arithmeticExpr->type = NUMBER_EXPRESSION;
        arithmeticExpr->numberExpr = new NumberExpression(number->getValue());
        return arithmeticExpr;
    } else if (const auto *identifier = dynamic_cast<const VerifyTAPN::AST::IdentifierExpression*>(expr)) {
        arithmeticExpr->type = IDENTIFIER_EXPRESSION;
        arithmeticExpr->identifierExpr = new IdentifierExpression(identifier->getPlace());
        return arithmeticExpr;
    }

    delete arithmeticExpr;
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
