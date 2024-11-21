#ifndef VERIFYYAPN_ATLER_SIMPLESMCQUERY_CONVERTER_HPP_
#define VERIFYYAPN_ATLER_SIMPLESMCQUERY_CONVERTER_HPP_

#include "Core/Query/AST.hpp"
#include "Core/Query/SMCQuery.hpp"
#include "DiscreteVerification/Atler/SimpleAST.hpp"
#include "DiscreteVerification/Atler/SimpleSMCQuery.hpp"

namespace VerifyTAPN::Atler::AST {

class SimpleSMCQueryConverter {
public:
    static AST::SimpleExpression* convert(const VerifyTAPN::AST::Expression* expr);
    static AST::SimpleSMCQuery* convert(VerifyTAPN::AST::SMCQuery* SMCQuery);
    static AST::ArithmeticExpression* convert(VerifyTAPN::AST::ArithmeticExpression* expr);
    static AST::SimpleSMCSettings convertSMCSettings(const VerifyTAPN::AST::SMCSettings& settings);

private:
    static AST::AtomicProposition::op_e convertOperator(VerifyTAPN::AST::AtomicProposition::op_e op);
    static AST::SimpleQuantifier convertQuantifier(VerifyTAPN::AST::Quantifier q);
};

// Implementation
inline AST::SimpleExpression* SimpleSMCQueryConverter::convert(const VerifyTAPN::AST::Expression* expr) {
    if (const auto* notExpr = dynamic_cast<const VerifyTAPN::AST::NotExpression*>(expr)) {
        return new AST::NotExpression(convert(&notExpr->getChild()));
    }
    else if (const auto* deadlock = dynamic_cast<const VerifyTAPN::AST::DeadlockExpression*>(expr)) {
        return new AST::DeadlockExpression();
    }
    else if (const auto* boolExpr = dynamic_cast<const VerifyTAPN::AST::BoolExpression*>(expr)) {
        return new AST::BoolExpression(boolExpr->getValue());
    }
    else if (const auto* atomic = dynamic_cast<const VerifyTAPN::AST::AtomicProposition*>(expr)) {
        return new AST::AtomicProposition(
            convert(&atomic->getLeft()),
            convertOperator(atomic->getOperator()),
            convert(&atomic->getRight())
        );
    }
    else if (const auto* andExpr = dynamic_cast<const VerifyTAPN::AST::AndExpression*>(expr)) {
        return new AST::AndExpression(
            convert(&andExpr->getLeft()),
            convert(&andExpr->getRight())
        );
    }
    else if (const auto* orExpr = dynamic_cast<const VerifyTAPN::AST::OrExpression*>(expr)) {
        return new AST::OrExpression(
            convert(&orExpr->getLeft()),
            convert(&orExpr->getRight())
        );
    }
    return nullptr;
}

inline AST::ArithmeticExpression* SimpleSMCQueryConverter::convert( VerifyTAPN::AST::ArithmeticExpression* expr) {
    if (auto* plus = dynamic_cast< VerifyTAPN::AST::PlusExpression*>(expr)) {
        return new AST::PlusExpression(
            convert(&plus->getLeft()),
            convert(&plus->getRight())
        );
    }
    else if ( auto* subtract = dynamic_cast< VerifyTAPN::AST::SubtractExpression*>(expr)) {
        return new AST::SubtractExpression(
            convert(&subtract->getLeft()),
            convert(&subtract->getRight())
        );
    }
    else if (const auto* minus = dynamic_cast<const VerifyTAPN::AST::MinusExpression*>(expr)) {
        return new AST::MinusExpression(convert(&minus->getValue()));
    }
    else if (auto* multiply = dynamic_cast<VerifyTAPN::AST::MultiplyExpression*>(expr)) {
        return new AST::MultiplyExpression(
            convert(&multiply->getLeft()),
            convert(&multiply->getRight())
        );
    }
    else if (const auto* number = dynamic_cast<const VerifyTAPN::AST::NumberExpression*>(expr)) {
        return new AST::NumberExpression(number->getValue());
    }
    else if (const auto* identifier = dynamic_cast<const VerifyTAPN::AST::IdentifierExpression*>(expr)) {
        return new AST::IdentifierExpression(identifier->getPlace());
    }
    return nullptr;
}

inline AST::SimpleSMCSettings SimpleSMCQueryConverter::convertSMCSettings(const VerifyTAPN::AST::SMCSettings& settings) {
    return AST::SimpleSMCSettings{
        .timeBound = settings.timeBound,
        .stepBound = settings.stepBound,
        .falsePositives = settings.falsePositives,
        .falseNegatives = settings.falseNegatives,
        .indifferenceRegionUp = settings.indifferenceRegionUp,
        .indifferenceRegionDown = settings.indifferenceRegionDown,
        .confidence = settings.confidence,
        .estimationIntervalWidth = settings.estimationIntervalWidth,
        .compareToFloat = settings.compareToFloat,
        .geqThan = settings.geqThan
    };
}

inline AST::SimpleSMCQuery* SimpleSMCQueryConverter::convert(VerifyTAPN::AST::SMCQuery* query) {
    return new AST::SimpleSMCQuery(
        convertQuantifier(query->getQuantifier()),
        convertSMCSettings(query->getSmcSettings()),
        convert(&query->getConstChild()));
}

inline AST::AtomicProposition::op_e SimpleSMCQueryConverter::convertOperator(VerifyTAPN::AST::AtomicProposition::op_e op) {
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

inline AST::SimpleQuantifier SimpleSMCQueryConverter::convertQuantifier(VerifyTAPN::AST::Quantifier q) {
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

} // namespace VerifyTAPN::Atler
#endif
