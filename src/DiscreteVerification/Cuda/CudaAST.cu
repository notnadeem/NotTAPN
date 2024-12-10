#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include <cstdio>

namespace VerifyTAPN::Cuda {
namespace AST {

// Definitions of CudaExpression methods
__host__ __device__ CudaExpression::CudaExpression()
    : type(UNKNOWN_EXPRESSION), eval(0), notExpr(nullptr) {}

__host__ __device__ CudaExpression::~CudaExpression() {
    // Properly delete the allocated memory
    switch (type) {
        case NOT_EXPRESSION:
            delete notExpr;
            break;
        case DEADLOCK_EXPRESSION:
            delete deadlockExpr;
            break;
        case BOOL_EXPRESSION:
            delete boolExpr;
            break;
        case ATOMIC_PROPOSITION:
            delete atomicProp;
            break;
        case AND_EXPRESSION:
            delete andExpr;
            break;
        case OR_EXPRESSION:
            delete orExpr;
            break;
        default:
            break;
    }
}

__host__ __device__ CudaExpression* CudaExpression::clone() const {
    CudaExpression* newExpr = new CudaExpression();
    newExpr->type = this->type;
    newExpr->eval = this->eval;
    
    switch (this->type) {
        case NOT_EXPRESSION:
            newExpr->notExpr = this->notExpr ? this->notExpr->clone() : nullptr;
            break;
        case DEADLOCK_EXPRESSION:
            newExpr->deadlockExpr = this->deadlockExpr ? this->deadlockExpr->clone() : nullptr;
            break;
        case BOOL_EXPRESSION:
            newExpr->boolExpr = this->boolExpr ? this->boolExpr->clone() : nullptr;
            break;
        case ATOMIC_PROPOSITION:
            newExpr->atomicProp = this->atomicProp ? this->atomicProp->clone() : nullptr;
            break;
        case AND_EXPRESSION:
            newExpr->andExpr = this->andExpr ? this->andExpr->clone() : nullptr;
            break;
        case OR_EXPRESSION:
            newExpr->orExpr = this->orExpr ? this->orExpr->clone() : nullptr;
            break;
        default:
            printf("Unknown expression type: %d\n", this->type);
            break;
    }
    return newExpr;
}

__host__ __device__ void CudaExpression::accept(CudaVisitor &visitor, Result &context) {
    switch (type) {
        case NOT_EXPRESSION:
            notExpr->accept(visitor, context);
            break;
        case DEADLOCK_EXPRESSION:
            deadlockExpr->accept(visitor, context);
            break;
        case BOOL_EXPRESSION:
            boolExpr->accept(visitor, context);
            break;
        case ATOMIC_PROPOSITION:
            atomicProp->accept(visitor, context);
            break;
        case AND_EXPRESSION:
            andExpr->accept(visitor, context);
            break;
        case OR_EXPRESSION:
            orExpr->accept(visitor, context);
            break;
        default:
            printf("Unknown expression type: %d\n", type);
            break;
    }
}


// Definitions of NotExpression methods
__host__ __device__ NotExpression::NotExpression(CudaExpression* expr)
    : expr(expr) {}

__host__ __device__ NotExpression* NotExpression::clone() const {
    return new NotExpression(expr ? expr->clone() : nullptr);
}

__host__ __device__ void NotExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of DeadlockExpression methods
__host__ __device__ DeadlockExpression::DeadlockExpression() {}

__host__ __device__ DeadlockExpression* DeadlockExpression::clone() const {
    return new DeadlockExpression();
}

__host__ __device__ void DeadlockExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of BoolExpression methods
__host__ __device__ BoolExpression::BoolExpression(bool value)
    : value(value) {}

__host__ __device__ BoolExpression* BoolExpression::clone() const {
    return new BoolExpression(value);
}

__host__ __device__ void BoolExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of AtomicProposition methods
__host__ __device__ AtomicProposition::AtomicProposition(ArithmeticExpression* left, ArithmeticExpression* right)
    : left(left), right(right), op(EQ) {}

__host__ __device__ AtomicProposition::AtomicProposition(ArithmeticExpression* l, const char* sop, ArithmeticExpression* r)
    : left(l), right(r) {
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
        printf("Unknown operator: %s\n", sop);
        op = EQ; 
    }
}

__host__ __device__ AtomicProposition::AtomicProposition(ArithmeticExpression* left, op_e op, ArithmeticExpression* right)
    : left(left), right(right), op(op) {}

__host__ __device__ AtomicProposition* AtomicProposition::clone() const {
    return new AtomicProposition(left ? left->clone() : nullptr, op, right ? right->clone() : nullptr);
}

__host__ __device__ void AtomicProposition::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of AndExpression methods
__host__ __device__ AndExpression::AndExpression(CudaExpression* left, CudaExpression* right)
    : left(left), right(right) {}

__host__ __device__ AndExpression* AndExpression::clone() const {
    return new AndExpression(left ? left->clone() : nullptr, right ? right->clone() : nullptr);
}

__host__ __device__ void AndExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of OrExpression methods
__host__ __device__ OrExpression::OrExpression(CudaExpression* left, CudaExpression* right)
    : left(left), right(right) {}

__host__ __device__ OrExpression* OrExpression::clone() const {
    return new OrExpression(left ? left->clone() : nullptr, right ? right->clone() : nullptr);
}

__host__ __device__ void OrExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of ArithmeticExpression methods
__host__ __device__ ArithmeticExpression::ArithmeticExpression()
    : type(UNKNOWN_EXPRESSION), eval(0), plusExpr(nullptr) {}

__host__ __device__ ArithmeticExpression::~ArithmeticExpression() {
    // Properly delete the allocated memory
    switch (type) {
        case PLUS_EXPRESSION:
            delete plusExpr;
            break;
        case SUBTRACT_EXPRESSION:
            delete subtractExpr;
            break;
        case MINUS_EXPRESSION:
            delete minusExpr;
            break;
        case MULTIPLY_EXPRESSION:
            delete multiplyExpr;
            break;
        case NUMBER_EXPRESSION:
            delete numberExpr;
            break;
        case IDENTIFIER_EXPRESSION:
            delete identifierExpr;
            break;
        default:
            break;
    }
}

__host__ __device__ ArithmeticExpression* ArithmeticExpression::clone() const {
    ArithmeticExpression* newExpr = new ArithmeticExpression();
    newExpr->type = this->type;
    newExpr->eval = this->eval;
    
    switch (this->type) {
        case PLUS_EXPRESSION:
            newExpr->plusExpr = this->plusExpr ? this->plusExpr->clone() : nullptr;
            break;
        case SUBTRACT_EXPRESSION:
            newExpr->subtractExpr = this->subtractExpr ? this->subtractExpr->clone() : nullptr;
            break;
        case MINUS_EXPRESSION:
            newExpr->minusExpr = this->minusExpr ? this->minusExpr->clone() : nullptr;
            break;
        case MULTIPLY_EXPRESSION:
            newExpr->multiplyExpr = this->multiplyExpr ? this->multiplyExpr->clone() : nullptr;
            break;
        case NUMBER_EXPRESSION:
            newExpr->numberExpr = this->numberExpr ? this->numberExpr->clone() : nullptr;
            break;
        case IDENTIFIER_EXPRESSION:
            newExpr->identifierExpr = this->identifierExpr ? this->identifierExpr->clone() : nullptr;
            break;
        default:
            printf("Unknown expression type: %d\n", this->type);
            break;
    }
    return newExpr;
}

__host__ __device__ void ArithmeticExpression::accept(CudaVisitor &visitor, Result &context) {
    switch (type) {
        case PLUS_EXPRESSION:
            plusExpr->accept(visitor, context);
            break;
        case SUBTRACT_EXPRESSION:
            subtractExpr->accept(visitor, context);
            break;
        case MINUS_EXPRESSION:
            minusExpr->accept(visitor, context);
            break;
        case MULTIPLY_EXPRESSION:
            multiplyExpr->accept(visitor, context);
            break;
        case NUMBER_EXPRESSION:
            numberExpr->accept(visitor, context);
            break;
        case IDENTIFIER_EXPRESSION:
            identifierExpr->accept(visitor, context);
            break;
        default:
            printf("Unknown expression type: %d\n", type);
            break;
    }
}

// Definitions of PlusExpression methods
__host__ __device__ PlusExpression::PlusExpression(ArithmeticExpression* left, ArithmeticExpression* right)
    : left(left), right(right) {}

__host__ __device__ PlusExpression* PlusExpression::clone() const {
    return new PlusExpression(left ? left->clone() : nullptr, right ? right->clone() : nullptr);
}

__host__ __device__ void PlusExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of SubtractExpression methods
__host__ __device__ SubtractExpression::SubtractExpression(ArithmeticExpression* left, ArithmeticExpression* right)
    : left(left), right(right) {}

__host__ __device__ SubtractExpression* SubtractExpression::clone() const {
    return new SubtractExpression(left ? left->clone() : nullptr, right ? right->clone() : nullptr);
}

__host__ __device__ void SubtractExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of MinusExpression methods
__host__ __device__ MinusExpression::MinusExpression(ArithmeticExpression* value)
    : value(value) {}

__host__ __device__ MinusExpression* MinusExpression::clone() const {
    return new MinusExpression(value ? value->clone() : nullptr);
}

__host__ __device__ void MinusExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of MultiplyExpression methods
__host__ __device__ MultiplyExpression::MultiplyExpression(ArithmeticExpression* left, ArithmeticExpression* right)
    : left(left), right(right) {}

__host__ __device__ MultiplyExpression* MultiplyExpression::clone() const {
    return new MultiplyExpression(left ? left->clone() : nullptr, right ? right->clone() : nullptr);
}

__host__ __device__ void MultiplyExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of NumberExpression methods
__host__ __device__ NumberExpression::NumberExpression(int value)
    : value(value) {}

__host__ __device__ NumberExpression* NumberExpression::clone() const {
    return new NumberExpression(*this);
}

__host__ __device__ void NumberExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of IdentifierExpression methods
__host__ __device__ IdentifierExpression::IdentifierExpression(int place)
    : place(place) {}

__host__ __device__ IdentifierExpression* IdentifierExpression::clone() const {
    return new IdentifierExpression(*this);
}

__host__ __device__ void IdentifierExpression::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

// Definitions of CudaQuery methods
__host__ __device__ CudaQuery::CudaQuery(CudaQuantifier quantifier, CudaExpression* expr)
    : quantifier(quantifier), expr(expr) {}

__host__ __device__ CudaQuery::CudaQuery(const CudaQuery& other)
    : quantifier(other.quantifier), expr(other.expr ? other.expr->clone() : nullptr) {}

__host__ __device__ CudaQuery& CudaQuery::operator=(const CudaQuery& other) {
    if (&other != this) {
        delete expr;
        expr = other.expr ? other.expr->clone() : nullptr;
        quantifier = other.quantifier;
    }
    return *this;
}

__host__ __device__ CudaQuery::~CudaQuery() {
    delete expr;
}

__host__ __device__ CudaQuery* CudaQuery::clone() const {
    return new CudaQuery(*this);
}

__host__ __device__ void CudaQuery::accept(CudaVisitor &visitor, Result &context) {
    visitor.visit(*this, context);
}

__host__ __device__ CudaQuantifier CudaQuery::getQuantifier() const {
    return quantifier;
}

__host__ __device__ const CudaExpression& CudaQuery::getConstChild() const {
    return *expr;
}

__host__ __device__ CudaExpression* CudaQuery::getChild() {
    return expr;
}

__host__ __device__ void CudaQuery::setChild(CudaExpression* expr) {
    this->expr = expr;
}

__host__ __device__ void CudaQuery::setQuantifier(CudaQuantifier q) {
    quantifier = q;
}

__host__ __device__ bool CudaQuery::hasSMCQuantifier() const {
    return quantifier == PF || quantifier == PG;
}

} // namespace AST
} // namespace VerifyTAPN::Cuda