#ifndef SMCQUERYALLOCATOR_CUH_
#define SMCQUERYALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"
#include "DiscreteVerification/Cuda/CudaAST.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {
using namespace VerifyTAPN::Cuda::AST;

struct SMCQueryAllocator {
  __host__ static CudaSMCQuery* allocate(CudaSMCQuery *h_smcQuery) {
    if (h_smcQuery == nullptr) {
        printf("Error: SMCQuery is null\n");
        return nullptr;
    }

    CudaSMCQuery *d_smcQuery;
    checkCudaError(cudaMalloc(&d_smcQuery, sizeof(CudaSMCQuery)), "Failed to allocate CudaSMCQuery on device");

    CudaSMCQuery *temp_query = (CudaSMCQuery *)malloc(sizeof(CudaSMCQuery));

    CudaExpression *d_expr;
    if (h_smcQuery->expr != nullptr) {
        d_expr = allocateExpression(h_smcQuery->expr);
    }

    temp_query->quantifier = h_smcQuery->quantifier;
    temp_query->smcSettings = h_smcQuery->smcSettings;
    temp_query->expr = d_expr;

    checkCudaError(cudaMemcpy(d_smcQuery, temp_query, sizeof(CudaSMCQuery), cudaMemcpyHostToDevice),
                  "Failed to copy CudaSMCQuery to device");

    free(temp_query);

    return d_smcQuery;
  }

__host__ static CudaExpression* allocateExpression(CudaExpression *h_expr) {
    if (h_expr == nullptr) {
        return nullptr;
    }

    CudaExpression* d_expr;
    checkCudaError(cudaMalloc(&d_expr, sizeof(CudaExpression)), "Failed to allocate CudaExpression on device");

    CudaExpression* temp_expr = (CudaExpression*)malloc(sizeof(CudaExpression));
    temp_expr->type = h_expr->type;
    temp_expr->eval = h_expr->eval;

    switch (h_expr->type) {
        case NOT_EXPRESSION: {
            Cuda::AST::NotExpression *d_not_expr;
            checkCudaError(cudaMalloc(&d_not_expr, sizeof(Cuda::AST::NotExpression)), "Failed to allocate NOT_EXPRESSION on device");

            Cuda::AST::NotExpression *temp_not_expr = (Cuda::AST::NotExpression *)malloc(sizeof(Cuda::AST::NotExpression));
            temp_not_expr->expr = allocateExpression(h_expr->notExpr->expr);

            checkCudaError(cudaMemcpy(d_not_expr, temp_not_expr, sizeof(Cuda::AST::NotExpression), cudaMemcpyHostToDevice),
                            "Failed to copy NotExpression to device");

            free(temp_not_expr);

            temp_expr->notExpr = d_not_expr;
            break;
        }
        case DEADLOCK_EXPRESSION: {
            Cuda::AST::DeadlockExpression *d_deadlock_expr;
            checkCudaError(cudaMalloc(&d_deadlock_expr, sizeof(Cuda::AST::DeadlockExpression)), "Failed to allocate DEADLOCK_EXPRESSION on device");

            Cuda::AST::DeadlockExpression *temp_deadlock_expr = (Cuda::AST::DeadlockExpression *)malloc(sizeof(Cuda::AST::DeadlockExpression));
            checkCudaError(cudaMemcpy(d_deadlock_expr, temp_deadlock_expr, sizeof(Cuda::AST::DeadlockExpression), cudaMemcpyHostToDevice),
                            "Failed to copy DeadlockExpression to device");

            free(temp_deadlock_expr);

            temp_expr->deadlockExpr = d_deadlock_expr;
            break;
        }
        case BOOL_EXPRESSION: {
            Cuda::AST::BoolExpression *d_bool_expr;
            checkCudaError(cudaMalloc(&d_bool_expr, sizeof(Cuda::AST::BoolExpression)), "Failed to allocate BOOL_EXPRESSION on device");

            Cuda::AST::BoolExpression *temp_bool_expr = (Cuda::AST::BoolExpression *)malloc(sizeof(Cuda::AST::BoolExpression));
            temp_bool_expr->value = h_expr->boolExpr->value;

            checkCudaError(cudaMemcpy(d_bool_expr, temp_bool_expr, sizeof(Cuda::AST::BoolExpression), cudaMemcpyHostToDevice),
                            "Failed to copy BoolExpression to device");

            free(temp_bool_expr);

            temp_expr->boolExpr = d_bool_expr;
            break;
        }
        case ATOMIC_PROPOSITION: {
            Cuda::AST::AtomicProposition *d_atomic_prop;
            checkCudaError(cudaMalloc(&d_atomic_prop, sizeof(Cuda::AST::AtomicProposition)), "Failed to allocate ATOMIC_PROPOSITION on device");

            Cuda::AST::AtomicProposition *temp_atomic_prop = (Cuda::AST::AtomicProposition *)malloc(sizeof(Cuda::AST::AtomicProposition));
            temp_atomic_prop->left = allocateArithmeticExpression(h_expr->atomicProp->left);
            temp_atomic_prop->right = allocateArithmeticExpression(h_expr->atomicProp->right);
            temp_atomic_prop->op = h_expr->atomicProp->op;

            checkCudaError(cudaMemcpy(d_atomic_prop, temp_atomic_prop, sizeof(Cuda::AST::AtomicProposition), cudaMemcpyHostToDevice),
                            "Failed to copy AtomicProposition to device");

            free(temp_atomic_prop);

            temp_expr->atomicProp = d_atomic_prop;
            break;
        }
        case AND_EXPRESSION: {
            Cuda::AST::AndExpression *d_and_expr;
            checkCudaError(cudaMalloc(&d_and_expr, sizeof(Cuda::AST::AndExpression)), "Failed to allocate AND_EXPRESSION on device");

            CudaExpression *d_left_expr = allocateExpression(h_expr->andExpr->left);
            CudaExpression *d_right_expr = allocateExpression(h_expr->andExpr->right);

            Cuda::AST::AndExpression *temp_and_expr = (Cuda::AST::AndExpression *)malloc(sizeof(Cuda::AST::AndExpression));
            temp_and_expr->left = d_left_expr;
            temp_and_expr->right = d_right_expr;

            checkCudaError(cudaMemcpy(d_and_expr, temp_and_expr, sizeof(Cuda::AST::AndExpression), cudaMemcpyHostToDevice),
                            "Failed to copy AndExpression to device");

            free(temp_and_expr);

            temp_expr->andExpr = d_and_expr;
            break;
        }
        case OR_EXPRESSION: {
            Cuda::AST::OrExpression *d_or_expr;
            checkCudaError(cudaMalloc(&d_or_expr, sizeof(Cuda::AST::OrExpression)), "Failed to allocate OR_EXPRESSION on device");

            CudaExpression *d_left_expr = allocateExpression(h_expr->orExpr->left);
            CudaExpression *d_right_expr = allocateExpression(h_expr->orExpr->right);

            Cuda::AST::OrExpression *temp_or_expr = (Cuda::AST::OrExpression *)malloc(sizeof(Cuda::AST::OrExpression));
            temp_or_expr->left = d_left_expr;
            temp_or_expr->right = d_right_expr;

            checkCudaError(cudaMemcpy(d_or_expr, temp_or_expr, sizeof(Cuda::AST::OrExpression), cudaMemcpyHostToDevice),
                            "Failed to copy OrExpression to device");

            free(temp_or_expr);

            temp_expr->orExpr = d_or_expr;
            break;
        }
        default:
            printf("Unknown expression type: %d\n", h_expr->type);
            break;
    }

    checkCudaError(cudaMemcpy(d_expr, temp_expr, sizeof(CudaExpression), cudaMemcpyHostToDevice),
                  "Failed to copy CudaExpression to device");

    free(temp_expr);

    return d_expr;
}

__host__ static Cuda::AST::ArithmeticExpression* allocateArithmeticExpression(Cuda::AST::ArithmeticExpression *h_expr) {
    if (h_expr == nullptr) {
        return nullptr;
    }

    Cuda::AST::ArithmeticExpression* d_expr;
    checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::ArithmeticExpression)), "Failed to allocate ArithmeticExpression on device");

    Cuda::AST::ArithmeticExpression* temp_expr = (Cuda::AST::ArithmeticExpression*)malloc(sizeof(Cuda::AST::ArithmeticExpression));
    temp_expr->type = h_expr->type;
    temp_expr->eval = h_expr->eval;

    switch (h_expr->type) {
        case PLUS_EXPRESSION: {
            Cuda::AST::PlusExpression *d_plus_expr;
            checkCudaError(cudaMalloc(&d_plus_expr, sizeof(Cuda::AST::PlusExpression)), "Failed to allocate PLUS_EXPRESSION on device");

            Cuda::AST::PlusExpression *temp_plus_expr = (Cuda::AST::PlusExpression *)malloc(sizeof(Cuda::AST::PlusExpression));
            temp_plus_expr->left = allocateArithmeticExpression(h_expr->plusExpr->left);
            temp_plus_expr->right = allocateArithmeticExpression(h_expr->plusExpr->right);

            checkCudaError(cudaMemcpy(d_plus_expr, temp_plus_expr, sizeof(Cuda::AST::PlusExpression), cudaMemcpyHostToDevice),
                            "Failed to copy PlusExpression to device");

            free(temp_plus_expr);

            temp_expr->plusExpr = d_plus_expr;
            break;
        }
        case SUBTRACT_EXPRESSION: {
            Cuda::AST::SubtractExpression *d_subtract_expr;
            checkCudaError(cudaMalloc(&d_subtract_expr, sizeof(Cuda::AST::SubtractExpression)), "Failed to allocate SUBTRACT_EXPRESSION on device");

            Cuda::AST::SubtractExpression *temp_subtract_expr = (Cuda::AST::SubtractExpression *)malloc(sizeof(Cuda::AST::SubtractExpression));
            temp_subtract_expr->left = allocateArithmeticExpression(h_expr->subtractExpr->left);
            temp_subtract_expr->right = allocateArithmeticExpression(h_expr->subtractExpr->right);

            checkCudaError(cudaMemcpy(d_subtract_expr, temp_subtract_expr, sizeof(Cuda::AST::SubtractExpression), cudaMemcpyHostToDevice),
                            "Failed to copy SubtractExpression to device");

            free(temp_subtract_expr);

            temp_expr->subtractExpr = d_subtract_expr;
            break;
        }
        case MINUS_EXPRESSION: {
            Cuda::AST::MinusExpression *d_minus_expr;
            checkCudaError(cudaMalloc(&d_minus_expr, sizeof(Cuda::AST::MinusExpression)), "Failed to allocate MINUS_EXPRESSION on device");

            Cuda::AST::MinusExpression *temp_minus_expr = (Cuda::AST::MinusExpression *)malloc(sizeof(Cuda::AST::MinusExpression));
            temp_minus_expr->value = allocateArithmeticExpression(h_expr->minusExpr->value);

            checkCudaError(cudaMemcpy(d_minus_expr, temp_minus_expr, sizeof(Cuda::AST::MinusExpression), cudaMemcpyHostToDevice),
                            "Failed to copy MinusExpression to device");

            free(temp_minus_expr);

            temp_expr->minusExpr = d_minus_expr;
            break;
        }
        case MULTIPLY_EXPRESSION: {
            Cuda::AST::MultiplyExpression *d_multiply_expr;
            checkCudaError(cudaMalloc(&d_multiply_expr, sizeof(Cuda::AST::MultiplyExpression)), "Failed to allocate MULTIPLY_EXPRESSION on device");

            Cuda::AST::MultiplyExpression *temp_multiply_expr = (Cuda::AST::MultiplyExpression *)malloc(sizeof(Cuda::AST::MultiplyExpression));
            temp_multiply_expr->left = allocateArithmeticExpression(h_expr->multiplyExpr->left);
            temp_multiply_expr->right = allocateArithmeticExpression(h_expr->multiplyExpr->right);

            checkCudaError(cudaMemcpy(d_multiply_expr, temp_multiply_expr, sizeof(Cuda::AST::MultiplyExpression), cudaMemcpyHostToDevice),
                            "Failed to copy MultiplyExpression to device");

            free(temp_multiply_expr);

            temp_expr->multiplyExpr = d_multiply_expr;
            break;
        }
        case NUMBER_EXPRESSION: {
            Cuda::AST::NumberExpression *d_number_expr;
            checkCudaError(cudaMalloc(&d_number_expr, sizeof(Cuda::AST::NumberExpression)), "Failed to allocate NUMBER_EXPRESSION on device");

            Cuda::AST::NumberExpression *temp_number_expr = (Cuda::AST::NumberExpression *)malloc(sizeof(Cuda::AST::NumberExpression));
            temp_number_expr->value = h_expr->numberExpr->value;

            checkCudaError(cudaMemcpy(d_number_expr, temp_number_expr, sizeof(Cuda::AST::NumberExpression), cudaMemcpyHostToDevice),
                            "Failed to copy NumberExpression to device");

            free(temp_number_expr);

            temp_expr->numberExpr = d_number_expr;
            break;
        }
        case IDENTIFIER_EXPRESSION: {
            Cuda::AST::IdentifierExpression *d_identifier_expr;
            checkCudaError(cudaMalloc(&d_identifier_expr, sizeof(Cuda::AST::IdentifierExpression)), "Failed to allocate IDENTIFIER_EXPRESSION on device");

            Cuda::AST::IdentifierExpression *temp_identifier_expr = (Cuda::AST::IdentifierExpression *)malloc(sizeof(Cuda::AST::IdentifierExpression));
            temp_identifier_expr->place = h_expr->identifierExpr->place;

            checkCudaError(cudaMemcpy(d_identifier_expr, temp_identifier_expr, sizeof(Cuda::AST::IdentifierExpression), cudaMemcpyHostToDevice),
                            "Failed to copy IdentifierExpression to device");

            free(temp_identifier_expr);

            temp_expr->identifierExpr = d_identifier_expr;
            break;
        }
        default:
            printf("Unknown ArithmeticExpression type: %d\n", h_expr->type);
            break;
    }

    checkCudaError(cudaMemcpy(d_expr, temp_expr, sizeof(Cuda::AST::ArithmeticExpression), cudaMemcpyHostToDevice),
                  "Failed to copy ArithmeticExpression to device");

    free(temp_expr);

    return d_expr;
}

  __host__ static void checkCudaError(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(result));
    }
  }
};
} // namespace VerifyTAPN::Alloc

#endif /* SMCQUERYALLOCATOR_CUH_ */