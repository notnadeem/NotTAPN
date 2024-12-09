#ifndef SMCQUERYALLOCATOR_CUH_
#define SMCQUERYALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

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

    CudaSMCQuery *h_temp = (CudaSMCQuery *)malloc(sizeof(CudaSMCQuery));
    cudaMemcpy(h_temp, d_smcQuery, sizeof(CudaSMCQuery), cudaMemcpyDeviceToHost);

    return d_smcQuery;
  }

__host__ static CudaExpression* allocateExpression(CudaExpression *h_expr) {
    CudaExpression* temp_expr = (CudaExpression*)malloc(sizeof(CudaExpression));
    if (temp_expr == nullptr) {
        printf("Failed to allocate memory for expression on host\n");
        return nullptr;
    }

    temp_expr->type = h_expr->type;
    temp_expr->eval = h_expr->eval;

    CudaExpression* d_expr;

    switch (h_expr->type) {
        case NOT_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(CudaExpression)), "Failed to allocate NOT_EXPRESSION on device");
            temp_expr->notExpr = new Cuda::AST::NotExpression(h_expr->notExpr ? allocateExpression(h_expr->notExpr->expr) : nullptr);
            break;
        }
        case DEADLOCK_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(CudaExpression)), "Failed to allocate DEADLOCK_EXPRESSION on device");
            temp_expr->deadlockExpr = new Cuda::AST::DeadlockExpression();
            break;
        }
        case BOOL_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(CudaExpression)), "Failed to allocate BOOL_EXPRESSION on device");
            temp_expr->boolExpr = new Cuda::AST::BoolExpression(h_expr->boolExpr->value);
            break;
        }
        case ATOMIC_PROPOSITION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(CudaExpression)), "Failed to allocate ATOMIC_PROPOSITION on device");
            temp_expr->atomicProp = new Cuda::AST::AtomicProposition(
                h_expr->atomicProp->left ? static_cast<Cuda::AST::ArithmeticExpression*>(allocateArithmeticExpression(h_expr->atomicProp->left)) : nullptr,
                h_expr->atomicProp->op,
                h_expr->atomicProp->right ? static_cast<Cuda::AST::ArithmeticExpression*>(allocateArithmeticExpression(h_expr->atomicProp->right)) : nullptr
            );
            break;
        }
        case AND_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(CudaExpression)), "Failed to allocate AND_EXPRESSION on device");
            temp_expr->andExpr = new Cuda::AST::AndExpression(
                h_expr->andExpr->left ? allocateExpression(h_expr->andExpr->left) : nullptr,
                h_expr->andExpr->right ? allocateExpression(h_expr->andExpr->right) : nullptr
            );
            break;
        }
        case OR_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(CudaExpression)), "Failed to allocate OR_EXPRESSION on device");
            temp_expr->orExpr = new Cuda::AST::OrExpression(
                h_expr->orExpr->left ? allocateExpression(h_expr->orExpr->left) : nullptr,
                h_expr->orExpr->right ? allocateExpression(h_expr->orExpr->right) : nullptr
            );
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
    Cuda::AST::ArithmeticExpression* temp_expr = (Cuda::AST::ArithmeticExpression*)malloc(sizeof(Cuda::AST::ArithmeticExpression));
    if (temp_expr == nullptr) {
        printf("Failed to allocate memory for expression on host\n");
        return nullptr;
    }

    temp_expr->type = h_expr->type;
    temp_expr->eval = h_expr->eval;

    Cuda::AST::ArithmeticExpression* d_expr;

    switch (h_expr->type) {
        case PLUS_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::ArithmeticExpression)), "Failed to allocate PLUS_EXPRESSION on device");
            temp_expr->plusExpr = new Cuda::AST::PlusExpression(
                h_expr->plusExpr->left ? allocateArithmeticExpression(h_expr->plusExpr->left) : nullptr,
                h_expr->plusExpr->right ? allocateArithmeticExpression(h_expr->plusExpr->right) : nullptr
            );
            break;
        }
        case SUBTRACT_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::ArithmeticExpression)), "Failed to allocate SUBTRACT_EXPRESSION on device");
            temp_expr->subtractExpr = new Cuda::AST::SubtractExpression(
                h_expr->subtractExpr->left ? allocateArithmeticExpression(h_expr->subtractExpr->left) : nullptr,
                h_expr->subtractExpr->right ? allocateArithmeticExpression(h_expr->subtractExpr->right) : nullptr
            );
            break;
        }
        case MINUS_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::ArithmeticExpression)), "Failed to allocate MINUS_EXPRESSION on device");
            temp_expr->minusExpr = new Cuda::AST::MinusExpression(
                h_expr->minusExpr->value ? allocateArithmeticExpression(h_expr->minusExpr->value) : nullptr
            );
            break;
        }
        case MULTIPLY_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::ArithmeticExpression)), "Failed to allocate MULTIPLY_EXPRESSION on device");
            temp_expr->multiplyExpr = new Cuda::AST::MultiplyExpression(
                h_expr->multiplyExpr->left ? allocateArithmeticExpression(h_expr->multiplyExpr->left) : nullptr,
                h_expr->multiplyExpr->right ? allocateArithmeticExpression(h_expr->multiplyExpr->right) : nullptr
            );
            break;
        }
        case NUMBER_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::ArithmeticExpression)), "Failed to allocate NUMBER_EXPRESSION on device");
            temp_expr->numberExpr = new Cuda::AST::NumberExpression(h_expr->numberExpr->value);
            break;
        }
        case IDENTIFIER_EXPRESSION: {
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::ArithmeticExpression)), "Failed to allocate IDENTIFIER_EXPRESSION on device");
            temp_expr->identifierExpr = new Cuda::AST::IdentifierExpression(h_expr->identifierExpr->place);
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