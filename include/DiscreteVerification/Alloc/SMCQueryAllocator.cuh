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

    printf("Allocating expression of type: %d\n", h_expr->getType());

    switch (h_expr->getType()) {
        case NOT_EXPRESSION: {
            Cuda::AST::NotExpression *d_expr;
            CudaExpression *childDevice = allocateExpression(&(static_cast<Cuda::AST::NotExpression*>(h_expr))->getChild());

            (static_cast<Cuda::AST::NotExpression*>(h_expr))->expr = childDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::NotExpression)), "Failed to allocate device memory for NotExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::NotExpression*>(h_expr), sizeof(Cuda::AST::NotExpression), cudaMemcpyHostToDevice),
                        "Failed to copy NotExpression to device");

            return d_expr;
        }
        case DEADLOCK_EXPRESSION: {
            Cuda::AST::DeadlockExpression *d_expr;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::DeadlockExpression)), "Failed to allocate device memory for DeadlockExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::DeadlockExpression*>(h_expr), sizeof(Cuda::AST::DeadlockExpression), cudaMemcpyHostToDevice),
                        "Failed to copy DeadlockExpression to device");
            return d_expr;
        }
        case BOOL_EXPRESSION: {
            Cuda::AST::BoolExpression *d_expr;
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::BoolExpression)), "Failed to allocate device memory for BoolExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::BoolExpression*>(h_expr), sizeof(Cuda::AST::BoolExpression), cudaMemcpyHostToDevice),
                        "Failed to copy BoolExpression to device");

            return d_expr;
        }
        case ATOMIC_PROPOSITION: {
            Cuda::AST::AtomicProposition *d_expr;
            Cuda::AST::ArithmeticExpression *leftDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::AtomicProposition*>(h_expr))->getLeft());
            Cuda::AST::ArithmeticExpression *rightDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::AtomicProposition*>(h_expr))->getRight());

            Cuda::AST::AtomicProposition* temp_casted = static_cast<Cuda::AST::AtomicProposition*>(h_expr);

            (static_cast<Cuda::AST::AtomicProposition*>(h_expr))->left = leftDevice;
            (static_cast<Cuda::AST::AtomicProposition*>(h_expr))->right = rightDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::AtomicProposition)), "Failed to allocate device memory for AtomicProposition");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::AtomicProposition*>(h_expr), sizeof(Cuda::AST::AtomicProposition), cudaMemcpyHostToDevice),
                        "Failed to copy AtomicProposition to device");

            Cuda::AST::AtomicProposition *h_temp = (Cuda::AST::AtomicProposition *)malloc(sizeof(Cuda::AST::AtomicProposition));
            cudaMemcpy(h_temp, d_expr, sizeof(Cuda::AST::AtomicProposition), cudaMemcpyDeviceToHost);

            return d_expr;
        }
        case AND_EXPRESSION: {
            Cuda::AST::AndExpression *d_expr;
            CudaExpression *leftDevice = allocateExpression(&(static_cast<Cuda::AST::AndExpression*>(h_expr))->getLeft());
            CudaExpression *rightDevice = allocateExpression(&(static_cast<Cuda::AST::AndExpression*>(h_expr))->getRight());

            (static_cast<Cuda::AST::AndExpression*>(h_expr))->left = leftDevice;
            (static_cast<Cuda::AST::AndExpression*>(h_expr))->right = rightDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::AndExpression)), "Failed to allocate device memory for AndExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::AndExpression*>(h_expr), sizeof(Cuda::AST::AndExpression), cudaMemcpyHostToDevice),
                        "Failed to copy AndExpression to device");

            Cuda::AST::AndExpression *h_temp = (Cuda::AST::AndExpression *)malloc(sizeof(Cuda::AST::AndExpression));
            cudaMemcpy(h_temp, d_expr, sizeof(Cuda::AST::AndExpression), cudaMemcpyDeviceToHost);

            return d_expr;
        }
        case OR_EXPRESSION: {
            Cuda::AST::OrExpression *d_expr;
            CudaExpression *leftDevice = allocateExpression(&(static_cast<Cuda::AST::OrExpression*>(h_expr))->getLeft());
            CudaExpression *rightDevice = allocateExpression(&(static_cast<Cuda::AST::OrExpression*>(h_expr))->getRight());

            (static_cast<Cuda::AST::OrExpression*>(h_expr))->left = leftDevice;
            (static_cast<Cuda::AST::OrExpression*>(h_expr))->right = rightDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::OrExpression)), "Failed to allocate device memory for OrExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::OrExpression*>(h_expr), sizeof(Cuda::AST::OrExpression), cudaMemcpyHostToDevice),
                        "Failed to copy OrExpression to device");

            return d_expr;
        }
        default: {
            printf("Unsupported CudaExpression type for allocation\n");
            break;
        }
    }
}

__host__ static Cuda::AST::ArithmeticExpression* allocateArithmeticExpression(Cuda::AST::ArithmeticExpression *h_expr) {
    switch (h_expr->getType()) {
        case PLUS_EXPRESSION: {
            Cuda::AST::PlusExpression *d_expr;
            Cuda::AST::ArithmeticExpression *leftDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::PlusExpression*>(h_expr))->getLeft());
            Cuda::AST::ArithmeticExpression *rightDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::PlusExpression*>(h_expr))->getRight());

            (static_cast<Cuda::AST::PlusExpression*>(h_expr))->left = leftDevice;
            (static_cast<Cuda::AST::PlusExpression*>(h_expr))->right = rightDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::PlusExpression)), "Failed to allocate device memory for PlusExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::PlusExpression*>(h_expr), sizeof(Cuda::AST::PlusExpression), cudaMemcpyHostToDevice),
                        "Failed to copy PlusExpression to device");

            return d_expr;
        }
        case SUBTRACT_EXPRESSION: {
            Cuda::AST::SubtractExpression *d_expr;
            Cuda::AST::ArithmeticExpression *leftDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::SubtractExpression*>(h_expr))->getLeft());
            Cuda::AST::ArithmeticExpression *rightDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::SubtractExpression*>(h_expr))->getRight());

            (static_cast<Cuda::AST::SubtractExpression*>(h_expr))->left = leftDevice;
            (static_cast<Cuda::AST::SubtractExpression*>(h_expr))->right = rightDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::SubtractExpression)), "Failed to allocate device memory for SubtractExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::SubtractExpression*>(h_expr), sizeof(Cuda::AST::SubtractExpression), cudaMemcpyHostToDevice),
                        "Failed to copy SubtractExpression to device");

            return d_expr;
        }
        case MINUS_EXPRESSION: {
            Cuda::AST::MinusExpression *d_expr;
            Cuda::AST::ArithmeticExpression *valueDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::MinusExpression*>(h_expr))->getValue());

            (static_cast<Cuda::AST::MinusExpression*>(h_expr))->value = valueDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::MinusExpression)), "Failed to allocate device memory for MinusExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::MinusExpression*>(h_expr), sizeof(Cuda::AST::MinusExpression), cudaMemcpyHostToDevice),
                        "Failed to copy MinusExpression to device");

            return d_expr;
        }
        case MULTIPLY_EXPRESSION: {
            Cuda::AST::MultiplyExpression *d_expr;
            Cuda::AST::ArithmeticExpression *leftDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::MultiplyExpression*>(h_expr))->getLeft());
            Cuda::AST::ArithmeticExpression *rightDevice = allocateArithmeticExpression(&(static_cast<Cuda::AST::MultiplyExpression*>(h_expr))->getRight());

            (static_cast<Cuda::AST::MultiplyExpression*>(h_expr))->left = leftDevice;
            (static_cast<Cuda::AST::MultiplyExpression*>(h_expr))->right = rightDevice;

            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::MultiplyExpression)), "Failed to allocate device memory for MultiplyExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::MultiplyExpression*>(h_expr), sizeof(Cuda::AST::MultiplyExpression), cudaMemcpyHostToDevice),
                        "Failed to copy MultiplyExpression to device");

            return d_expr;
        }
        case NUMBER_EXPRESSION: {
            Cuda::AST::NumberExpression *d_expr;
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::NumberExpression)), "Failed to allocate device memory for NumberExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::NumberExpression*>(h_expr), sizeof(Cuda::AST::NumberExpression), cudaMemcpyHostToDevice),
                        "Failed to copy NumberExpression to device");

            return d_expr;
        }
        case IDENTIFIER_EXPRESSION: {
            Cuda::AST::IdentifierExpression *d_expr;
            checkCudaError(cudaMalloc(&d_expr, sizeof(Cuda::AST::IdentifierExpression)), "Failed to allocate device memory for IdentifierExpression");
            checkCudaError(cudaMemcpy(d_expr, static_cast<Cuda::AST::IdentifierExpression*>(h_expr), sizeof(Cuda::AST::IdentifierExpression), cudaMemcpyHostToDevice),
                        "Failed to copy IdentifierExpression to device");            

            Cuda::AST::IdentifierExpression *h_temp = (Cuda::AST::IdentifierExpression *)malloc(sizeof(Cuda::AST::IdentifierExpression));
            cudaMemcpy(h_temp, d_expr, sizeof(Cuda::AST::IdentifierExpression), cudaMemcpyDeviceToHost);

            return d_expr;
        }
        default: {
            printf("Unsupported ArithmeticExpression type for allocation\n");
            break;
        }
    }
}

  __host__ static void checkCudaError(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(result));
    }
  }
};
} // namespace VerifyTAPN::Alloc

#endif /* SMCQUERYALLOCATOR_CUH_ */