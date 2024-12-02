#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"
#include "DiscreteVerification/Alloc/SMCQueryAllocator.cuh"

#include <cuda_runtime.h>
// #include <stdexcept>
//#include <iostream>

namespace VerifyTAPN::Alloc {

using namespace Cuda::AST;

__host__ CudaSMCQuery* SMCQueryAllocator::allocate(CudaSMCQuery *smcQueryHost) { // TODO: Double check this
    //assert(smcQueryHost != nullptr);

    CudaSMCQuery *d_smcQuery = nullptr;
    checkCudaError(cudaMalloc(&d_smcQuery, sizeof(CudaSMCQuery)), "Failed to allocate CudaSMCQuery on device");

    checkCudaError(cudaMemcpy(d_smcQuery, smcQueryHost, sizeof(CudaSMCQuery), cudaMemcpyHostToDevice),
                  "Failed to copy CudaSMCQuery to device");

    allocatePointerMembers(smcQueryHost, d_smcQuery);

    return d_smcQuery;
}

__host__ void SMCQueryAllocator::allocatePointerMembers(CudaSMCQuery *smcQueryHost, CudaSMCQuery *d_smcQuery) { // TODO: Double check this
    //assert(smcQueryHost != nullptr && d_smcQuery != nullptr);

    if (smcQueryHost->expr != nullptr) {
        CudaExpression *d_expr = allocateExpression(smcQueryHost->expr);

        checkCudaError(cudaMemcpy(&(d_smcQuery->expr), &d_expr, sizeof(CudaExpression*), cudaMemcpyHostToDevice),
                      "Failed to update expr pointer in device CudaSMCQuery");
    }
}

__host__ CudaExpression* SMCQueryAllocator::allocateExpression(CudaExpression *exprHost) {
    //assert(exprHost != nullptr);

    CudaExpression *d_expr = nullptr;

    // NotExpression
    if (auto *notExpr = dynamic_cast<NotExpression*>(exprHost)) {
        NotExpression *host_copy = static_cast<NotExpression*>(notExpr->clone());

        CudaExpression *childDevice = allocateExpression(&host_copy->getChild());

        checkCudaError(cudaMalloc(&d_expr, sizeof(NotExpression)), "Failed to allocate device memory for NotExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(NotExpression), cudaMemcpyHostToDevice),
                    "Failed to copy NotExpression to device");

        checkCudaError(cudaMemcpy(&static_cast<NotExpression*>(d_expr)->getChild(), &childDevice, sizeof(CudaExpression*), cudaMemcpyHostToDevice),
                    "Failed to update child pointer in NotExpression on device");

        delete host_copy;
    }
    // DeadlockExpression
    else if (auto *deadlockExpr = dynamic_cast<DeadlockExpression*>(exprHost)) {
        DeadlockExpression *host_copy = deadlockExpr->clone();

        checkCudaError(cudaMalloc(&d_expr, sizeof(DeadlockExpression)), "Failed to allocate device memory for DeadlockExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(DeadlockExpression), cudaMemcpyHostToDevice),
                       "Failed to copy DeadlockExpression to device");

        delete host_copy;
    }
    // BoolExpression
    else if (auto *boolExpr = dynamic_cast<BoolExpression*>(exprHost)) {
        BoolExpression *host_copy = boolExpr->clone();

        checkCudaError(cudaMalloc(&d_expr, sizeof(BoolExpression)), "Failed to allocate device memory for BoolExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(BoolExpression), cudaMemcpyHostToDevice),
                       "Failed to copy BoolExpression to device");

        delete host_copy;
    }
    // AtomicProposition
    else if (auto *atomicExpr = dynamic_cast<AtomicProposition*>(exprHost)) {
        AtomicProposition *host_copy = atomicExpr->clone();

        ArithmeticExpression *leftDevice = allocateArithmeticExpression(&host_copy->getLeft());
        ArithmeticExpression *rightDevice = allocateArithmeticExpression(&host_copy->getRight());

        checkCudaError(cudaMalloc(&d_expr, sizeof(AtomicProposition)), "Failed to allocate device memory for AtomicProposition");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(AtomicProposition), cudaMemcpyHostToDevice),
                    "Failed to copy AtomicProposition to device");

        checkCudaError(cudaMemcpy(&static_cast<AtomicProposition*>(d_expr)->getLeft(), &leftDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to copy left pointer in AtomicProposition to device");
        checkCudaError(cudaMemcpy(&static_cast<AtomicProposition*>(d_expr)->getRight(), &rightDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to copy right pointer in AtomicProposition to device");

        delete host_copy;
    }
    // AndExpression
    else if (auto *andExpr = dynamic_cast<AndExpression*>(exprHost)) {
        AndExpression *host_copy = andExpr->clone();

        CudaExpression *leftDevice = allocateExpression(&host_copy->getLeft());
        CudaExpression *rightDevice = allocateExpression(&host_copy->getRight());

        checkCudaError(cudaMalloc(&d_expr, sizeof(AndExpression)), "Failed to allocate device memory for AndExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(AndExpression), cudaMemcpyHostToDevice),
                    "Failed to copy AndExpression to device");

        checkCudaError(cudaMemcpy(&static_cast<AndExpression*>(d_expr)->getLeft(), &leftDevice, sizeof(CudaExpression*), cudaMemcpyHostToDevice),
                    "Failed to copy left pointer in AndExpression to device");
        checkCudaError(cudaMemcpy(&static_cast<AndExpression*>(d_expr)->getRight(), &rightDevice, sizeof(CudaExpression*), cudaMemcpyHostToDevice),
                    "Failed to copy right pointer in AndExpression to device");

        delete host_copy;
    }
    // OrExpression
    else if (auto *orExpr = dynamic_cast<OrExpression*>(exprHost)) {
        OrExpression *host_copy = orExpr->clone();

        CudaExpression *leftDevice = allocateExpression(&host_copy->getLeft());
        CudaExpression *rightDevice = allocateExpression(&host_copy->getRight());

        checkCudaError(cudaMalloc(&d_expr, sizeof(OrExpression)), "Failed to allocate device memory for OrExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(OrExpression), cudaMemcpyHostToDevice),
                    "Failed to copy OrExpression to device");

        checkCudaError(cudaMemcpy(&static_cast<OrExpression*>(d_expr)->getLeft(), &leftDevice, sizeof(CudaExpression*), cudaMemcpyHostToDevice),
                    "Failed to copy left pointer in OrExpression to device");
        checkCudaError(cudaMemcpy(&static_cast<OrExpression*>(d_expr)->getRight(), &rightDevice, sizeof(CudaExpression*), cudaMemcpyHostToDevice),
                    "Failed to copy right pointer in OrExpression to device");

        delete host_copy;
    }
    else {
        //throw std::runtime_error("Unsupported CudaExpression type for allocation: '" + std::string(typeid(*exprHost).name()) + "'");
        //std::cout << "Unsupported CudaExpression type for allocation: '" << typeid(*exprHost).name() << "'" << std::endl;
    }
    
    return d_expr;
}

__host__ ArithmeticExpression* SMCQueryAllocator::allocateArithmeticExpression(ArithmeticExpression *exprHost) {
    //assert(exprHost != nullptr);

    ArithmeticExpression *d_expr = nullptr;

    // PlusExpression
    if (auto *plusExpr = dynamic_cast<PlusExpression*>(exprHost)) {
        PlusExpression *host_copy = static_cast<PlusExpression*>(plusExpr->clone());

        ArithmeticExpression *leftDevice = allocateArithmeticExpression(&host_copy->getLeft());
        ArithmeticExpression *rightDevice = allocateArithmeticExpression(&host_copy->getRight());

        checkCudaError(cudaMalloc(&d_expr, sizeof(PlusExpression)), "Failed to allocate device memory for PlusExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(PlusExpression), cudaMemcpyHostToDevice),
                    "Failed to copy PlusExpression to device");

        checkCudaError(cudaMemcpy(&(static_cast<PlusExpression*>(d_expr)->getLeft()), &leftDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to update left pointer in PlusExpression on device");
        checkCudaError(cudaMemcpy(&(static_cast<PlusExpression*>(d_expr)->getRight()), &rightDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to update right pointer in PlusExpression on device");

        delete host_copy;
    }
    // SubtractExpression
    else if (auto *subtractExpr = dynamic_cast<SubtractExpression*>(exprHost)) {
        SubtractExpression *host_copy = static_cast<SubtractExpression*>(subtractExpr->clone());

        ArithmeticExpression *leftDevice = allocateArithmeticExpression(&host_copy->getLeft());
        ArithmeticExpression *rightDevice = allocateArithmeticExpression(&host_copy->getRight());

        checkCudaError(cudaMalloc(&d_expr, sizeof(SubtractExpression)), "Failed to allocate device memory for SubtractExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(SubtractExpression), cudaMemcpyHostToDevice),
                    "Failed to copy SubtractExpression to device");

        checkCudaError(cudaMemcpy(&(static_cast<SubtractExpression*>(d_expr)->getLeft()), &leftDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to update left pointer in SubtractExpression on device");
        checkCudaError(cudaMemcpy(&(static_cast<SubtractExpression*>(d_expr)->getRight()), &rightDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to update right pointer in SubtractExpression on device");

        delete host_copy;
    }
    // MinusExpression
    else if (auto *minusExpr = dynamic_cast<MinusExpression*>(exprHost)) {
        MinusExpression *host_copy = static_cast<MinusExpression*>(minusExpr->clone());

        ArithmeticExpression *valueDevice = allocateArithmeticExpression(&host_copy->getValue());

        checkCudaError(cudaMalloc(&d_expr, sizeof(MinusExpression)), "Failed to allocate device memory for MinusExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(MinusExpression), cudaMemcpyHostToDevice),
                    "Failed to copy MinusExpression to device");

        checkCudaError(cudaMemcpy(&(static_cast<MinusExpression*>(d_expr)->getValue()), &valueDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to update value pointer in MinusExpression on device");

        delete host_copy;
    }
    // MultiplyExpression
    else if (auto *multiplyExpr = dynamic_cast<MultiplyExpression*>(exprHost)) {
        MultiplyExpression *host_copy = multiplyExpr->clone();

        ArithmeticExpression *leftDevice = allocateArithmeticExpression(&host_copy->getLeft());
        ArithmeticExpression *rightDevice = allocateArithmeticExpression(&host_copy->getRight());

        checkCudaError(cudaMalloc(&d_expr, sizeof(MultiplyExpression)), "Failed to allocate device memory for MultiplyExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(MultiplyExpression), cudaMemcpyHostToDevice),
                    "Failed to copy MultiplyExpression to device");

        // Use getters for left and right
        checkCudaError(cudaMemcpy(&(static_cast<MultiplyExpression*>(d_expr)->getLeft()), &leftDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to update left pointer in MultiplyExpression on device");
        checkCudaError(cudaMemcpy(&(static_cast<MultiplyExpression*>(d_expr)->getRight()), &rightDevice, sizeof(ArithmeticExpression*), cudaMemcpyHostToDevice),
                    "Failed to update right pointer in MultiplyExpression on device");

        delete host_copy;
    }
    // NumberExpression
    else if (auto *numberExpr = dynamic_cast<NumberExpression*>(exprHost)) {
        NumberExpression *host_copy = static_cast<NumberExpression*>(numberExpr->clone());

        checkCudaError(cudaMalloc(&d_expr, sizeof(NumberExpression)), "Failed to allocate device memory for NumberExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(NumberExpression), cudaMemcpyHostToDevice),
                    "Failed to copy NumberExpression to device");

        delete host_copy;
    }
    // IdentifierExpression
    else if (auto *identifierExpr = dynamic_cast<IdentifierExpression*>(exprHost)) {
        IdentifierExpression *host_copy = static_cast<IdentifierExpression*>(identifierExpr->clone());

        checkCudaError(cudaMalloc(&d_expr, sizeof(IdentifierExpression)), "Failed to allocate device memory for IdentifierExpression");
        checkCudaError(cudaMemcpy(d_expr, host_copy, sizeof(IdentifierExpression), cudaMemcpyHostToDevice),
                    "Failed to copy IdentifierExpression to device");

        delete host_copy;
    }
    else {
        //throw std::runtime_error("Unsupported ArithmeticExpression type for allocation: '" + std::string(typeid(*exprHost).name()) + "'");
        //std::cout << "Unsupported ArithmeticExpression type for allocation: '" << typeid(*exprHost).name() << "'" << std::endl;
    }
    
    return d_expr;
}

// Helper function to check for CUDA errors
inline void SMCQueryAllocator::checkCudaError(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        //std::cout << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        //throw std::runtime_error(msg);
    }
}

} // namespace VerifyTAPN::Alloc