#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"

namespace VerifyTAPN::Cuda::AST {

__host__ __device__ CudaSMCQuery *CudaSMCQuery::clone() const { return new CudaSMCQuery(*this); }

__device__ void CudaSMCQuery::accept(CudaVisitor &visitor, Result &context) { visitor.visit(*this, context); }

} // namespace VerifyTAPN::Cuda::AST
