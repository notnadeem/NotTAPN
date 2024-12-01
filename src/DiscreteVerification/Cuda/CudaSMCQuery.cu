#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"

namespace VerifyTAPN::Cuda::AST {

CudaSMCQuery *CudaSMCQuery::clone() const { return new CudaSMCQuery(*this); }

void CudaSMCQuery::accept(CudaVisitor &visitor, Result &context) { visitor.visit(*this, context); }

} // namespace VerifyTAPN::Cuda::AST
