#ifndef SMCQUERYALLOCATOR_CUH_
#define SMCQUERYALLOCATOR_CUH_

#include "DiscreteVerification/Cuda/CudaSMCQuery.cuh"
#include <cuda_runtime.h>

namespace VerifyTAPN::Alloc {

using namespace Cuda::AST;

struct SMCQueryAllocator {
  __host__ static CudaSMCQuery* allocate(CudaSMCQuery *smcQueryHost);

private:
  __host__ static void allocatePointerMembers(CudaSMCQuery *smcQueryHost, CudaSMCQuery *d_smcQuery);
  __host__ static CudaExpression* allocateExpression(CudaExpression *exprHost);
  __host__ static ArithmeticExpression* allocateArithmeticExpression(ArithmeticExpression *exprHost);
  __host__ static void checkCudaError(cudaError_t result, const char *msg);
};
} // namespace VerifyTAPN::Alloc

#endif /* SMCQUERYALLOCATOR_CUH_ */