#ifndef VERIFYTAPN_ATLER_CUDAPAIR_CUH_
#define VERIFYTAPN_ATLER_CUDAPAIR_CUH_

#include <cuda_runtime.h>
#include <utility> // for std::pair

namespace VerifyTAPN::Cuda {
    template<typename T1, typename T2>
    struct CudaPair {
        T1 first;
        T2 second;
        
        __host__ __device__ CudaPair() : first(), second() {}

        __host__ __device__ CudaPair(T1 f, T2 s) : first(f), second(s) {}
        
        // Add conversion operator to std::pair
        __host__ operator std::pair<T1, T2>() const {
            return std::pair<T1, T2>(first, second);
        }
    };

    template<typename T1, typename T2>
    __host__ __device__ CudaPair<T1, T2> makeCudaPair(T1 t1, T2 t2) {
        return CudaPair<T1, T2>(t1, t2);
    }
}

#endif