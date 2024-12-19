find_package(CUDAToolkit REQUIRED)

add_library(Cuda CudaProbabilityEstimation.cu CudaAST.cu CudaSMCQuery.cu)

target_compile_options(Cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        -O0
        --extended-lambda 
        --expt-relaxed-constexpr
        -Xptxas=-O0
        -Xptxas=-v
        --verbose
    >
)

# target_link_libraries(Cuda PRIVATE CUDA::cudart CUDA::cudadevrt)
# target_link_libraries(Cuda PRIVATE CUDA::cudart)

set_target_properties(Cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)