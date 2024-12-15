find_package(CUDAToolkit REQUIRED)

add_library(Cuda CudaProbabilityEstimation.cu CudaAST.cu CudaSMCQuery.cu)

target_compile_options(Cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
    >
)

set_target_properties(Cuda PROPERTIES 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)