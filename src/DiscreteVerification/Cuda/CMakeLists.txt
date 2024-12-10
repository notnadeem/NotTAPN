find_package(CUDAToolkit REQUIRED)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

add_library(Cuda CudaProbabilityEstimation.cu CudaAST.cu CudaSMCQuery.cu)

target_compile_options(Cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -g
        -G
        --relocatable-device-code=true
    >
)

message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Runtime Location: ${CUDAToolkit_BIN_DIR}")

# target_link_libraries(Cuda PRIVATE CUDA::cudart CUDA::cudadevrt)
# target_link_libraries(Cuda PRIVATE CUDA::cudart)

set_target_properties(Cuda PROPERTIES 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)