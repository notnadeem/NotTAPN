find_package(CUDAToolkit REQUIRED)

add_library(Alloc RunResultAllocator.cu)
