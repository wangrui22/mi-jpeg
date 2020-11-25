# project(culib LANGUAGES CUDA CXX)
# set (CMAKE_CXX_STANDARD 11)

aux_source_directory(./ CULIB_SRC)

#gdb支持
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -O3 -Wall")

include_directories(/usr/local/cuda/include)

#CUDA支持1：find package
#find_package(CUDA REQUIRED)

#CUDA支持2：添加NVCC的flag参数，适配不通版本的cuda以及显卡架构
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=sm_61;${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=compute_61;${CUDA_NVCC_FLAGS}")

#CUDA支持3：.cu .cuh kernel文件的编译，并生成so，注意是 cuda_add_library 而不是 add_library
#cuda_add_library(culibKernel SHARED kernel_smooth.cu)

#使用cuda kernel的 cpp文件以及库，按照普通的，直接链接就行了
#add_library(culib SHARED smooth.h smooth.cpp)
#target_link_libraries(culib culibKernel)

#注意，这里不需要特意的link cuda的so（如cudart）,确可以work，这点和demo56,的 QT Boost不同

#根据官网的资料[FindCUDA](https://cmake.org/cmake/help/latest/module/FindCUDA.html)
#可以直接用project(demo7 LANGUAGES CXX CUDA) 或者 enable_language(CUDA)就可以cpp和cu混编译
#即 注释掉find_package(CUDA REQUIRED)
#即 add_library(culib SHARED kernel_smooth.cu smooth.h smooth.cpp)
#但是这样编译的时候会报错：～/culib/smooth.cpp:1:10: fatal error: cuda_runtime.h: No such file or directory #include <cuda_runtime.h>
#暂时原因不明

add_library(culib SHARED kernel_gpu_jpeg_encoder.cu mi_gpu_jpeg_define.h mi_gpu_jpeg_encoder.h mi_gpu_jpeg_encoder.cpp)