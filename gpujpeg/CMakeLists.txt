project(gpujpegtest LANGUAGES CUDA CXX)

#enable_language(CUDA)

include_directories(/usr/local/include)
include_directories(/usr/local/cuda/include)

link_directories(/usr/local/cuda/lib64)

add_executable(gpujpegtest mi_main.cpp)

target_link_libraries(gpujpegtest gpujpeg cudart)
