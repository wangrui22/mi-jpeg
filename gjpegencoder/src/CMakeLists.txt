#gdb支持
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -O3 -Wall")

include_directories(/usr/local/cuda/include)
include_directories(${PROJECT_SOURCE_DIR}/culib)
aux_source_directory(./ SRC)

add_executable(gjpegencoder ${SRC})

target_link_libraries(gjpegencoder culib)