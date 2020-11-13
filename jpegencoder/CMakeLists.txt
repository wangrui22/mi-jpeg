cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

project(jpegencoder)

aux_source_directory(./ SRC )
add_executable(jpegencoder ${SRC})