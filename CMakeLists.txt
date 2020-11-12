cmake_minimum_required(VERSION 3.10.0)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set (CMAKE_CXX_STANDARD 11)

project(mi-jpeg)

add_subdirectory(cjpeg)
add_subdirectory(libjpeg)
