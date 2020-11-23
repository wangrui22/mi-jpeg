cmake_minimum_required(VERSION 3.10.0)

project(gjpegencoder LANGUAGES CXX CUDA)

add_subdirectory(./culib)
add_subdirectory(./src)