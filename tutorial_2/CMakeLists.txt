cmake_minimum_required(VERSION 3.16)
project(tutorial_2_memory_access LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(triad_baseline src/triad_baseline.cpp)
add_executable(triad_restrict src/triad_restrict.cpp)

target_compile_options(triad_baseline PRIVATE -fno-tree-vectorize)
target_compile_options(triad_restrict PRIVATE -ftree-vectorize)
