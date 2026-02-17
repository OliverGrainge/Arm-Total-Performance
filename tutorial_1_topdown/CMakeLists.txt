cmake_minimum_required(VERSION 3.16)
project(tutorial_1_topdown LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O2 so the compiler generates realistic code but doesn't auto-vectorise
# away the performance differences we want to demonstrate.
# -g for debug symbols so ATP can map samples back to source lines.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(matmul_naive   src/matmul_naive.cpp)
add_executable(matmul_reordered src/matmul_reordered.cpp)
add_executable(matmul_tiled   src/matmul_tiled.cpp)
