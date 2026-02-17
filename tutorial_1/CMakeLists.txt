cmake_minimum_required(VERSION 3.16)
project(tutorial_1_topdown LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O2 so the compiler generates realistic code but doesn't auto-vectorise
# away the performance differences we want to demonstrate.
# -g for debug symbols so ATP can map samples back to source lines.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(matmul_naive     src/matmul_naive.cpp)
add_executable(matmul_tiled_1d  src/matmul_tiled_1d.cpp)
add_executable(matmul_tiled_2d  src/matmul_tiled_2d.cpp)

# Disable auto-vectorisation on all targets so the progression reflects
# memory optimisations only (no hidden SIMD from the compiler).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(matmul_naive     PRIVATE -fno-tree-vectorize)
    target_compile_options(matmul_tiled_1d  PRIVATE -fno-tree-vectorize)
    target_compile_options(matmul_tiled_2d  PRIVATE -fno-tree-vectorize)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(matmul_naive     PRIVATE -fno-vectorize -fno-slp-vectorize)
    target_compile_options(matmul_tiled_1d  PRIVATE -fno-vectorize -fno-slp-vectorize)
    target_compile_options(matmul_tiled_2d  PRIVATE -fno-vectorize -fno-slp-vectorize)
endif()
