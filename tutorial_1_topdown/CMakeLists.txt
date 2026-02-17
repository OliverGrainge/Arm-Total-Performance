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
add_executable(matmul_tiled_neon src/matmul_tiled_neon.cpp)

# Keep naive/reordered/tiled mostly scalar so progression reflects memory
# optimisations first, then explicit NEON in the final step.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(matmul_naive PRIVATE -fno-tree-vectorize)
    target_compile_options(matmul_reordered PRIVATE -fno-tree-vectorize)
    target_compile_options(matmul_tiled PRIVATE -O3 -fno-tree-vectorize)
    target_compile_options(matmul_tiled_neon PRIVATE -O3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(matmul_naive PRIVATE -fno-vectorize -fno-slp-vectorize)
    target_compile_options(matmul_reordered PRIVATE -fno-vectorize -fno-slp-vectorize)
    target_compile_options(matmul_tiled PRIVATE -O3 -fno-vectorize -fno-slp-vectorize)
    target_compile_options(matmul_tiled_neon PRIVATE -O3)
endif()
