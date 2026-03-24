cmake_minimum_required(VERSION 3.16)
project(tutorial_6_hnswlib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O2 so all examples build with the same optimization level.
# -g for debug symbols so ATP can map samples back to source lines.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

# HNSWlib is header-only — just add the include path.
include_directories(src/hnswlib)

# Also add the data/ directory so generated data_config.h is found.
include_directories(data)

# Baseline: disable auto-vectorisation so the scalar L2 distance remains
# scalar — this ensures a clear Instruction Mix contrast against Pass 1.
add_executable(search_baseline  src/search_baseline.cpp)
target_compile_options(search_baseline PRIVATE -fno-tree-vectorize)
target_link_libraries(search_baseline pthread)

# Pass 1: NEON distance function (explicit intrinsics, auto-vec still off
# for the rest of the code so the NEON improvement is clearly attributable).
add_executable(search_pass1 src/search_pass1.cpp)
target_link_libraries(search_pass1 pthread)

# Pass 2: tuned search parameters (same NEON distance as Pass 1).
add_executable(search_pass2 src/search_pass2.cpp)
target_link_libraries(search_pass2 pthread)

# Fully optimised: NEON distance + tuned parameters + sorted queries.
add_executable(search_optimised src/search_optimised.cpp)
target_link_libraries(search_optimised pthread)
