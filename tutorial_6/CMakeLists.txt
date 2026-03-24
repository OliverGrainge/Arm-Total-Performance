cmake_minimum_required(VERSION 3.16)
project(tutorial_6_clip_image_search LANGUAGES CXX)

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
# scalar — this ensures a clear Instruction Mix contrast against the NEON version.
add_executable(image_search_baseline  src/image_search_baseline.cpp)
target_compile_options(image_search_baseline PRIVATE -fno-tree-vectorize)
target_link_libraries(image_search_baseline pthread)

# NEON: hand-written NEON distance function, same over-provisioned parameters.
add_executable(image_search_neon src/image_search_neon.cpp)
target_link_libraries(image_search_neon pthread)

# Tuned: NEON distance + production-grade HNSW parameters.
add_executable(image_search_tuned src/image_search_tuned.cpp)
target_link_libraries(image_search_tuned pthread)

# Fully optimised: NEON distance + tuned parameters + sorted queries.
add_executable(image_search_optimised src/image_search_optimised.cpp)
target_link_libraries(image_search_optimised pthread)
