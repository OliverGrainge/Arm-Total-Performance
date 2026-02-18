cmake_minimum_required(VERSION 3.16)
project(tutorial_2_memory_access LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O2: realistic optimisation level.
# -g: debug symbols so ATP can map samples back to source lines.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(triad_baseline src/triad_baseline.cpp)
add_executable(triad_restrict src/triad_restrict.cpp)
add_executable(triad_aligned  src/triad_aligned.cpp)

# Keep baseline scalar so Source Code Inspector shows the before-state.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(triad_baseline PRIVATE -fno-tree-vectorize)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(triad_baseline PRIVATE -fno-vectorize -fno-slp-vectorize)
endif()
