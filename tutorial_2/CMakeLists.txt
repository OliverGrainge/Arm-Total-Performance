cmake_minimum_required(VERSION 3.16)
project(tutorial_2_memory_access LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O2: realistic optimisation level.
# -g:  debug symbols so ATP can map samples back to source lines.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(softmax_baseline  src/softmax_baseline.cpp)
add_executable(softmax_restrict  src/softmax_restrict.cpp)
add_executable(softmax_aligned   src/softmax_aligned.cpp)

# Disable auto-vectorisation on the baseline so ATP's Source Code Inspector
# shows the scalar access pattern that exists before aliasing annotations are
# added.  The restrict and aligned variants are compiled with default
# vectorisation enabled so the improvement is clearly visible.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(softmax_baseline PRIVATE -fno-tree-vectorize)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(softmax_baseline PRIVATE -fno-vectorize -fno-slp-vectorize)
endif()

# Link math library (required on some Linux toolchains for expf).
target_link_libraries(softmax_baseline PRIVATE m)
target_link_libraries(softmax_restrict PRIVATE m)
target_link_libraries(softmax_aligned  PRIVATE m)
