cmake_minimum_required(VERSION 3.16)
project(tutorial_2_memory_access LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -O2: optimise both binaries identically.
# -g:  retain debug info so ATP source view can resolve line numbers.
# The only variable between the two binaries is data layout, not compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(aos_baseline  src/aos_baseline.cpp)
add_executable(soa_optimized src/soa_optimized.cpp)
