cmake_minimum_required(VERSION 3.16)
project(tutorial_3_gpt2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Match the other tutorials: optimise and keep debug info for ATP.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(gpt2 src/gpt2.cpp)

# Enable OpenMP where available (e.g. GCC/Clang with libomp). On platforms
# that don't provide it (like default AppleClang), the build stays
# single-threaded without failing.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building gpt2 with OpenMP support")
    target_link_libraries(gpt2 PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found; building gpt2 without -fopenmp")
endif()

