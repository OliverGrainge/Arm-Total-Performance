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

# ── gpt2_kleidiai: same as gpt2 but uses KleidiAI microkernels ───────────────
set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
add_subdirectory(src/kleidiai EXCLUDE_FROM_ALL)

add_executable(gpt2_kleidiai src/gpt2_kleidiai.cpp)
target_include_directories(gpt2_kleidiai PRIVATE src/kleidiai)
target_link_libraries(gpt2_kleidiai PRIVATE kleidiai)
if(OpenMP_CXX_FOUND)
    message(STATUS "Building gpt2_kleidiai with OpenMP support")
    target_link_libraries(gpt2_kleidiai PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found; building gpt2_kleidiai without -fopenmp")
endif()

