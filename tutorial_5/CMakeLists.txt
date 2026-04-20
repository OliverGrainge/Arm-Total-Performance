cmake_minimum_required(VERSION 3.16)
project(base64_porting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- platform detection ----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    set(IS_X86 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(IS_ARM TRUE)
endif()

# ---------- benchmark executable ----------
set(BENCH_SOURCES src/main.cpp src/base64_scalar.cpp)

if(IS_X86)
    list(APPEND BENCH_SOURCES src/base64_x86.cpp)
elseif(IS_ARM)
    list(APPEND BENCH_SOURCES src/base64_arm.cpp)
endif()

add_executable(base64_bench ${BENCH_SOURCES})
target_include_directories(base64_bench PRIVATE src)
target_compile_options(base64_bench PRIVATE -O2 -Wall)

if(IS_X86)
    target_compile_definitions(base64_bench PRIVATE USE_X86_SIMD)
    target_compile_options(base64_bench PRIVATE -mssse3)
elseif(IS_ARM)
    target_compile_definitions(base64_bench PRIVATE USE_ARM_SIMD)
endif()

# ---------- test executable ----------
set(TEST_SOURCES src/test_base64.cpp src/base64_scalar.cpp)

if(IS_X86)
    list(APPEND TEST_SOURCES src/base64_x86.cpp)
elseif(IS_ARM)
    list(APPEND TEST_SOURCES src/base64_arm.cpp)
endif()

add_executable(base64_test ${TEST_SOURCES})
target_include_directories(base64_test PRIVATE src)
target_compile_options(base64_test PRIVATE -O2 -Wall)

if(IS_X86)
    target_compile_definitions(base64_test PRIVATE USE_X86_SIMD)
    target_compile_options(base64_test PRIVATE -mssse3)
elseif(IS_ARM)
    target_compile_definitions(base64_test PRIVATE USE_ARM_SIMD)
endif()
