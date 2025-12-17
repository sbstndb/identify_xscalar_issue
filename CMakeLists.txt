cmake_minimum_required(VERSION 3.15)
project(identify_xscalar_issue CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Option for xsimd
option(USE_XSIMD "Enable xsimd support" ON)

# Find dependencies
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

if(USE_XSIMD)
    find_package(xsimd REQUIRED)
endif()

# Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Benchmark executable
add_executable(bench_xscalar bench_xscalar.cpp)

target_link_libraries(bench_xscalar
    PRIVATE
    xtensor
    benchmark::benchmark
    benchmark::benchmark_main
)

if(USE_XSIMD)
    target_link_libraries(bench_xscalar PRIVATE xsimd)
    target_compile_definitions(bench_xscalar PRIVATE XTENSOR_USE_XSIMD)
    message(STATUS "XSIMD: ENABLED")
else()
    message(STATUS "XSIMD: DISABLED")
endif()

# Print configuration
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
