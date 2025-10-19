# CMakeLists.txt for property-based tests
cmake_minimum_required(VERSION 3.20)

project(JadeVectorDB_PropertyTests VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O2")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../backend/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../backend/src/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../backend/src/models)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../backend/src/services)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create property-based test executable
add_executable(property_tests
    test_property_suite.cpp
    vector-space/test_vector_space_properties.cpp
    consistency/test_consistency_properties.cpp
    concurrency/test_concurrency_properties.cpp
    distributed-system/test_distributed_invariants.cpp
)

# Link libraries
target_link_libraries(property_tests
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Add to CTest
add_test(NAME PropertyBasedTests COMMAND property_tests)

# Define a custom target to run property tests specifically
add_custom_target(run_property_tests
    COMMAND property_tests
    DEPENDS property_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running property-based tests"
)