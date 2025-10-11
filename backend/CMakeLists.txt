cmake_minimum_required(VERSION 3.20)

project(JadeVectorDB VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# Find required packages
find_package(Threads REQUIRED)

# Use FetchContent to include required libraries directly
include(FetchContent)

# Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# gRPC
find_package(gRPC CONFIG REQUIRED)
if(NOT gRPC_FOUND)
    message(FATAL_ERROR "gRPC not found. Please install gRPC.")
endif()

# Google Test for unit tests
find_package(GTest REQUIRED)
if(NOT GTest_FOUND)
    message(FATAL_ERROR "GTest not found. Please install GTest.")
endif()

# Include directories
include_directories(${eigen_SOURCE_DIR})

# For FlatBuffers, Apache Arrow and other dependencies, we'll use package managers or find modules
# These may need to be installed separately or via vcpkg/conan

# Define backend executable
add_executable(jadevectordb
    src/main.cpp
    src/models/vector.cpp
    src/models/database.cpp
    src/models/index.cpp
    src/models/embedding_model.cpp
    src/services/vector_storage.cpp
    src/services/similarity_search.cpp
    src/services/database_service.cpp
    src/services/index_service.cpp
    src/api/rest/rest_api.cpp
    src/lib/mmap_utils.cpp
    src/lib/simd_ops.cpp
    src/lib/serialization.cpp
    src/lib/storage_format.cpp
    src/lib/logging.cpp
    src/lib/error_handling.cpp
    src/lib/config.cpp
    src/lib/thread_pool.cpp
    src/lib/auth.cpp
    src/lib/metrics.cpp
)

# Link libraries
target_link_libraries(jadevectordb 
    Threads::Threads
    gRPC::gpr
    gRPC::grpc
    gRPC::grpc++
    ${CMAKE_DL_LIBS}
)

# Create a library target for the core functionality
add_library(jadevectordb_core
    src/models/vector.cpp
    src/models/database.cpp
    src/models/index.cpp
    src/models/embedding_model.cpp
    src/services/vector_storage.cpp
    src/services/similarity_search.cpp
    src/services/database_service.cpp
    src/services/index_service.cpp
    src/lib/mmap_utils.cpp
    src/lib/simd_ops.cpp
    src/lib/serialization.cpp
    src/lib/storage_format.cpp
    src/lib/logging.cpp
    src/lib/error_handling.cpp
    src/lib/config.cpp
    src/lib/thread_pool.cpp
    src/lib/auth.cpp
    src/lib/metrics.cpp
)

target_link_libraries(jadevectordb_core 
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Add tests executable
add_executable(jadevectordb_tests
    tests/test_vector_storage.cpp
    tests/test_similarity_search.cpp
    tests/test_database_service.cpp
)

target_link_libraries(jadevectordb_tests 
    GTest::gtest 
    GTest::gtest_main 
    jadevectordb_core
)

enable_testing()
add_test(NAME VectorStorageTest COMMAND jadevectordb_tests)