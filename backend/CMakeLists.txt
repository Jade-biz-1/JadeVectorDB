cmake_minimum_required(VERSION 3.20)

project(JadeVectorDB VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")
    # Coverage options - only when BUILD_COVERAGE is enabled
    option(BUILD_COVERAGE "Build with gcov coverage" OFF)
    if(BUILD_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -g")
    endif()
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

# FlatBuffers
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v23.5.26
)
FetchContent_MakeAvailable(flatbuffers)

# Crow web framework
FetchContent_Declare(
    crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG v1.0+5
)
FetchContent_MakeAvailable(crow)

# Add the Crow include directory explicitly
include_directories(${crow_SOURCE_DIR}/include)

# Apache Arrow
find_package(Arrow REQUIRED)
if(NOT Arrow_FOUND)
    message(FATAL_ERROR "Apache Arrow not found. Please install Apache Arrow.")
endif()

# Check if Arrow::arrow target exists, if not try alternative target names
if(TARGET Arrow::arrow)
    # Standard installation
    set(ARROW_LIB_NAME Arrow::arrow)
else()
    # Check if it's built from source with different target name
    if(TARGET arrow_shared)
        set(ARROW_LIB_NAME arrow_shared)
    elseif(TARGET arrow_static)
        set(ARROW_LIB_NAME arrow_static)
    else()
        # Check for libarrow directly
        find_library(ARROW_LIB arrow PATHS /usr/local/lib /opt/local/lib)
        if(ARROW_LIB)
            set(ARROW_LIB_NAME ${ARROW_LIB})
        else()
            message(FATAL_ERROR "Arrow target not found. Please check Arrow installation.")
        endif()
    endif()
endif()

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
include_directories(${flatbuffers_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/models)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/services)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib)

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
    ${ARROW_LIB_NAME}
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
    ${ARROW_LIB_NAME}
    ${CMAKE_DL_LIBS}
)

# Add tests executable
add_executable(jadevectordb_tests
    tests/test_vector_storage.cpp
    tests/test_similarity_search.cpp
    tests/test_database_service.cpp
    tests/test_vector_storage_service.cpp
    tests/test_vector_api_integration.cpp
    tests/test_similarity_search_unit.cpp
    tests/test_search_api_integration.cpp
    tests/test_metadata_filter_unit.cpp
    tests/test_advanced_search_integration.cpp
    tests/test_database_service_unit.cpp
    tests/test_database_api_integration.cpp
    tests/test_e2e_filtered_search.cpp
    tests/test_search_quality.cpp
    tests/test_integration_comprehensive.cpp
    tests/test_core_services_comprehensive.cpp
    tests/test_service_interactions.cpp
)

target_link_libraries(jadevectordb_tests 
    GTest::gtest 
    GTest::gtest_main 
    jadevectordb_core
)

enable_testing()
add_test(NAME VectorStorageTest COMMAND jadevectordb_tests)

# Add Google Benchmark executable for performance testing
find_package(benchmark REQUIRED)
if(NOT benchmark_FOUND)
    message(FATAL_ERROR "Google Benchmark not found. Please install Google Benchmark.")
endif()

add_executable(search_benchmarks
    tests/benchmarks/search_benchmarks.cpp
    src/services/vector_storage.cpp
    src/services/similarity_search.cpp
    src/services/database_service.cpp
    src/services/database_layer.cpp
    src/models/vector.cpp
    src/models/database.cpp
    src/models/index.cpp
    src/models/embedding_model.cpp
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

target_link_libraries(search_benchmarks
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
    ${ARROW_LIB_NAME}
    ${CMAKE_DL_LIBS}
)

add_executable(filtered_search_benchmarks
    tests/benchmarks/filtered_search_benchmarks.cpp
    src/services/vector_storage.cpp
    src/services/similarity_search.cpp
    src/services/database_service.cpp
    src/services/database_layer.cpp
    src/services/metadata_filter.cpp
    src/models/vector.cpp
    src/models/database.cpp
    src/models/index.cpp
    src/models/embedding_model.cpp
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

target_link_libraries(filtered_search_benchmarks
    benchmark::benchmark
    benchmark::benchmark_main
    Threads::Threads
    ${ARROW_LIB_NAME}
    ${CMAKE_DL_LIBS}
)

# Coverage target (only when BUILD_COVERAGE is enabled)
if(BUILD_COVERAGE)
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)

    if(LCOV_PATH AND GENHTML_PATH)
        add_custom_target(coverage
            # Capture coverage data
            COMMAND ${LCOV_PATH} --directory . --zerocounters
            COMMAND ${CMAKE_CTEST_COMMAND} --verbose
            COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
            # Remove system and test files from coverage
            COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' '*/tests/*' '*/test_*' '*_test*' --output-file coverage.info
            # Generate HTML report
            COMMAND ${GENHTML_PATH} coverage.info --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report..."
        )
        message(STATUS "Coverage target is available. Build with -DBUILD_COVERAGE=ON to enable it.")
    else()
        message(WARNING "lcov or genhtml not found. Coverage target will not be available.")
    endif()
endif()