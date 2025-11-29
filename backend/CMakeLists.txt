cmake_minimum_required(VERSION 3.20)

project(JadeVectorDB VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_WITH_ARROW "Build with Apache Arrow support" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -std=c++20")
    # Coverage options - only when BUILD_COVERAGE is enabled
    option(BUILD_COVERAGE "Build with gcov coverage" OFF)
    if(BUILD_COVERAGE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -g")
    endif()
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /std:c++20")
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

# Use FetchContent to include all required libraries directly from source
include(FetchContent)

# Set FetchContent base directory for better organization
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps)

#############################################################################
# CORE DEPENDENCIES - All fetched from source for reproducible builds
#############################################################################

# Eigen - Linear algebra library
message(STATUS "Fetching Eigen...")
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

# FlatBuffers - Serialization library
message(STATUS "Fetching FlatBuffers...")
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v23.5.26
    GIT_SHALLOW TRUE
)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flatbuffers)

# nlohmann/json - JSON library for C++
message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Crow - Web framework for REST API
message(STATUS "Fetching Crow...")
FetchContent_Declare(
    crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG v1.0+5
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(crow)

# Google Test - Testing framework
message(STATUS "Fetching Google Test...")
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark - Performance testing
message(STATUS "Fetching Google Benchmark...")
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
    GIT_SHALLOW TRUE
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Apache Arrow - In-memory columnar data format
# Note: Arrow is large and complex, we'll make it conditional with minimal options
# Temporarily disabled due to CMake issues
# if(BUILD_WITH_ARROW)
#     message(STATUS "Fetching Apache Arrow...")
#     # Fetch a version that is known to work with modern CMake
#     FetchContent_Declare(
#         arrow
#         GIT_REPOSITORY https://github.com/apache/arrow.git
#         GIT_TAG apache-arrow-10.0.1  # Using a stable version that works with CMake
#         GIT_SHALLOW TRUE
#         SOURCE_SUBDIR cpp
#     )
#     # Minimize Arrow build - only build what we need
#     set(ARROW_BUILD_STATIC ON CACHE BOOL "" FORCE)
#     set(ARROW_BUILD_SHARED OFF CACHE BOOL "" FORCE)
#     set(ARROW_DEPENDENCY_SOURCE "BUNDLED" CACHE STRING "" FORCE)
#     set(ARROW_VERBOSE_THIRDPARTY_BUILD OFF CACHE BOOL "" FORCE)
#     set(ARROW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#     set(ARROW_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
#     set(ARROW_IPC OFF CACHE BOOL "" FORCE)
#     set(ARROW_FLIGHT OFF CACHE BOOL "" FORCE)
#     set(ARROW_COMPUTE OFF CACHE BOOL "" FORCE)
#     set(ARROW_DATASET OFF CACHE BOOL "" FORCE)
#     set(ARROW_FILESYSTEM OFF CACHE BOOL "" FORCE)
#     set(ARROW_HDFS OFF CACHE BOOL "" FORCE)
#     set(ARROW_JSON OFF CACHE BOOL "" FORCE)
#     set(ARROW_PARQUET OFF CACHE BOOL "" FORCE)
#     set(ARROW_WITH_BROTLI OFF CACHE BOOL "" FORCE)
#     set(ARROW_WITH_BZ2 OFF CACHE BOOL "" FORCE)
#     set(ARROW_WITH_LZ4 OFF CACHE BOOL "" FORCE)
#     set(ARROW_WITH_SNAPPY OFF CACHE BOOL "" FORCE)
#     set(ARROW_WITH_ZLIB OFF CACHE BOOL "" FORCE)
#     set(ARROW_WITH_ZSTD OFF CACHE BOOL "" FORCE)

#     # Only fetch Arrow if not already available
#     FetchContent_GetProperties(arrow)
#     if(NOT arrow_POPULATED)
#         FetchContent_Populate(arrow)
#         add_subdirectory(${arrow_SOURCE_DIR}/cpp ${arrow_BINARY_DIR})
#     endif()
# endif()

# gRPC - RPC framework
# Note: gRPC is very large, we'll make it optional with stubs as fallback
message(STATUS "Fetching gRPC (this may take a while)...")
option(BUILD_WITH_GRPC "Build with full gRPC support" OFF)

if(BUILD_WITH_GRPC)
    FetchContent_Declare(
        grpc
        GIT_REPOSITORY https://github.com/grpc/grpc.git
        GIT_TAG v1.60.0
        GIT_SHALLOW TRUE
    )
    set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_CSHARP_EXT OFF CACHE BOOL "" FORCE)
    set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(grpc)
    set(GRPC_AVAILABLE TRUE)
else()
    message(STATUS "Building with gRPC stubs (use -DBUILD_WITH_GRPC=ON for full gRPC)")
    set(GRPC_AVAILABLE FALSE)
endif()

#############################################################################
# INCLUDE DIRECTORIES
#############################################################################

include_directories(${eigen_SOURCE_DIR})
include_directories(${flatbuffers_SOURCE_DIR}/include)
include_directories(${nlohmann_json_SOURCE_DIR}/include)
include_directories(${crow_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/models)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/services)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/api)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# Arrow includes
if(TARGET arrow_static)
    include_directories(${arrow_SOURCE_DIR}/cpp/src)
    include_directories(${arrow_BINARY_DIR}/cpp/src)
endif()

#############################################################################
# SOURCE FILES
#############################################################################

set(CORE_SOURCES
    src/models/vector.cpp
    src/models/database.cpp
    src/models/index.cpp
    src/models/embedding_model.cpp
    src/services/vector_storage.cpp
    src/services/similarity_search.cpp
    src/services/database_service.cpp
    src/services/database_layer.cpp
    src/services/metadata_filter.cpp
    src/services/index_service.cpp
    src/services/cluster_service.cpp
    src/services/sharding_service.cpp
    src/services/replication_service.cpp
    src/services/lifecycle_service.cpp
    src/services/query_router.cpp
    src/services/distributed_service_manager.cpp
    src/services/performance_benchmark.cpp
    src/services/security_audit_logger.cpp
    src/services/archival_service.cpp
    src/services/monitoring_service.cpp
    src/services/authentication_service.cpp
    src/services/index/pq_index.cpp
    src/services/index/opq_index.cpp
    src/services/index/sq_index.cpp
    src/services/index/flat_index.cpp
    src/services/index/hnsw_index.cpp
    src/services/index/lsh_index.cpp
    src/services/index/ivf_index.cpp
    src/services/index/composite_index.cpp
    src/lib/mmap_utils.cpp
    src/lib/simd_ops.cpp
    src/lib/serialization.cpp
    src/lib/storage_format.cpp
    src/lib/logging.cpp
    src/lib/error_handling.cpp
    src/lib/config.cpp
    src/lib/thread_pool.cpp
    src/lib/auth.cpp
    src/lib/zero_trust.cpp
    src/lib/metrics.cpp
    src/lib/compression.cpp
    src/lib/encryption.cpp
    src/lib/field_encryption_service.cpp
    src/lib/gpu_acceleration.cpp
    src/lib/vector_operations.cpp
)

set(API_SOURCES
    src/api/rest/rest_api.cpp
    src/api/rest/rest_api_auth_handlers.cpp
    src/api/rest/rest_api_user_handlers.cpp
    src/api/rest/rest_api_apikey_handlers.cpp
    src/api/rest/rest_api_security_handlers.cpp
    src/api/grpc/grpc_service_stub.cpp
)

#############################################################################
# BUILD TARGETS
#############################################################################

# Core library - contains all business logic
add_library(jadevectordb_core STATIC ${CORE_SOURCES})

target_link_libraries(jadevectordb_core PUBLIC
    Threads::Threads
    OpenSSL::SSL
    OpenSSL::Crypto
    ZLIB::ZLIB
    $<$<BOOL:${GRPC_AVAILABLE}>:grpc++>
    $<$<BOOL:${GRPC_AVAILABLE}>:grpc>
    $<$<BOOL:${GRPC_AVAILABLE}>:gpr>
    $<$<BOOL:${BUILD_WITH_ARROW}>:arrow_static>
    ${CMAKE_DL_LIBS}
)

# Main executable
add_executable(jadevectordb
    src/main.cpp
    ${API_SOURCES}
)

target_link_libraries(jadevectordb PRIVATE
    jadevectordb_core
    Threads::Threads
    $<$<BOOL:${GRPC_AVAILABLE}>:grpc++>
    $<$<BOOL:${GRPC_AVAILABLE}>:grpc>
    $<$<BOOL:${GRPC_AVAILABLE}>:gpr>
    $<$<BOOL:${BUILD_WITH_ARROW}>:arrow_static>
    ${CMAKE_DL_LIBS}
)

#############################################################################
# TESTING
#############################################################################

option(BUILD_TESTS "Build test suite" ON)

if(BUILD_TESTS)
    enable_testing()

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
        tests/test_advanced_filtering.cpp
        tests/test_advanced_filtering_integration.cpp
        tests/test_advanced_embedding_unit.cpp
        tests/test_advanced_embedding_integration.cpp
        tests/test_search_serialization.cpp
    )

    target_link_libraries(jadevectordb_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
        jadevectordb_core
    )

    add_test(NAME VectorStorageTest COMMAND jadevectordb_tests)
endif()

#############################################################################
# BENCHMARKING
#############################################################################

option(BUILD_BENCHMARKS "Build benchmark suite" ON)

if(BUILD_BENCHMARKS)
    add_executable(search_benchmarks
        tests/benchmarks/search_benchmarks.cpp
        tests/benchmarks/advanced_filtering_benchmarks.cpp
        tests/benchmarks/advanced_indexing_benchmarks.cpp
    )

    target_link_libraries(search_benchmarks PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
        jadevectordb_core
    )

    add_executable(filtered_search_benchmarks
        tests/benchmarks/filtered_search_benchmarks.cpp
    )

    target_link_libraries(filtered_search_benchmarks PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
        jadevectordb_core
    )
endif()

#############################################################################
# COVERAGE
#############################################################################

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

#############################################################################
# INSTALLATION
#############################################################################

install(TARGETS jadevectordb
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS jadevectordb_core
    ARCHIVE DESTINATION lib
)

#############################################################################
# BUILD INFORMATION
#############################################################################

message(STATUS "")
message(STATUS "JadeVectorDB Build Configuration:")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Build coverage: ${BUILD_COVERAGE}")
message(STATUS "  Build with gRPC: ${BUILD_WITH_GRPC}")
message(STATUS "")
