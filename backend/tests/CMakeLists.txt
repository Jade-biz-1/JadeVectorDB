# CMakeLists.txt for unit tests
cmake_minimum_required(VERSION 3.20)

project(JadeVectorDB_Tests VERSION 1.0.0 LANGUAGES CXX)

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
find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/lib)
include_directories(${CMAKE_SOURCE_DIR}/src/models)
include_directories(${CMAKE_SOURCE_DIR}/src/services)
include_directories(${CMAKE_SOURCE_DIR}/src/api)

# Create test executable for vector storage service
add_executable(test_vector_storage_service
    unit/test_vector_storage_service.cpp
    ${CMAKE_SOURCE_DIR}/src/models/vector.cpp
    ${CMAKE_SOURCE_DIR}/src/models/database.cpp
    ${CMAKE_SOURCE_DIR}/src/models/index.cpp
    ${CMAKE_SOURCE_DIR}/src/models/embedding_model.cpp
    ${CMAKE_SOURCE_DIR}/src/services/vector_storage.cpp
    ${CMAKE_SOURCE_DIR}/src/services/database_layer.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/error_handling.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/config.cpp
)

# Link libraries
target_link_libraries(test_vector_storage_service
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Add test to CTest
add_test(NAME VectorStorageServiceTest COMMAND test_vector_storage_service)

# Create test executable for similarity search service
add_executable(test_similarity_search_service
    unit/test_similarity_search_service.cpp
    ${CMAKE_SOURCE_DIR}/src/models/vector.cpp
    ${CMAKE_SOURCE_DIR}/src/models/database.cpp
    ${CMAKE_SOURCE_DIR}/src/models/index.cpp
    ${CMAKE_SOURCE_DIR}/src/models/embedding_model.cpp
    ${CMAKE_SOURCE_DIR}/src/services/similarity_search.cpp
    ${CMAKE_SOURCE_DIR}/src/services/vector_storage.cpp
    ${CMAKE_SOURCE_DIR}/src/services/database_layer.cpp
    ${CMAKE_SOURCE_DIR}/src/services/search_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/services/metadata_filter.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/error_handling.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/config.cpp
)

# Link libraries
target_link_libraries(test_similarity_search_service
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Add test to CTest
add_test(NAME SimilaritySearchServiceTest COMMAND test_similarity_search_service)

# Create test executable for database service
add_executable(test_database_service
    unit/test_database_service.cpp
    ${CMAKE_SOURCE_DIR}/src/models/vector.cpp
    ${CMAKE_SOURCE_DIR}/src/models/database.cpp
    ${CMAKE_SOURCE_DIR}/src/models/index.cpp
    ${CMAKE_SOURCE_DIR}/src/models/embedding_model.cpp
    ${CMAKE_SOURCE_DIR}/src/services/database_service.cpp
    ${CMAKE_SOURCE_DIR}/src/services/database_layer.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/error_handling.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/logging.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/config.cpp
)

# Link libraries
target_link_libraries(test_database_service
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Add test to CTest
add_test(NAME DatabaseServiceTest COMMAND test_database_service)

# Create test executable for metadata filtering
add_executable(test_metadata_filtering
    unit/test_metadata_filtering.cpp
    ${CMAKE_SOURCE_DIR}/src/models/vector.cpp
    ${CMAKE_SOURCE_DIR}/src/models/database.cpp
    ${CMAKE_SOURCE_DIR}/src/models/index.cpp
    ${CMAKE_SOURCE_DIR}/src/models/embedding_model.cpp
    ${CMAKE_SOURCE_DIR}/src/services/metadata_filter.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/error_handling.cpp
    ${CMAKE_SOURCE_DIR}/src/lib/logging.cpp
)

# Link libraries
target_link_libraries(test_metadata_filtering
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

# Add test to CTest
add_test(NAME MetadataFilteringTest COMMAND test_metadata_filtering)