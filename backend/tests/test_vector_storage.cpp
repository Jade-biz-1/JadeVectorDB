#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>

#include "services/database_service.h"
#include "services/vector_storage.h"
#include "services/similarity_search.h"
#include "models/vector.h"
#include "models/database.h"

namespace jadevectordb {

// Test fixture for database service tests
class DatabaseServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        db_service_ = std::make_unique<DatabaseService>();
        db_service_->initialize();
    }

    void TearDown() override {
        db_service_.reset();
    }

    std::unique_ptr<DatabaseService> db_service_;
};

// Test fixture for vector storage service tests
class VectorStorageServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        vector_storage_ = std::make_unique<VectorStorageService>();
        vector_storage_->initialize();
        
        // Create a test database for vector operations
        Database db;
        db.name = "test_db";
        db.description = "Test database for vector operations";
        db.vectorDimension = 3;
        
        auto result = vector_storage_->get_db_layer_for_testing()->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            vector_storage_->get_db_layer_for_testing()->delete_database(test_database_id_);
        }
        vector_storage_.reset();
    }

    std::unique_ptr<VectorStorageService> vector_storage_;
    std::string test_database_id_;
};

// Test fixture for similarity search service tests
class SimilaritySearchServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        search_service_ = std::make_unique<SimilaritySearchService>();
        search_service_->initialize();
        
        // Create a test database for search operations
        Database db;
        db.name = "test_search_db";
        db.description = "Test database for search operations";
        db.vectorDimension = 3;
        
        auto result = search_service_->get_vector_storage_for_testing()->get_db_layer_for_testing()->create_database(db);
        ASSERT_TRUE(result.has_value());
        test_database_id_ = result.value();

        // Add some test vectors
        Vector v1;
        v1.id = "vector_1";
        v1.values = {0.1f, 0.2f, 0.3f};

        Vector v2;
        v2.id = "vector_2";
        v2.values = {0.4f, 0.5f, 0.6f};

        Vector v3;
        v3.id = "vector_3";
        v3.values = {0.7f, 0.8f, 0.9f};

        search_service_->get_vector_storage_for_testing()->store_vector(test_database_id_, v1);
        search_service_->get_vector_storage_for_testing()->store_vector(test_database_id_, v2);
        search_service_->get_vector_storage_for_testing()->store_vector(test_database_id_, v3);
    }

    void TearDown() override {
        // Clean up test database
        if (!test_database_id_.empty()) {
            search_service_->get_vector_storage_for_testing()->get_db_layer_for_testing()->delete_database(test_database_id_);
        }
        search_service_.reset();
    }

    std::unique_ptr<SimilaritySearchService> search_service_;
    std::string test_database_id_;
};

// Database Service Tests
TEST_F(DatabaseServiceTest, CreateDatabase) {
    DatabaseCreationParams params;
    params.name = "test_db";
    params.description = "A test database";
    params.vectorDimension = 128;

    auto result = db_service_->create_database(params);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result.value().empty());  // Should return a valid database ID
}

TEST_F(DatabaseServiceTest, GetDatabase) {
    // Create a database first
    DatabaseCreationParams params;
    params.name = "get_test_db";
    params.description = "A test database for get operation";
    params.vectorDimension = 64;

    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = create_result.value();

    // Get the database
    auto get_result = db_service_->get_database(db_id);
    EXPECT_TRUE(get_result.has_value());
    EXPECT_EQ(get_result.value().name, "get_test_db");
    EXPECT_EQ(get_result.value().vectorDimension, 64);
}

TEST_F(DatabaseServiceTest, ListDatabases) {
    // Create multiple databases
    DatabaseCreationParams params1;
    params1.name = "list_test_db1";
    params1.vectorDimension = 32;

    DatabaseCreationParams params2;
    params2.name = "list_test_db2";
    params2.vectorDimension = 64;

    auto result1 = db_service_->create_database(params1);
    auto result2 = db_service_->create_database(params2);

    ASSERT_TRUE(result1.has_value());
    ASSERT_TRUE(result2.has_value());

    // List databases
    auto list_result = db_service_->list_databases();
    EXPECT_TRUE(list_result.has_value());
    EXPECT_GE(list_result.value().size(), 2);  // At least 2 databases
}

TEST_F(DatabaseServiceTest, DeleteDatabase) {
    // Create a database
    DatabaseCreationParams params;
    params.name = "delete_test_db";
    params.vectorDimension = 16;

    auto create_result = db_service_->create_database(params);
    ASSERT_TRUE(create_result.has_value());
    std::string db_id = create_result.value();

    // Verify it exists
    EXPECT_TRUE(db_service_->database_exists(db_id).value_or(false));

    // Delete the database
    auto delete_result = db_service_->delete_database(db_id);
    EXPECT_TRUE(delete_result.has_value());

    // Verify it no longer exists
    EXPECT_FALSE(db_service_->database_exists(db_id).value_or(false));
}

// Vector Storage Service Tests
TEST_F(VectorStorageServiceTest, StoreAndRetrieveVector) {
    Vector test_vector;
    test_vector.id = "test_vector_1";
    test_vector.values = {0.5f, 0.6f, 0.7f};
    
    // Store the vector
    auto store_result = vector_storage_->store_vector(test_database_id_, test_vector);
    EXPECT_TRUE(store_result.has_value());
    
    // Retrieve the vector
    auto retrieve_result = vector_storage_->retrieve_vector(test_database_id_, "test_vector_1");
    EXPECT_TRUE(retrieve_result.has_value());
    EXPECT_EQ(retrieve_result.value().id, "test_vector_1");
    ASSERT_EQ(retrieve_result.value().values.size(), 3);
    EXPECT_FLOAT_EQ(retrieve_result.value().values[0], 0.5f);
    EXPECT_FLOAT_EQ(retrieve_result.value().values[1], 0.6f);
    EXPECT_FLOAT_EQ(retrieve_result.value().values[2], 0.7f);
}

TEST_F(VectorStorageServiceTest, BatchStoreVectors) {
    std::vector<Vector> vectors;
    
    Vector v1;
    v1.id = "batch_vec_1";
    v1.values = {0.1f, 0.2f, 0.3f};
    
    Vector v2;
    v2.id = "batch_vec_2";
    v2.values = {0.4f, 0.5f, 0.6f};
    
    Vector v3;
    v3.id = "batch_vec_3";
    v3.values = {0.7f, 0.8f, 0.9f};
    
    vectors.push_back(v1);
    vectors.push_back(v2);
    vectors.push_back(v3);
    
    // Batch store vectors
    auto batch_result = vector_storage_->batch_store_vectors(test_database_id_, vectors);
    EXPECT_TRUE(batch_result.has_value());
    
    // Retrieve and verify each vector
    for (const auto& vec : vectors) {
        auto result = vector_storage_->retrieve_vector(test_database_id_, vec.id);
        EXPECT_TRUE(result.has_value());
        EXPECT_EQ(result.value().id, vec.id);
    }
}

TEST_F(VectorStorageServiceTest, UpdateVector) {
    Vector original_vector;
    original_vector.id = "update_test";
    original_vector.values = {0.1f, 0.2f, 0.3f};
    
    // Store original vector
    auto store_result = vector_storage_->store_vector(test_database_id_, original_vector);
    EXPECT_TRUE(store_result.has_value());
    
    // Retrieve and verify original values
    auto retrieve_result = vector_storage_->retrieve_vector(test_database_id_, "update_test");
    EXPECT_TRUE(retrieve_result.has_value());
    EXPECT_FLOAT_EQ(retrieve_result.value().values[0], 0.1f);
    
    // Update the vector
    Vector updated_vector = original_vector;
    updated_vector.values = {0.9f, 0.8f, 0.7f};
    
    auto update_result = vector_storage_->update_vector(test_database_id_, updated_vector);
    EXPECT_TRUE(update_result.has_value());
    
    // Retrieve and verify updated values
    auto updated_retrieve = vector_storage_->retrieve_vector(test_database_id_, "update_test");
    EXPECT_TRUE(updated_retrieve.has_value());
    EXPECT_FLOAT_EQ(updated_retrieve.value().values[0], 0.9f);
    EXPECT_FLOAT_EQ(updated_retrieve.value().values[1], 0.8f);
    EXPECT_FLOAT_EQ(updated_retrieve.value().values[2], 0.7f);
}

// Similarity Search Service Tests
TEST_F(SimilaritySearchServiceTest, CosineSimilaritySearch) {
    Vector query_vector;
    query_vector.id = "query";
    query_vector.values = {0.3f, 0.4f, 0.5f};
    
    SearchParams params;
    params.top_k = 5;  // Get top 5 results
    
    // Perform similarity search
    auto result = search_service_->similarity_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 1);  // Should have at least one result
    
    // Results should be sorted by similarity score (descending)
    auto results = result.value();
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].similarity_score, results[i].similarity_score);
    }
}

TEST_F(SimilaritySearchServiceTest, EuclideanSearch) {
    Vector query_vector;
    query_vector.id = "euclidean_query";
    query_vector.values = {0.3f, 0.4f, 0.5f};
    
    SearchParams params;
    params.top_k = 5;
    
    // Perform Euclidean search
    auto result = search_service_->euclidean_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 1);
    
    // Results should be sorted by similarity score (descending)
    auto results = result.value();
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].similarity_score, results[i].similarity_score);
    }
}

TEST_F(SimilaritySearchServiceTest, DotProductSearch) {
    Vector query_vector;
    query_vector.id = "dot_product_query";
    query_vector.values = {0.3f, 0.4f, 0.5f};
    
    SearchParams params;
    params.top_k = 5;
    
    // Perform dot product search
    auto result = search_service_->dot_product_search(test_database_id_, query_vector, params);
    EXPECT_TRUE(result.has_value());
    EXPECT_GE(result.value().size(), 1);
    
    // Results should be sorted by dot product value (descending)
    auto results = result.value();
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].similarity_score, results[i].similarity_score);
    }
}

} // namespace jadevectordb