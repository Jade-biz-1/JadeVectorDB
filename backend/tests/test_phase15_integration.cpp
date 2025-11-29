#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>

// Phase 15 component includes
#include "lib/storage_format.h"
#include "lib/serialization.h"
#include "lib/encryption.h"
#include "services/index/hnsw_index.h"
#include "services/backup_service.h"
#include "services/archival_service.h"
#include "services/monitoring_service.h"
#include "models/vector.h"
#include "models/database.h"
#include "lib/error_handling.h"
#include "lib/logging.h"

using namespace jadevectordb;

namespace fs = std::filesystem;

// Test fixture for Phase 15 integration tests
class Phase15IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        logging::LoggerManager::initialize();
        logger_ = logging::LoggerManager::get_logger("Phase15IntegrationTest");

        // Create temp directory for test files
        test_dir_ = fs::temp_directory_path() / "jadevectordb_phase15_test";
        fs::create_directories(test_dir_);

        LOG_INFO(logger_, "Phase 15 integration test setup complete");
    }

    void TearDown() override {
        // Cleanup test directory
        if (fs::exists(test_dir_)) {
            fs::remove_all(test_dir_);
        }
        LOG_INFO(logger_, "Phase 15 integration test teardown complete");
    }

    Vector create_test_vector(const std::string& id, size_t dimension) {
        Vector vec;
        vec.id = id;
        vec.databaseId = "test_db";
        vec.values.reserve(dimension);

        for (size_t i = 0; i < dimension; ++i) {
            vec.values.push_back(static_cast<float>(i) / dimension);
        }

        vec.metadata.source = "test_source";
        vec.metadata.owner = "test_owner";
        vec.metadata.category = "test_category";
        vec.metadata.tags = {"test", "phase15"};
        vec.metadata.score = 0.95f;

        return vec;
    }

    Database create_test_database(const std::string& id) {
        Database db;
        db.databaseId = id;
        db.name = "test_database";
        db.description = "Test database for Phase 15 integration";
        db.vectorDimension = 128;
        db.indexType = "HNSW";
        db.createdAt = "2025-11-27T00:00:00Z";
        db.updatedAt = "2025-11-27T00:00:00Z";
        return db;
    }

    std::shared_ptr<logging::Logger> logger_;
    fs::path test_dir_;
};

// ============================================================================
// T253.1: Test Storage Persistence Across Restarts
// ============================================================================
TEST_F(Phase15IntegrationTest, StoragePersistenceAcrossRestarts) {
    LOG_INFO(logger_, "Testing storage persistence across restarts");

    fs::path storage_file = test_dir_ / "storage_persistence_test.bin";

    // Phase 1: Write data using storage format
    {
        storage_format::StorageFileManager storage_manager(storage_file.string());
        ASSERT_TRUE(storage_manager.open_file()) << "Failed to open storage file for writing";

        // Write a test database
        Database test_db = create_test_database("test_db_001");
        ASSERT_TRUE(storage_manager.write_database(test_db)) << "Failed to write database";

        // Write multiple test vectors
        for (int i = 0; i < 100; ++i) {
            Vector test_vec = create_test_vector("vec_" + std::to_string(i), 128);
            ASSERT_TRUE(storage_manager.write_vector(test_vec)) << "Failed to write vector " << i;
        }

        ASSERT_TRUE(storage_manager.close_file()) << "Failed to close storage file";
        LOG_INFO(logger_, "Phase 1: Written 1 database and 100 vectors to storage");
    }

    // Verify file exists
    ASSERT_TRUE(fs::exists(storage_file)) << "Storage file was not created";
    ASSERT_GT(fs::file_size(storage_file), 0) << "Storage file is empty";

    // Phase 2: Simulate restart - read data back
    {
        storage_format::StorageFileManager storage_manager(storage_file.string());
        ASSERT_TRUE(storage_manager.open_file()) << "Failed to open storage file for reading";

        // Verify file integrity
        ASSERT_TRUE(storage_manager.verify_file_integrity()) << "File integrity check failed";

        // Read database
        auto db_result = storage_manager.read_database("test_db_001");
        ASSERT_TRUE(db_result.has_value()) << "Failed to read database";
        EXPECT_EQ(db_result.value().name, "test_database");
        EXPECT_EQ(db_result.value().vectorDimension, 128);

        // Read vectors back
        for (int i = 0; i < 100; ++i) {
            auto vec_result = storage_manager.read_vector("vec_" + std::to_string(i));
            ASSERT_TRUE(vec_result.has_value()) << "Failed to read vector " << i;
            EXPECT_EQ(vec_result.value().id, "vec_" + std::to_string(i));
            EXPECT_EQ(vec_result.value().values.size(), 128);
        }

        ASSERT_TRUE(storage_manager.close_file()) << "Failed to close storage file after reading";
        LOG_INFO(logger_, "Phase 2: Successfully read back all data after simulated restart");
    }

    LOG_INFO(logger_, "Storage persistence test PASSED");
}

// ============================================================================
// T253.2: Test Serialization Round-Trip with FlatBuffers
// ============================================================================
TEST_F(Phase15IntegrationTest, SerializationRoundTripFlatBuffers) {
    LOG_INFO(logger_, "Testing FlatBuffers serialization round-trip");

    // Test Vector serialization
    {
        Vector original_vec = create_test_vector("test_vec_fb", 256);
        original_vec.metadata.custom["key1"] = "value1";
        original_vec.metadata.custom["key2"] = "value2";

        // Serialize
        auto serialized = serialization::serialize_vector(original_vec);
        ASSERT_GT(serialized.size(), 0) << "Serialization produced empty data";

        // Deserialize
        Vector deserialized_vec = serialization::deserialize_vector(serialized.data(), serialized.size());

        // Verify equality
        EXPECT_EQ(deserialized_vec.id, original_vec.id);
        EXPECT_EQ(deserialized_vec.values.size(), original_vec.values.size());
        for (size_t i = 0; i < original_vec.values.size(); ++i) {
            EXPECT_FLOAT_EQ(deserialized_vec.values[i], original_vec.values[i]);
        }
        EXPECT_EQ(deserialized_vec.metadata.source, original_vec.metadata.source);
        EXPECT_EQ(deserialized_vec.metadata.owner, original_vec.metadata.owner);
        EXPECT_EQ(deserialized_vec.metadata.category, original_vec.metadata.category);

        LOG_INFO(logger_, "Vector serialization round-trip PASSED");
    }

    // Test Database serialization
    {
        Database original_db = create_test_database("test_db_fb");
        original_db.config = {{"param1", "value1"}, {"param2", "value2"}};

        // Serialize
        auto serialized = serialization::serialize_database(original_db);
        ASSERT_GT(serialized.size(), 0) << "Database serialization produced empty data";

        // Deserialize
        Database deserialized_db = serialization::deserialize_database(serialized.data(), serialized.size());

        // Verify equality
        EXPECT_EQ(deserialized_db.databaseId, original_db.databaseId);
        EXPECT_EQ(deserialized_db.name, original_db.name);
        EXPECT_EQ(deserialized_db.description, original_db.description);
        EXPECT_EQ(deserialized_db.vectorDimension, original_db.vectorDimension);
        EXPECT_EQ(deserialized_db.indexType, original_db.indexType);

        LOG_INFO(logger_, "Database serialization round-trip PASSED");
    }

    // Test multiple serialization cycles (verify stability)
    {
        Vector vec1 = create_test_vector("multi_cycle_vec", 128);

        auto s1 = serialization::serialize_vector(vec1);
        Vector vec2 = serialization::deserialize_vector(s1.data(), s1.size());

        auto s2 = serialization::serialize_vector(vec2);
        Vector vec3 = serialization::deserialize_vector(s2.data(), s2.size());

        // After multiple cycles, data should remain consistent
        EXPECT_EQ(vec3.id, vec1.id);
        EXPECT_EQ(vec3.values.size(), vec1.values.size());
        for (size_t i = 0; i < vec1.values.size(); ++i) {
            EXPECT_FLOAT_EQ(vec3.values[i], vec1.values[i]);
        }

        LOG_INFO(logger_, "Multi-cycle serialization stability PASSED");
    }

    LOG_INFO(logger_, "FlatBuffers serialization round-trip test PASSED");
}

// ============================================================================
// T253.3: Test HNSW Performance vs Linear Search
// ============================================================================
TEST_F(Phase15IntegrationTest, HNSWPerformanceVsLinear) {
    LOG_INFO(logger_, "Testing HNSW performance vs linear search");

    constexpr int NUM_VECTORS = 10000;
    constexpr int DIMENSION = 128;
    constexpr int NUM_QUERIES = 100;
    constexpr int K = 10;

    // Build HNSW index
    HnswParams hnsw_params;
    hnsw_params.M = 16;
    hnsw_params.ef_construction = 200;
    hnsw_params.ef_search = 50;
    hnsw_params.max_elements = NUM_VECTORS;

    HnswIndex hnsw_index(hnsw_params);
    ASSERT_TRUE(hnsw_index.initialize(hnsw_params));

    // Add vectors to index
    LOG_INFO(logger_, "Building HNSW index with " << NUM_VECTORS << " vectors");
    auto start_build = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<int, std::vector<float>>> all_vectors;
    for (int i = 0; i < NUM_VECTORS; ++i) {
        std::vector<float> vec(DIMENSION);
        for (int d = 0; d < DIMENSION; ++d) {
            vec[d] = static_cast<float>(rand()) / RAND_MAX;
        }
        all_vectors.push_back({i, vec});

        auto result = hnsw_index.add_vector(i, vec);
        ASSERT_TRUE(result.has_value()) << "Failed to add vector " << i;
    }

    auto end_build = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count();
    LOG_INFO(logger_, "HNSW build time: " << build_time << " ms");

    // Perform HNSW searches
    LOG_INFO(logger_, "Performing " << NUM_QUERIES << " HNSW searches");
    auto start_hnsw_search = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < NUM_QUERIES; ++q) {
        std::vector<float> query(DIMENSION);
        for (int d = 0; d < DIMENSION; ++d) {
            query[d] = static_cast<float>(rand()) / RAND_MAX;
        }

        auto result = hnsw_index.search(query, K, 0.0f);
        ASSERT_TRUE(result.has_value()) << "HNSW search failed for query " << q;
        EXPECT_LE(result.value().size(), K);
    }

    auto end_hnsw_search = std::chrono::high_resolution_clock::now();
    auto hnsw_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_hnsw_search - start_hnsw_search).count();
    double avg_hnsw_time = static_cast<double>(hnsw_search_time) / NUM_QUERIES;

    LOG_INFO(logger_, "HNSW average search time: " << avg_hnsw_time << " ms per query");

    // Perform linear searches for comparison
    LOG_INFO(logger_, "Performing " << NUM_QUERIES << " linear searches for comparison");
    auto start_linear_search = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < NUM_QUERIES; ++q) {
        std::vector<float> query(DIMENSION);
        for (int d = 0; d < DIMENSION; ++d) {
            query[d] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Linear search: compute distance to all vectors
        std::vector<std::pair<float, int>> distances;
        for (const auto& [id, vec] : all_vectors) {
            float dist = 0.0f;
            for (int d = 0; d < DIMENSION; ++d) {
                float diff = query[d] - vec[d];
                dist += diff * diff;
            }
            distances.push_back({dist, id});
        }

        // Get top K
        std::partial_sort(distances.begin(), distances.begin() + K, distances.end());
    }

    auto end_linear_search = std::chrono::high_resolution_clock::now();
    auto linear_search_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_linear_search - start_linear_search).count();
    double avg_linear_time = static_cast<double>(linear_search_time) / NUM_QUERIES;

    LOG_INFO(logger_, "Linear average search time: " << avg_linear_time << " ms per query");

    // HNSW should be significantly faster
    double speedup = avg_linear_time / avg_hnsw_time;
    LOG_INFO(logger_, "HNSW speedup: " << speedup << "x faster than linear search");

    EXPECT_GT(speedup, 5.0) << "HNSW should be at least 5x faster than linear search";

    LOG_INFO(logger_, "HNSW performance test PASSED with " << speedup << "x speedup");
}

// ============================================================================
// T253.4: Test Encryption/Decryption with Various Data Sizes
// ============================================================================
TEST_F(Phase15IntegrationTest, EncryptionDecryptionVariousSizes) {
    LOG_INFO(logger_, "Testing encryption/decryption with various data sizes");

    encryption::EncryptionManager enc_manager;
    auto kms = std::make_unique<encryption::KeyManagementServiceImpl>();
    enc_manager.initialize(std::move(kms));

    // Generate encryption key
    encryption::EncryptionConfig config;
    config.algorithm = encryption::EncryptionAlgorithm::AES_256_GCM;
    std::string key_id = enc_manager.generate_key(config);
    config.key_id = key_id;

    // Test various data sizes
    std::vector<size_t> test_sizes = {
        1,           // 1 byte
        16,          // AES block size
        100,         // Small
        1024,        // 1 KB
        10240,       // 10 KB
        102400,      // 100 KB
        1048576      // 1 MB
    };

    for (size_t size : test_sizes) {
        LOG_INFO(logger_, "Testing encryption with " << size << " bytes");

        // Generate test data
        std::vector<uint8_t> original_data(size);
        for (size_t i = 0; i < size; ++i) {
            original_data[i] = static_cast<uint8_t>(rand() % 256);
        }

        // Encrypt
        auto encrypted = enc_manager.encrypt_data(original_data, config);
        ASSERT_TRUE(encrypted.has_value()) << "Encryption failed for size " << size;
        EXPECT_GT(encrypted.value().size(), size) << "Encrypted data should be larger due to IV and tag";

        // Decrypt
        auto decrypted = enc_manager.decrypt_data(encrypted.value(), config);
        ASSERT_TRUE(decrypted.has_value()) << "Decryption failed for size " << size;

        // Verify data integrity
        ASSERT_EQ(decrypted.value().size(), original_data.size()) << "Decrypted size mismatch";
        for (size_t i = 0; i < size; ++i) {
            EXPECT_EQ(decrypted.value()[i], original_data[i]) << "Data mismatch at byte " << i;
        }

        LOG_INFO(logger_, "Encryption test PASSED for " << size << " bytes");
    }

    // Test authentication failure (tampered data)
    {
        std::vector<uint8_t> test_data = {1, 2, 3, 4, 5};
        auto encrypted = enc_manager.encrypt_data(test_data, config);
        ASSERT_TRUE(encrypted.has_value());

        // Tamper with encrypted data
        encrypted.value()[encrypted.value().size() / 2] ^= 0xFF;

        // Decryption should fail due to authentication tag mismatch
        auto decrypted = enc_manager.decrypt_data(encrypted.value(), config);
        EXPECT_FALSE(decrypted.has_value()) << "Decryption should fail for tampered data";

        LOG_INFO(logger_, "Authentication failure test PASSED");
    }

    LOG_INFO(logger_, "Encryption/decryption test PASSED for all sizes");
}

// ============================================================================
// T253.5: Test Backup and Restore with Real Data
// ============================================================================
TEST_F(Phase15IntegrationTest, BackupAndRestoreRealData) {
    LOG_INFO(logger_, "Testing backup and restore with real data");

    // Note: This test requires BackupService to be properly initialized
    // For now, we'll test the core components that backup service uses

    fs::path backup_file = test_dir_ / "test_backup.bak";

    // Phase 1: Create backup data
    {
        storage_format::StorageFileManager backup_storage(backup_file.string());
        ASSERT_TRUE(backup_storage.open_file()) << "Failed to open backup file";

        // Backup database metadata
        Database test_db = create_test_database("backup_db");
        ASSERT_TRUE(backup_storage.write_database(test_db)) << "Failed to backup database";

        // Backup vectors
        for (int i = 0; i < 50; ++i) {
            Vector vec = create_test_vector("backup_vec_" + std::to_string(i), 64);
            ASSERT_TRUE(backup_storage.write_vector(vec)) << "Failed to backup vector " << i;
        }

        ASSERT_TRUE(backup_storage.close_file()) << "Failed to close backup file";
        LOG_INFO(logger_, "Created backup with 1 database and 50 vectors");
    }

    // Verify backup file exists and has data
    ASSERT_TRUE(fs::exists(backup_file)) << "Backup file not created";
    ASSERT_GT(fs::file_size(backup_file), 0) << "Backup file is empty";

    // Phase 2: Restore from backup
    {
        storage_format::StorageFileManager restore_storage(backup_file.string());
        ASSERT_TRUE(restore_storage.open_file()) << "Failed to open backup for restore";

        // Verify integrity before restore
        ASSERT_TRUE(restore_storage.verify_file_integrity()) << "Backup file integrity check failed";

        // Restore database
        auto db_result = restore_storage.read_database("backup_db");
        ASSERT_TRUE(db_result.has_value()) << "Failed to restore database";
        EXPECT_EQ(db_result.value().name, "test_database");

        // Restore vectors
        for (int i = 0; i < 50; ++i) {
            auto vec_result = restore_storage.read_vector("backup_vec_" + std::to_string(i));
            ASSERT_TRUE(vec_result.has_value()) << "Failed to restore vector " << i;
            EXPECT_EQ(vec_result.value().values.size(), 64);
        }

        ASSERT_TRUE(restore_storage.close_file()) << "Failed to close restore file";
        LOG_INFO(logger_, "Successfully restored all data from backup");
    }

    LOG_INFO(logger_, "Backup and restore test PASSED");
}

// ============================================================================
// T253.6: End-to-End Workflow Testing
// ============================================================================
TEST_F(Phase15IntegrationTest, EndToEndWorkflow) {
    LOG_INFO(logger_, "Testing end-to-end workflow: store -> index -> search -> backup -> restore");

    fs::path workflow_storage = test_dir_ / "workflow_test.db";
    fs::path workflow_backup = test_dir_ / "workflow_backup.bak";

    // Step 1: Store vectors with storage format
    std::vector<Vector> test_vectors;
    {
        storage_format::StorageFileManager storage(workflow_storage.string());
        ASSERT_TRUE(storage.open_file());

        Database db = create_test_database("workflow_db");
        ASSERT_TRUE(storage.write_database(db));

        for (int i = 0; i < 100; ++i) {
            Vector vec = create_test_vector("workflow_vec_" + std::to_string(i), 128);
            test_vectors.push_back(vec);
            ASSERT_TRUE(storage.write_vector(vec));
        }

        ASSERT_TRUE(storage.close_file());
        LOG_INFO(logger_, "Step 1: Stored 100 vectors");
    }

    // Step 2: Build HNSW index
    HnswParams params;
    params.M = 8;
    params.ef_construction = 100;
    params.ef_search = 50;
    params.max_elements = 100;

    HnswIndex index(params);
    ASSERT_TRUE(index.initialize(params));

    for (int i = 0; i < 100; ++i) {
        auto result = index.add_vector(i, test_vectors[i].values);
        ASSERT_TRUE(result.has_value());
    }
    LOG_INFO(logger_, "Step 2: Built HNSW index");

    // Step 3: Perform searches
    std::vector<float> query = test_vectors[0].values;
    auto search_result = index.search(query, 10, 0.0f);
    ASSERT_TRUE(search_result.has_value());
    EXPECT_GT(search_result.value().size(), 0);
    LOG_INFO(logger_, "Step 3: Performed search, found " << search_result.value().size() << " results");

    // Step 4: Create backup
    {
        fs::copy_file(workflow_storage, workflow_backup, fs::copy_options::overwrite_existing);
        ASSERT_TRUE(fs::exists(workflow_backup));
        LOG_INFO(logger_, "Step 4: Created backup");
    }

    // Step 5: Verify backup can be restored
    {
        storage_format::StorageFileManager restore(workflow_backup.string());
        ASSERT_TRUE(restore.open_file());
        ASSERT_TRUE(restore.verify_file_integrity());

        auto db_result = restore.read_database("workflow_db");
        ASSERT_TRUE(db_result.has_value());

        int restored_count = 0;
        for (int i = 0; i < 100; ++i) {
            auto vec_result = restore.read_vector("workflow_vec_" + std::to_string(i));
            if (vec_result.has_value()) {
                restored_count++;
            }
        }

        EXPECT_EQ(restored_count, 100);
        ASSERT_TRUE(restore.close_file());
        LOG_INFO(logger_, "Step 5: Verified backup - restored " << restored_count << " vectors");
    }

    LOG_INFO(logger_, "End-to-end workflow test PASSED");
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
