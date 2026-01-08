#include "bm25_index_builder.h"
#include <thread>
#include <chrono>
#include <algorithm>

namespace jadedb {
namespace search {

BM25IndexBuilder::BM25IndexBuilder(
    const std::string& database_id,
    const BuildConfig& config
)
    : database_id_(database_id),
      config_(config),
      building_(false),
      cancel_requested_(false)
{
    logger_ = jadevectordb::logging::LoggerManager::get_logger("BM25IndexBuilder");

    // Initialize components
    bm25_scorer_ = std::make_unique<BM25Scorer>(config.bm25_config);
    inverted_index_ = std::make_unique<InvertedIndex>();

    // Initialize persistence if path is provided
    if (!config.persistence_path.empty()) {
        persistence_ = std::make_unique<BM25IndexPersistence>(database_id, config.persistence_path);
        if (!persistence_->initialize()) {
            LOG_WARN(logger_, "Failed to initialize BM25 persistence at: " << config.persistence_path);
        }
    }

    // Initialize progress
    progress_.status = BuildStatus::IDLE;
    progress_.total_documents = 0;
    progress_.processed_documents = 0;
    progress_.indexed_terms = 0;
    progress_.progress_percentage = 0.0;
}

bool BM25IndexBuilder::build_from_documents(
    const std::vector<BM25Document>& documents,
    ProgressCallback callback
) {
    if (building_.load()) {
        LOG_WARN(logger_, "Build already in progress for database: " << database_id_);
        return false;
    }

    if (documents.empty()) {
        LOG_WARN(logger_, "No documents provided for indexing");
        return false;
    }

    LOG_INFO(logger_, "Starting BM25 index build for " << documents.size() << " documents");

    // Start build in background thread
    std::thread build_thread([this, documents, callback]() {
        build_internal(documents, callback);
    });
    build_thread.detach();

    return true;
}

bool BM25IndexBuilder::build_from_provider(
    DocumentSourceProvider provider,
    ProgressCallback callback
) {
    if (!provider) {
        LOG_ERROR(logger_, "Invalid document provider");
        return false;
    }

    try {
        auto documents = provider();
        return build_from_documents(documents, callback);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception in document provider: " << e.what());
        return false;
    }
}

bool BM25IndexBuilder::rebuild_index(
    const std::vector<BM25Document>& documents,
    ProgressCallback callback
) {
    LOG_INFO(logger_, "Rebuilding BM25 index from scratch");

    // Clear existing index
    clear_index();

    // Build new index
    return build_from_documents(documents, callback);
}

bool BM25IndexBuilder::add_documents(const std::vector<BM25Document>& documents) {
    if (building_.load()) {
        LOG_WARN(logger_, "Cannot add documents while build is in progress");
        return false;
    }

    std::lock_guard<std::mutex> lock(build_mutex_);

    LOG_INFO(logger_, "Adding " << documents.size() << " documents to BM25 index");

    try {
        // NOTE: BM25Scorer::index_documents() clears the index first,
        // so we can't use it for incremental updates.
        // Instead, we need to rebuild the entire index.

        // For now, just add to inverted index without BM25 scorer
        // This is a limitation - incremental adds won't update BM25 statistics properly
        // TODO: Implement proper incremental BM25 indexing

        for (const auto& doc : documents) {
            // Tokenize
            auto tokens = bm25_scorer_->tokenize(doc.text);

            // Build term frequencies manually
            std::unordered_map<std::string, int> term_freqs;
            for (const auto& token : tokens) {
                term_freqs[token]++;
            }

            // Add to inverted index
            inverted_index_->add_document_with_frequencies(doc.doc_id, term_freqs);
        }

        // Persist if enabled
        if (config_.persist_on_completion && persistence_) {
            persist_index();
        }

        LOG_INFO(logger_, "Successfully added " << documents.size() << " documents to inverted index");
        LOG_WARN(logger_, "Note: BM25 statistics not updated for incremental adds. Consider rebuild for accurate BM25 scores.");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception adding documents: " << e.what());
        return false;
    }
}

bool BM25IndexBuilder::remove_documents(const std::vector<std::string>& doc_ids) {
    if (building_.load()) {
        LOG_WARN(logger_, "Cannot remove documents while build is in progress");
        return false;
    }

    std::lock_guard<std::mutex> lock(build_mutex_);

    LOG_INFO(logger_, "Removing " << doc_ids.size() << " documents from BM25 index");

    try {
        for (const auto& doc_id : doc_ids) {
            inverted_index_->remove_document(doc_id);
        }

        // Persist if enabled
        if (config_.persist_on_completion && persistence_) {
            persist_index();
        }

        LOG_INFO(logger_, "Successfully removed " << doc_ids.size() << " documents");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception removing documents: " << e.what());
        return false;
    }
}

bool BM25IndexBuilder::update_documents(const std::vector<BM25Document>& documents) {
    if (building_.load()) {
        LOG_WARN(logger_, "Cannot update documents while build is in progress");
        return false;
    }

    LOG_INFO(logger_, "Updating " << documents.size() << " documents in BM25 index");

    // Extract document IDs
    std::vector<std::string> doc_ids;
    doc_ids.reserve(documents.size());
    for (const auto& doc : documents) {
        doc_ids.push_back(doc.doc_id);
    }

    // Remove old versions
    if (!remove_documents(doc_ids)) {
        return false;
    }

    // Add new versions
    return add_documents(documents);
}

BuildProgress BM25IndexBuilder::get_progress() const {
    std::lock_guard<std::mutex> lock(build_mutex_);
    return progress_;
}

bool BM25IndexBuilder::is_building() const {
    return building_.load();
}

bool BM25IndexBuilder::wait_for_completion(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (building_.load()) {
        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start
            ).count();

            if (elapsed >= timeout_ms) {
                return false;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return true;
}

void BM25IndexBuilder::cancel_build() {
    LOG_WARN(logger_, "Cancelling BM25 index build");
    cancel_requested_.store(true);
}

bool BM25IndexBuilder::persist_index(const std::string& path) {
    if (!persistence_) {
        std::string persistence_path = path.empty() ? config_.persistence_path : path;
        if (persistence_path.empty()) {
            LOG_ERROR(logger_, "No persistence path configured");
            return false;
        }

        persistence_ = std::make_unique<BM25IndexPersistence>(database_id_, persistence_path);
        if (!persistence_->initialize()) {
            LOG_ERROR(logger_, "Failed to initialize persistence");
            return false;
        }
    }

    LOG_INFO(logger_, "Persisting BM25 index to storage");

    try {
        return persistence_->save_index(*bm25_scorer_, *inverted_index_);
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception persisting index: " << e.what());
        return false;
    }
}

bool BM25IndexBuilder::load_index(const std::string& path) {
    if (!persistence_) {
        persistence_ = std::make_unique<BM25IndexPersistence>(database_id_, path);
        if (!persistence_->initialize()) {
            LOG_ERROR(logger_, "Failed to initialize persistence for loading");
            return false;
        }
    }

    LOG_INFO(logger_, "Loading BM25 index from storage");

    try {
        bool success = persistence_->load_index(*bm25_scorer_, *inverted_index_);

        if (success) {
            LOG_INFO(logger_, "Successfully loaded BM25 index");

            // Update progress
            std::lock_guard<std::mutex> lock(build_mutex_);
            progress_.status = BuildStatus::COMPLETED;
            progress_.total_documents = bm25_scorer_->get_document_count();
            progress_.processed_documents = progress_.total_documents;
            progress_.indexed_terms = inverted_index_->term_count();
            progress_.progress_percentage = 100.0;
        }

        return success;

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception loading index: " << e.what());
        return false;
    }
}

void BM25IndexBuilder::get_index_stats(
    size_t& total_docs,
    size_t& total_terms,
    double& avg_doc_length
) const {
    total_docs = bm25_scorer_->get_document_count();
    total_terms = inverted_index_->term_count();
    avg_doc_length = bm25_scorer_->get_avg_doc_length();
}

bool BM25IndexBuilder::is_index_ready() const {
    return bm25_scorer_->get_document_count() > 0 &&
           inverted_index_->term_count() > 0;
}

std::shared_ptr<HybridSearchEngine> BM25IndexBuilder::get_search_engine() {
    if (!search_engine_) {
        // Create hybrid search engine
        HybridSearchConfig hybrid_config;
        hybrid_config.bm25_config = config_.bm25_config;

        search_engine_ = std::make_shared<HybridSearchEngine>(
            database_id_,
            hybrid_config
        );

        // Build BM25 index in search engine
        // Note: We need to rebuild with documents, but for now
        // we'll create it and expect the user to build it separately
        LOG_INFO(logger_, "Created HybridSearchEngine for database: " << database_id_);
    }

    return search_engine_;
}

void BM25IndexBuilder::set_config(const BuildConfig& config) {
    std::lock_guard<std::mutex> lock(build_mutex_);
    config_ = config;
}

const BuildConfig& BM25IndexBuilder::get_config() const {
    return config_;
}

// Private methods

void BM25IndexBuilder::build_internal(
    const std::vector<BM25Document>& documents,
    ProgressCallback callback
) {
    building_.store(true);
    cancel_requested_.store(false);

    {
        std::lock_guard<std::mutex> lock(build_mutex_);
        progress_.status = BuildStatus::IN_PROGRESS;
        progress_.total_documents = documents.size();
        progress_.processed_documents = 0;
        progress_.indexed_terms = 0;
        progress_.error_message.clear();
        progress_.started_at = std::chrono::system_clock::now();
        progress_.duration_ms = 0;
    }

    LOG_INFO(logger_, "Building BM25 index for " << documents.size() << " documents");

    try {
        size_t total_indexed_terms = 0;
        size_t processed = 0;

        // Process documents in batches
        for (size_t i = 0; i < documents.size(); i += config_.batch_size) {
            // Check for cancellation
            if (cancel_requested_.load()) {
                LOG_WARN(logger_, "Build cancelled by user");
                std::lock_guard<std::mutex> lock(build_mutex_);
                progress_.status = BuildStatus::FAILED;
                progress_.error_message = "Build cancelled by user";
                building_.store(false);
                return;
            }

            // Extract batch
            size_t batch_end = std::min(i + config_.batch_size, documents.size());
            std::vector<BM25Document> batch(
                documents.begin() + i,
                documents.begin() + batch_end
            );

            // Process batch
            if (!process_batch(batch, total_indexed_terms)) {
                std::lock_guard<std::mutex> lock(build_mutex_);
                progress_.status = BuildStatus::FAILED;
                progress_.error_message = "Failed to process batch";
                building_.store(false);
                return;
            }

            processed += batch.size();

            // Update progress
            update_progress(processed, documents.size());

            // Call progress callback if provided
            if (callback) {
                callback(get_progress());
            }
        }

        // Finalize build
        finalize_build();

        // Mark as completed
        {
            std::lock_guard<std::mutex> lock(build_mutex_);
            progress_.status = BuildStatus::COMPLETED;
            progress_.processed_documents = documents.size();
            progress_.indexed_terms = total_indexed_terms;
            progress_.progress_percentage = 100.0;
            progress_.completed_at = std::chrono::system_clock::now();
            progress_.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                progress_.completed_at - progress_.started_at
            ).count();
        }

        LOG_INFO(logger_, "BM25 index build completed successfully in "
                 << progress_.duration_ms << "ms. Indexed "
                 << total_indexed_terms << " unique terms from "
                 << documents.size() << " documents");

        // Final callback
        if (callback) {
            callback(get_progress());
        }

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception during index build: " << e.what());

        std::lock_guard<std::mutex> lock(build_mutex_);
        progress_.status = BuildStatus::FAILED;
        progress_.error_message = std::string("Exception: ") + e.what();
    }

    building_.store(false);
}

void BM25IndexBuilder::update_progress(size_t processed, size_t total) {
    std::lock_guard<std::mutex> lock(build_mutex_);
    progress_.processed_documents = processed;
    progress_.progress_percentage = progress_.get_progress();
}

bool BM25IndexBuilder::process_batch(
    const std::vector<BM25Document>& batch,
    size_t& indexed_terms
) {
    try {
        // Index documents in BM25 scorer
        bm25_scorer_->index_documents(batch);

        // Add to inverted index
        for (const auto& doc : batch) {
            // Tokenize
            auto tokens = bm25_scorer_->tokenize(doc.text);

            // Build term frequencies manually
            std::unordered_map<std::string, int> term_freqs;
            for (const auto& token : tokens) {
                term_freqs[token]++;
            }

            // Add to index
            inverted_index_->add_document_with_frequencies(doc.doc_id, term_freqs);
        }

        // Update indexed terms count
        indexed_terms = inverted_index_->term_count();

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Exception processing batch: " << e.what());
        return false;
    }
}

void BM25IndexBuilder::finalize_build() {
    // Persist if enabled
    if (config_.persist_on_completion) {
        if (!persist_index()) {
            LOG_WARN(logger_, "Failed to persist index after build completion");
        }
    }

    LOG_INFO(logger_, "Index build finalized. Stats: "
             << "docs=" << bm25_scorer_->get_document_count() << ", "
             << "terms=" << inverted_index_->term_count() << ", "
             << "avg_len=" << bm25_scorer_->get_avg_doc_length());
}

void BM25IndexBuilder::clear_index() {
    std::lock_guard<std::mutex> lock(build_mutex_);

    LOG_INFO(logger_, "Clearing BM25 index");

    // Reinitialize components
    bm25_scorer_ = std::make_unique<BM25Scorer>(config_.bm25_config);
    inverted_index_ = std::make_unique<InvertedIndex>();

    // Reset progress
    progress_.status = BuildStatus::IDLE;
    progress_.total_documents = 0;
    progress_.processed_documents = 0;
    progress_.indexed_terms = 0;
    progress_.progress_percentage = 0.0;
    progress_.error_message.clear();
}

} // namespace search
} // namespace jadedb
