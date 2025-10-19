// Property-based testing framework for JadeVectorDB
// Based on the concept of property-based testing but implemented for C++/GTest

#ifndef PROPERTY_TEST_FRAMEWORK_H
#define PROPERTY_TEST_FRAMEWORK_H

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace property_tests {

// Generator base class
template<typename T>
class Generator {
public:
    virtual ~Generator() = default;
    virtual T generate(std::mt19937& rng) = 0;
};

// Integer generator
class IntGenerator : public Generator<int> {
private:
    int min_val;
    int max_val;
public:
    IntGenerator(int min, int max) : min_val(min), max_val(max) {
        if (min > max) throw std::invalid_argument("Min value cannot be greater than max value");
    }
    
    int generate(std::mt19937& rng) override {
        std::uniform_int_distribution<int> dist(min_val, max_val);
        return dist(rng);
    }
};

// Float generator
class FloatGenerator : public Generator<float> {
private:
    float min_val;
    float max_val;
public:
    FloatGenerator(float min, float max) : min_val(min), max_val(max) {
        if (min > max) throw std::invalid_argument("Min value cannot be greater than max value");
    }
    
    float generate(std::mt19937& rng) override {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        return dist(rng);
    }
};

// Vector generator (generates vectors of random floats)
class VectorGenerator : public Generator<std::vector<float>> {
private:
    int dimension;
    float min_val;
    float max_val;
public:
    VectorGenerator(int dim, float min, float max) : dimension(dim), min_val(min), max_val(max) {
        if (dim <= 0) throw std::invalid_argument("Dimension must be positive");
    }
    
    std::vector<float> generate(std::mt19937& rng) override {
        std::vector<float> result(dimension);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        for (int i = 0; i < dimension; ++i) {
            result[i] = dist(rng);
        }
        
        return result;
    }
};

// Property test runner
template<typename T>
class PropertyTest {
private:
    std::string name;
    std::function<bool(const T&)> property;
    Generator<T>* generator;
    int num_tests;
    
public:
    PropertyTest(const std::string& test_name, 
                 std::function<bool(const T&)> prop, 
                 Generator<T>* gen, 
                 int n_tests = 100) 
        : name(test_name), property(prop), generator(gen), num_tests(n_tests) {}
    
    void run() {
        std::mt19937 rng(std::random_device{}());
        
        for (int i = 0; i < num_tests; ++i) {
            T value = generator->generate(rng);
            bool result = property(value);
            
            if (!result) {
                ADD_FAILURE() << "Property '" << name << "' failed on test " << (i+1) 
                             << " with value: " << value_to_string(value);
                return; // Stop on first failure to avoid spam
            }
        }
        
        SUCCEED() << "Property '" << name << "' passed " << num_tests << " tests";
    }
    
private:
    // Helper function to convert values to string for error messages
    std::string value_to_string(const std::vector<float>& vec) {
        std::string result = "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(vec[i]);
        }
        result += "]";
        return result;
    }
    
    std::string value_to_string(const int& val) {
        return std::to_string(val);
    }
    
    std::string value_to_string(const float& val) {
        return std::to_string(val);
    }
};

// Specialized runner for pairs of values (for properties that take two arguments)
template<typename T1, typename T2>
class PropertyTest2 {
private:
    std::string name;
    std::function<bool(const T1&, const T2&)> property;
    Generator<T1>* gen1;
    Generator<T2>* gen2;
    int num_tests;
    
public:
    PropertyTest2(const std::string& test_name, 
                  std::function<bool(const T1&, const T2&)> prop, 
                  Generator<T1>* g1, 
                  Generator<T2>* g2, 
                  int n_tests = 100) 
        : name(test_name), property(prop), gen1(g1), gen2(g2), num_tests(n_tests) {}
    
    void run() {
        std::mt19937 rng(std::random_device{}());
        
        for (int i = 0; i < num_tests; ++i) {
            T1 val1 = gen1->generate(rng);
            T2 val2 = gen2->generate(rng);
            bool result = property(val1, val2);
            
            if (!result) {
                ADD_FAILURE() << "Property '" << name << "' failed on test " << (i+1);
                return; // Stop on first failure to avoid spam
            }
        }
        
        SUCCEED() << "Property '" << name << "' passed " << num_tests << " tests";
    }
};

// Utility functions for common vector operations
namespace utils {
    float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension");
        }
        
        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension");
        }
        
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension");
        }
        
        float dot = dot_product(a, b);
        float norm_a = std::sqrt(dot_product(a, a));
        float norm_b = std::sqrt(dot_product(b, b));
        
        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f; // Return 0 for zero vectors
        }
        
        return dot / (norm_a * norm_b);
    }
    
    float vector_norm(const std::vector<float>& v) {
        return std::sqrt(dot_product(v, v));
    }
    
    std::vector<float> normalize_vector(const std::vector<float>& v) {
        float norm = vector_norm(v);
        if (norm == 0.0f) {
            return v; // Return as-is if zero vector
        }
        
        std::vector<float> result = v;
        for (auto& val : result) {
            val /= norm;
        }
        return result;
    }
}

} // namespace property_tests

#endif // PROPERTY_TEST_FRAMEWORK_H