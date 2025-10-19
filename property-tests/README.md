# Property-Based Testing Framework for JadeVectorDB

## Overview

Property-based testing is a testing methodology where instead of testing specific examples, you define general properties that should hold true for a wide range of inputs. This framework implements property-based testing for JadeVectorDB to validate system invariants and improve test coverage.

## Components

The framework consists of:

1. **Core Framework** - Provides the infrastructure for property-based testing
2. **Property Definitions** - Specific properties for different aspects of JadeVectorDB
3. **Generators** - Create random test data for property testing
4. **Test Suites** - Organized collections of property tests

## Directory Structure

```
property-tests/
├── framework/                    # Core property testing framework
│   └── property_test_framework.h
├── vector-space/                 # Vector space property tests
│   ├── vector_space_properties.h
│   └── test_vector_space_properties.cpp
├── consistency/                  # Consistency guarantee tests
│   ├── consistency_properties.h
│   └── test_consistency_properties.cpp
├── concurrency/                  # Concurrency property tests
│   ├── concurrency_properties.h
│   └── test_concurrency_properties.cpp
├── distributed-system/           # Distributed system invariants
│   ├── distributed_invariants.h
│   └── test_distributed_invariants.cpp
├── CMakeLists.txt               # Build configuration
├── test_property_suite.cpp      # Main test suite
└── README.md                    # This documentation
```

## Framework Architecture

The property testing framework is built around these key concepts:

### Generators
Generators create random test data. The framework includes:
- `Generator<T>` base class
- `IntGenerator`, `FloatGenerator`, `VectorGenerator` for primitive types
- Specialized generators for complex data structures

### Properties
Properties are functions that return true if an invariant holds:
- `PropertyTest<T>` for single-parameter properties
- `PropertyTest2<T1, T2>` for two-parameter properties

### Utilities
Helper functions for common operations:
- Distance calculations (Euclidean, Cosine)
- Vector normalization
- Linear algebra operations

## Implemented Properties

### Vector Space Properties

1. **Dimension Consistency**
   - All vectors in the same space have the same dimension

2. **Norm Bounds** 
   - Normalized vectors have a norm close to 1.0

3. **Distance Metrics**
   - Non-negativity: distance ≥ 0
   - Identity: distance(a, a) = 0
   - Symmetry: distance(a, b) = distance(b, a)
   - Triangle inequality: distance(a, c) ≤ distance(a, b) + distance(b, c)

4. **Cosine Similarity Bounds**
   - Cosine similarity values are in the range [-1, 1]

5. **Linear Combination Preservation**
   - Vector operations preserve space properties

### Consistency Guarantees

1. **Atomicity**
   - Database operations are atomic (all-or-nothing)

2. **Consistency**
   - Database remains in valid state after operations

3. **Vector Integrity**
   - Vectors maintain their values after operations

4. **Size Consistency**
   - Database size changes predictably with operations

### Concurrency Properties

1. **Thread Safety**
   - Multiple threads can access data without race conditions

2. **Atomic Operations**
   - Individual operations complete without interruption

3. **Read-Write Consistency**
   - Readers and writers maintain data consistency

### Distributed System Invariants

1. **Data Consistency**
   - Same data available across all replicas

2. **Partition Tolerance**
   - System continues operating during network partitions

3. **Eventual Consistency**
   - After partitions heal, all nodes have consistent data

## Usage

### Running Tests

To build and run the property tests:

```bash
cd /path/to/JadeVectorDB/property-tests
mkdir build && cd build
cmake ..
make
./property_tests
```

Or using the custom target:
```bash
make run_property_tests
```

### Creating New Properties

To add new properties:

1. Define the property function in the appropriate header file
2. Create a generator for the test data if needed
3. Add a test case to the relevant test file
4. Include the test file in `test_property_suite.cpp`

Example property definition:
```cpp
// Property: Example property that validates a condition
inline bool example_property(const std::vector<float>& vector) {
    // Define the invariant that should hold
    return !vector.empty() && vector.size() > 0;
}
```

Example test case:
```cpp
TEST_F(ExamplePropertyTest, ExampleProperty) {
    ArbitraryVectorGenerator gen(5, 20);  // 5-20 dimensions
    PropertyTest<std::vector<float>> test(
        "Example Property",
        example_property,
        &gen,
        50  // Run 50 tests
    );
    
    test.run();
}
```

## Examples

The framework includes several example test files demonstrating different property types:

- `vector-space/test_vector_space_properties.cpp` - Mathematical properties of vector operations
- `consistency/test_consistency_properties.cpp` - Database consistency properties
- `concurrency/test_concurrency_properties.cpp` - Thread safety properties
- `distributed-system/test_distributed_invariants.cpp` - Distributed system properties

## Integration with Existing Tests

This property-based testing framework complements the existing test infrastructure:

- Uses the same Google Test framework as other tests
- Can run alongside unit and integration tests
- Follows the same build and CI/CD processes
- Supports the same code coverage tools

## Best Practices

1. **Start Simple**: Begin with basic properties and gradually add complexity
2. **Focus on Invariants**: Test properties that should always hold true
3. **Use Appropriate Generators**: Create generators that produce meaningful test data
4. **Aim for Coverage**: Design properties that cover different aspects of the system
5. **Balance Performance**: Property tests can run many iterations, so consider performance

## Troubleshooting

### Common Issues

1. **Floating Point Comparisons**: Use appropriate epsilon values for floating point comparisons
2. **Generator State**: Make sure generators are properly initialized with random seeds
3. **Performance**: Some properties may be computationally expensive; adjust test counts accordingly

### Debugging Property Failures

When a property fails:
1. The framework will report the specific input that caused the failure
2. Use this input to create a targeted unit test
3. Fix the underlying issue
4. Verify the property passes with the corrected implementation

## Security Considerations

- Property tests use randomly generated inputs, which can help identify edge cases
- Tests should not expose sensitive system internals
- Generated test data is kept within the test environment