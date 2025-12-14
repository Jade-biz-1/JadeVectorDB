# JadeVectorDB CLI Test Suite

Automated test suite for validating all CLI implementations and the cURL command generation feature.

## Overview

This directory contains test scripts that verify the functionality of:
- Python CLI operations
- Shell CLI operations
- JavaScript CLI operations
- cURL command generation for all operations
- API endpoint correctness

## Test Files

| File | Purpose | Type |
|------|---------|------|
| `test_curl_generation.py` | Tests cURL command generation in Python CLI | Python (pytest) |
| `test_cli_curl.sh` | Tests Shell CLI and cURL generation | Bash script |

## Prerequisites

### For Python Tests

```bash
# Install Python dependencies
pip install pytest requests

# Or install from requirements
cd cli/python
pip install -e .[dev]
```

### For Shell Tests

```bash
# Requires bash, curl, and jq
# On Ubuntu/Debian:
sudo apt-get install curl jq

# On macOS:
brew install curl jq
```

## Running Tests

### Run All Tests

```bash
# From repository root
cd cli/tests

# Run Python tests
pytest test_curl_generation.py -v

# Run shell tests
bash test_cli_curl.sh
```

### Run Specific Test Categories

#### Test cURL Generation Only

```bash
pytest test_curl_generation.py::TestCurlGeneration -v
```

#### Test Specific Operations

```bash
# Test database operations
pytest test_curl_generation.py::TestCurlGeneration::test_create_database_curl -v

# Test vector operations
pytest test_curl_generation.py::TestCurlGeneration::test_store_vector_curl -v

# Test search operations
pytest test_curl_generation.py::TestCurlGeneration::test_search_curl -v
```

## Test Coverage

### Python CLI Tests (`test_curl_generation.py`)

**Coverage:**
- ✅ Database creation cURL generation
- ✅ Database listing cURL generation
- ✅ Vector storage cURL generation
- ✅ Vector retrieval cURL generation
- ✅ Vector deletion cURL generation
- ✅ Similarity search cURL generation
- ✅ Health check cURL generation
- ✅ Status check cURL generation
- ✅ Metadata handling in cURL commands
- ✅ JSON escaping and formatting

**Example Test:**
```python
def test_create_database_curl(self):
    """Test cURL generation for database creation"""
    args = argparse.Namespace(
        url='http://localhost:8080',
        api_key='test-key',
        name='test-db',
        description='Test database',
        dimension=768,
        index_type='HNSW',
        curl_only=True
    )

    # Capture cURL command
    output = capture_curl_command(create_database, args)

    # Verify cURL command format
    assert 'curl -X POST' in output
    assert 'http://localhost:8080/api/v1/databases' in output
    assert '"name":"test-db"' in output
```

### Shell CLI Tests (`test_cli_curl.sh`)

**Coverage:**
- ✅ Shell script execution
- ✅ cURL command generation from shell
- ✅ Parameter passing
- ✅ JSON formatting
- ✅ Error handling
- ✅ Exit codes

**Test Sections:**
1. Basic cURL generation
2. Database operations
3. Vector operations
4. Search operations
5. System operations

## Test Execution Examples

### Successful Test Run

```bash
$ pytest test_curl_generation.py -v

test_curl_generation.py::TestCurlGeneration::test_create_database_curl PASSED   [12%]
test_curl_generation.py::TestCurlGeneration::test_list_databases_curl PASSED    [25%]
test_curl_generation.py::TestCurlGeneration::test_store_vector_curl PASSED      [37%]
test_curl_generation.py::TestCurlGeneration::test_retrieve_vector_curl PASSED   [50%]
test_curl_generation.py::TestCurlGeneration::test_delete_vector_curl PASSED     [62%]
test_curl_generation.py::TestCurlGeneration::test_search_curl PASSED            [75%]
test_curl_generation.py::TestCurlGeneration::test_health_curl PASSED            [87%]
test_curl_generation.py::TestCurlGeneration::test_status_curl PASSED            [100%]

=============================== 8 passed in 0.25s ===============================
```

### Shell Test Output

```bash
$ bash test_cli_curl.sh

Testing JadeVectorDB CLI cURL Generation
=========================================

[✓] Test 1: Create database cURL generation
[✓] Test 2: List databases cURL generation
[✓] Test 3: Store vector cURL generation
[✓] Test 4: Search cURL generation
[✓] Test 5: Health check cURL generation

All tests passed! ✓
```

## Writing New Tests

### Python Test Template

```python
import pytest
from jadevectordb.cli import <command_function>
import argparse

class TestNewFeature:
    def test_new_operation_curl(self):
        """Test cURL generation for new operation"""
        args = argparse.Namespace(
            url='http://localhost:8080',
            api_key='test-key',
            # Add operation-specific parameters
            curl_only=True
        )

        # Test the cURL generation
        output = capture_curl_command(<command_function>, args)

        # Assertions
        assert 'curl' in output
        assert 'http://localhost:8080' in output
        assert '<expected-content>' in output
```

### Shell Test Template

```bash
#!/bin/bash
# test_new_feature.sh

set -e

# Test new operation
echo "Testing new operation..."
OUTPUT=$(bash cli/shell/scripts/jade-db.sh --curl-only \
  --url http://localhost:8080 \
  new-command --params)

# Verify output
if echo "$OUTPUT" | grep -q "expected-pattern"; then
  echo "[✓] Test passed"
else
  echo "[✗] Test failed"
  exit 1
fi
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: CLI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install pytest requests
        pip install -e cli/python

    - name: Run Python tests
      run: |
        cd cli/tests
        pytest test_curl_generation.py -v

    - name: Install shell dependencies
      run: sudo apt-get install -y curl jq

    - name: Run shell tests
      run: |
        cd cli/tests
        bash test_cli_curl.sh
```

## Test Data

### Sample Vectors

```json
{
  "id": "test-vector-1",
  "values": [0.1, 0.2, 0.3, 0.4, 0.5],
  "metadata": {
    "category": "test",
    "source": "automated-test"
  }
}
```

### Sample Database Configuration

```json
{
  "name": "test-database",
  "description": "Database for automated testing",
  "dimension": 768,
  "index_type": "HNSW"
}
```

## Troubleshooting

### Tests Fail with Connection Error

The tests don't require a running JadeVectorDB server - they only test cURL command generation, not actual API calls.

If you see connection errors:
```bash
# Ensure you're testing cURL generation only
pytest test_curl_generation.py --cov=cli -v
```

### Import Errors

```bash
# Install the Python CLI package in development mode
cd cli/python
pip install -e .

# Then run tests
cd ../tests
pytest test_curl_generation.py
```

### Shell Tests Fail

```bash
# Verify dependencies
which curl jq bash

# Make test script executable
chmod +x cli/tests/test_cli_curl.sh

# Run with verbose output
bash -x cli/tests/test_cli_curl.sh
```

## Test Coverage Reports

### Generate Python Coverage Report

```bash
cd cli/tests
pytest test_curl_generation.py --cov=../python/jadevectordb --cov-report=html

# Open htmlcov/index.html in browser
```

### Expected Coverage

- **cURL Generator**: 95%+ coverage
- **CLI Commands**: 85%+ coverage
- **Client Library**: 90%+ coverage

## Best Practices

1. **Test cURL Generation** - Verify commands are syntactically correct
2. **Test All Parameters** - Ensure all command parameters generate correct cURL
3. **Test Edge Cases** - Special characters, empty values, large inputs
4. **Verify JSON Formatting** - Proper escaping and structure
5. **Check HTTP Methods** - GET, POST, DELETE used correctly
6. **Validate Headers** - Content-Type, Authorization headers present

## Future Enhancements

- [ ] Add integration tests with actual server
- [ ] Add performance benchmarks for CLI operations
- [ ] Add tests for JavaScript CLI
- [ ] Add tests for distributed CLI operations
- [ ] Add end-to-end workflow tests
- [ ] Add API compatibility tests

## Related Documentation

- [Python CLI Documentation](../python/README.md)
- [Shell CLI Documentation](../shell/README.md)
- [JavaScript CLI Documentation](../js/README.md)
- [CLI Examples](../../examples/cli/README.md)
- [Main CLI README](../README.md)

## Support

For test-related issues:
- Check test output carefully
- Verify all dependencies are installed
- See related documentation above
- GitHub Issues: [JadeVectorDB Issues](https://github.com/Jade-biz-1/JadeVectorDB/issues)
