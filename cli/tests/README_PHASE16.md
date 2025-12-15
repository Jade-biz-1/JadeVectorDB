# JadeVectorDB CLI Test Suite - Phase 16 Enhancements

Comprehensive test suite for Phase 16 CLI enhancements covering user management, bulk import/export, and output formats across Python, Shell, and JavaScript CLIs.

## Test Overview

### T264: User Management CLI Tests ✅
Tests for user management commands (add, list, show, update, delete, activate, deactivate)

**Files:**
- `test_user_management.py` - Python pytest tests with mocks (35+ tests)
- `test_user_management.sh` - Shell script tests (23 tests)
- `user-management.test.js` - JavaScript Jest tests (30+ tests)

**Coverage:**
- ✅ User creation with/without password
- ✅ User listing with role/status filters
- ✅ User details retrieval
- ✅ User updates (role, status)
- ✅ User deletion
- ✅ User activation/deactivation
- ✅ Complete user lifecycle workflow
- ✅ Error handling (already exists, not found, unauthorized)
- ✅ Output format support (JSON, YAML, Table, CSV)
- ✅ cURL command generation
- ✅ API endpoint consistency

### T268: Import/Export CLI Tests ✅
Tests for bulk import/export functionality

**Files:**
- `test_import_export.py` - Python pytest tests (30+ tests)
- `test_import_export.sh` - Shell script tests (17 tests)

**Coverage:**
- ✅ JSON file import/export
- ✅ CSV file import/export
- ✅ Progress tracking callbacks
- ✅ Batch processing (configurable batch sizes)
- ✅ Error handling (file not found, malformed JSON, partial failures)
- ✅ Large dataset handling (1000+ vectors)
- ✅ Empty dataset handling
- ✅ Vector filtering during export
- ✅ Metadata preservation
- ✅ Round-trip export/import workflow
- ✅ Performance with various dataset sizes

### T272: CLI Integration Tests ✅
Comprehensive cross-CLI consistency and workflow tests

**Files:**
- `test_cli_integration.py` - Python integration tests (25+ tests)
- `test_cli_integration.sh` - Shell cross-CLI tests (11 tests)

**Coverage:**
- ✅ Cross-CLI command consistency
- ✅ Output format compatibility
- ✅ API endpoint consistency
- ✅ Authentication handling
- ✅ Error handling consistency
- ✅ Complete user lifecycle workflow
- ✅ Complete import/export workflow
- ✅ Database and vector workflow
- ✅ Performance and scalability tests
- ✅ Permission and authorization tests

## Coverage Summary

| Test Category | Python CLI | Shell CLI | JavaScript CLI | Total Tests |
|--------------|------------|-----------|----------------|-------------|
| User Management | 35 tests | 23 tests | 30 tests | **88 tests** |
| Import/Export | 30 tests | 17 tests | N/A | **47 tests** |
| Integration | 25 tests | 11 tests | N/A | **36 tests** |
| **TOTAL** | **90 tests** | **51 tests** | **30 tests** | **171 tests** |

## Running the Tests

### Prerequisites

**Python Tests:**
```bash
pip install pytest pytest-mock
cd cli/tests
```

**JavaScript Tests:**
```bash
cd cli/js
npm install  # Installs jest
```

**Shell Tests:**
```bash
# Optional: install jq, yq
sudo apt-get install jq yq
```

### Running Individual Test Suites

**Python:**
```bash
pytest cli/tests/test_user_management.py -v
pytest cli/tests/test_import_export.py -v
pytest cli/tests/test_cli_integration.py -v
```

**Shell:**
```bash
bash cli/tests/test_user_management.sh
bash cli/tests/test_import_export.sh
bash cli/tests/test_cli_integration.sh
```

**JavaScript:**
```bash
cd cli/js
npm test -- user-management.test.js
```

### Run All Phase 16 Tests

```bash
# From project root
pytest cli/tests/test_user_management.py cli/tests/test_import_export.py cli/tests/test_cli_integration.py -v
bash cli/tests/test_user_management.sh
bash cli/tests/test_import_export.sh
bash cli/tests/test_cli_integration.sh
cd cli/js && npm test
```

See README.md for original CLI tests documentation.
