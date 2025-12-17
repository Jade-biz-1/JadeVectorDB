# Unit Testing Directory

This directory contains unit tests for the SQLite persistence layer implementation.

## Test Files

### Sprint 1.2: User, Group, and Role Management
- **test_sprint_1_2.cpp** - Tests for user CRUD operations, group management, role assignments
- **Executable**: `test_sprint_1_2`
- **Tests**: 25 test cases
- **Coverage**: User creation, group membership, role assignments, user-role and group-role relationships

### Sprint 1.3: Permission System, API Keys, Auth Tokens & Sessions
- **test_sprint_1_3.cpp** - Tests for permissions, API keys, authentication tokens, and sessions
- **Executable**: `test_sprint_1_3`
- **Tests**: 27 test cases
- **Coverage**: 
  - Permission management (list, get, grant, revoke, check)
  - API key lifecycle (create, retrieve, revoke, usage tracking)
  - Auth token management (create, invalidate, cleanup)
  - Session management (create, update activity, end, cleanup)

### Sprint 1.4: Database Metadata & Audit Logging
- **test_sprint_1_4.cpp** - Tests for database metadata persistence and audit logging
- **Executable**: `test_sprint_1_4`
- **Tests**: 24 test cases
- **Coverage**:
  - Database metadata CRUD operations
  - Database stats tracking (vector count, index count)
  - Audit event logging
  - Audit log querying with filters
  - Transaction support (rollback/commit)

### Legacy Tests
- **test_sqlite_persistence.cpp** - Initial SQLite persistence tests (superseded by sprint tests)
- **Executable**: `test_sqlite_persistence`

## Building and Running Tests

### Compile Individual Tests

```bash
cd /home/deepak/Public/JadeVectorDB/backend

# Sprint 1.2 test
g++ -std=c++20 -I. -I./src unittesting/test_sprint_1_2.cpp \
    build/CMakeFiles/jadevectordb_core.dir/src/services/sqlite_persistence_layer.cpp.o \
    build/CMakeFiles/jadevectordb_core.dir/src/lib/logging.cpp.o \
    build/CMakeFiles/jadevectordb_core.dir/src/lib/error_handling.cpp.o \
    -lsqlite3 -lpthread -o unittesting/test_sprint_1_2

# Sprint 1.3 test
g++ -std=c++20 -I. -I./src unittesting/test_sprint_1_3.cpp \
    build/CMakeFiles/jadevectordb_core.dir/src/services/sqlite_persistence_layer.cpp.o \
    build/CMakeFiles/jadevectordb_core.dir/src/lib/logging.cpp.o \
    build/CMakeFiles/jadevectordb_core.dir/src/lib/error_handling.cpp.o \
    -lsqlite3 -lpthread -o unittesting/test_sprint_1_3

# Sprint 1.4 test
g++ -std=c++20 -I. -I./src unittesting/test_sprint_1_4.cpp \
    build/CMakeFiles/jadevectordb_core.dir/src/services/sqlite_persistence_layer.cpp.o \
    build/CMakeFiles/jadevectordb_core.dir/src/lib/logging.cpp.o \
    build/CMakeFiles/jadevectordb_core.dir/src/lib/error_handling.cpp.o \
    -lsqlite3 -lpthread -o unittesting/test_sprint_1_4
```

### Run All Tests

```bash
cd /home/deepak/Public/JadeVectorDB/backend

# Run Sprint 1.2
./unittesting/test_sprint_1_2

# Run Sprint 1.3
./unittesting/test_sprint_1_3

# Run Sprint 1.4
./unittesting/test_sprint_1_4
```

### Run All Tests in Sequence

```bash
cd /home/deepak/Public/JadeVectorDB/backend
for test in unittesting/test_sprint_1_*; do
    echo "=== Running $(basename $test) ==="
    $test
    echo ""
done
```

## Test Database Locations

Each test creates a temporary database in `/tmp/jadevectordb_test_*`:
- Sprint 1.2: `/tmp/jadevectordb_test_sprint12/system.db`
- Sprint 1.3: `/tmp/jadevectordb_test_sprint13/system.db`
- Sprint 1.4: `/tmp/jadevectordb_test_sprint14/system.db`

Databases are recreated on each test run to ensure clean state.

## Test Results Summary

| Sprint | Tests | Status | Coverage |
|--------|-------|--------|----------|
| 1.2 | 25 | ✅ All Pass | User, Group, Role management |
| 1.3 | 27 | ✅ All Pass | Permissions, API Keys, Tokens, Sessions |
| 1.4 | 24 | ✅ All Pass | Database Metadata, Audit Logging |
| **Total** | **76** | **✅ All Pass** | **Complete RBAC persistence layer** |

## Adding New Tests

When adding new tests:

1. Create `test_sprint_X_Y.cpp` in this directory
2. Follow the existing test pattern (ASSERT macro, TEST sections)
3. Use unique temporary database path
4. Update this README with test description
5. Add compilation and run instructions

## Notes

- All tests use the same SQLitePersistenceLayer implementation
- Tests are independent and can run in any order
- Each test cleans up its temporary database on startup
- Build core library first with `./build.sh` before compiling tests
