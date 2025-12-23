# JadeVectorDB CLI Testing

Comprehensive testing suite for all JadeVectorDB CLI functionality including:
- Basic CLI operations (Python & Shell)
- Persistence testing
- RBAC/Authorization testing
- **Phase 16 Features**: User management and bulk import/export

## Quick Start

### 1. Start the Server

```bash
cd backend/build
./jadevectordb
```

### 2. Run All Tests

From project root:

```bash
# Recommended: Use the master test runner
./tests/run_all_tests.sh

# Or run the Python test suite directly
python3 tests/run_cli_tests.py

# Or use the simple wrapper
./tests/run_tests.sh
```

## Test Data Configuration

All test data is centralized in `test_data.json`. This includes:

- **Server configuration**: URL and timeouts
- **Authentication**: Test user credentials
- **Databases**: Test database configurations for Python and Shell CLIs
- **Vectors**: Test vector data with auto-generated values
- **Search**: Search query configuration

### Modifying Test Data

Edit `tests/test_data.json`:

```json
{
  "auth": {
    "test_user": {
      "username": "cli_test_user",
      "password": "CliTest123@",
      "email": "cli_test@jadevectordb.com"
    }
  },
  "databases": {
    "python_test": {
      "name": "python_test_db",
      "dimension": 128,
      "index_type": "hnsw"
    }
  }
}
```

**Note**: Passwords must meet security requirements:
- At least 8 characters
- Contains uppercase letter
- Contains lowercase letter
- Contains digit
- Contains special character

## Test Output

The test runner outputs results in a clean table format:

```
================================================================================
#     Tool            Test                           Result
================================================================================
1     Python CLI      Health Check                   ✓ PASS
2     Python CLI      Status Check                   ✓ PASS
3     Python CLI      List Databases                 ✓ PASS
4     Python CLI      Create Database                ✓ PASS
5     Python CLI      Get Database                   ✓ PASS
6     Python CLI      Store Vector                   ✓ PASS
7     Python CLI      Search Vectors                 ✗ FAIL
8     Shell CLI       Health Check                   ✓ PASS
...
================================================================================

Summary: 11/12 tests passed
  Failed: 1
  Skipped: 0
```

### Result Indicators

- ✓ **PASS** - Test completed successfully
- ✗ **FAIL** - Test failed, see hints below for troubleshooting
- ⊘ **SKIP** - Test skipped (usually due to dependency failure)

## Troubleshooting

### Server Not Running

```
❌ Server is not running at http://localhost:8080

Please start the server first:
  cd backend/build && ./jadevectordb
```

**Solution**: Start the JadeVectorDB server before running tests.

### Authentication Failed

```
❌ Authentication setup failed
```

**Common causes**:
1. Server is not running
2. Password doesn't meet security requirements
3. User already exists with different password

**Solution**:
- Restart the server (user data is in-memory)
- Update test_data.json with correct password format

### Test Failures with Hints

The test runner provides specific hints for each failure:

```
[Test #6] Python CLI - Store Vector:
  • Ensure database was created successfully
  • Verify vector dimensions match database configuration
  • Check that vector ID is unique
```

### Viewing Server Logs

Check server logs for detailed error messages:

```bash
# If server is running in foreground
# Check the terminal output

# If server is running in background
tail -f /tmp/srv.log  # or wherever you redirected output
```

## Test Coverage

The test suite includes **36 comprehensive tests** covering:

### Basic CLI Tests (Tests 1-12)

**Python CLI (Tests 1-7):**
1. **Health Check** - Verify server is responding
2. **Status Check** - Get system status (requires auth)
3. **List Databases** - List all databases
4. **Create Database** - Create a test database
5. **Get Database** - Retrieve database details
6. **Store Vector** - Store a test vector
7. **Search Vectors** - Perform similarity search

**Shell CLI (Tests 8-12):**
8. **Health Check** - Verify server is responding
9. **Status Check** - Get system status (requires auth)
10. **List Databases** - List all databases
11. **Create Database** - Create a test database
12. **Get Database** - Retrieve database details

### Persistence Tests (Tests 13-15)

13. **User Login After Operations** - Verify user persistence
14. **Databases Persist** - Verify database metadata persistence
15. **New User Persists** - Verify new user registration persists

### RBAC Tests (Tests 16-20)

16-20. **Role-Based Access Control** - Verify permission system

### Phase 16 Tests - Python User Management (Tests 21-26)

21. **Add User** - Create new user via Python CLI
22. **List Users** - Display all users
23. **Show User** - Display user details
24. **Deactivate User** - Disable user account
25. **Activate User** - Re-enable user account
26. **Delete User** - Remove user from system

### Phase 16 Tests - Python Import/Export (Tests 27-28)

27. **Export Vectors** - Export vectors to JSON file (Python CLI)
28. **Import Vectors** - Import vectors from JSON file (Python CLI)

### Phase 16 Tests - Shell User Management (Tests 29-34)

29. **Add User** - Create new user via Shell CLI
30. **List Users** - Display all users
31. **Show User** - Display user details
32. **Deactivate User** - Disable user account
33. **Activate User** - Re-enable user account
34. **Delete User** - Remove user from system

### Phase 16 Tests - Shell Import/Export (Tests 35-36)

35. **Export Vectors** - Export vectors to JSON file (Shell CLI)
36. **Import Vectors** - Import vectors from JSON file (Shell CLI)

### JavaScript CLI Tests

JavaScript CLI tests use Jest and are not part of the unified test suite. If needed, they can be run separately:
```bash
cd cli/js
npm test
```

## Advanced Usage

### Running Specific Tests

Currently, the test runner runs all tests. To run specific tests, modify `run_cli_tests.py`:

```python
# Comment out test methods you don't want to run
def run_all_tests(self):
    # self.run_python_cli_tests()  # Skip Python tests
    self.run_shell_cli_tests()     # Run only Shell tests
```

### Changing Server URL

Update `test_data.json`:

```json
{
  "server": {
    "url": "http://your-server:8080"
  }
}
```

### Adding New Tests

1. Add test data to `test_data.json` if needed
2. Add test method to appropriate section in `run_cli_tests.py`:

```python
def run_python_cli_tests(self):
    # ... existing tests ...

    # New test
    success, output = self.run_python_cli_test('My New Test', [
        '--api-key', self.token, 'my-command', '--arg', 'value'
    ])
    self.results.append([8, "Python CLI", "My New Test",
                       "PASS" if success else "FAIL"])
```

3. Optionally add failure hints in `get_failure_hints()`

## Integration with CI/CD

The test runner returns exit code 0 for success and 1 for failure, making it suitable for CI/CD pipelines:

```bash
# In CI/CD script
./tests/run_tests.sh
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Tests failed!"
    exit 1
fi
```

## Files

- `test_data.json` - Centralized test data configuration
- `run_cli_tests.py` - Main test runner (Python)
- `run_tests.sh` - Shell wrapper for convenience
- `README.md` - This file

## Requirements

- Python 3.8+
- `requests` library (`pip install requests`)
- Bash (for shell CLI tests)
- JadeVectorDB server running

## Support

For issues or questions:
- Check the troubleshooting section above
- Review server logs for detailed errors
- See main documentation: `../README.md`
- See CLI documentation: `../cli/README.md`
