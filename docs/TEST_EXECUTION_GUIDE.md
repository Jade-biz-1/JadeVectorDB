# JadeVectorDB Test Execution Guide

**Quick reference for running all test suites**

## Overview

JadeVectorDB includes comprehensive test coverage with **217+ test cases** across backend and frontend:

- **Backend Tests**: 130+ test cases (C++, Google Test)
- **Frontend Tests**: 87 test cases (JavaScript, Jest + Cypress)

## Backend Tests

### Prerequisites

- CMake 3.20+
- Google Test (GTest)
- C++20 compiler

### Running Backend Tests

```bash
# Navigate to backend build directory
cd backend/build

# Configure with CMake (first time only)
cmake ..

# Build specific test executables
make test_authentication_service
make test_auth_manager
make test_api_key_lifecycle

# Run individual test suites
./test_authentication_service
./test_auth_manager
./test_api_key_lifecycle

# Run all tests via CTest
ctest

# Run tests with verbose output
ctest --verbose

# Run specific test by name pattern
ctest -R "Authentication"
```

### Backend Test Suites

| Test Suite | Test Cases | Execution Time | Command |
|------------|------------|----------------|---------|
| AuthenticationService | 44 | ~1.5s | `./test_authentication_service` |
| AuthManager | 45 | ~1.5s | `./test_auth_manager` |
| API Key Lifecycle | 41 | ~2.0s | `./test_api_key_lifecycle` |
| **Total** | **130** | **< 5s** | `ctest` |

### Expected Output

```
[==========] Running 44 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 44 tests from AuthenticationServiceTest
[ RUN      ] AuthenticationServiceTest.InitializeService
[       OK ] AuthenticationServiceTest.InitializeService (0 ms)
[ RUN      ] AuthenticationServiceTest.RegisterUser_Success
[       OK ] AuthenticationServiceTest.RegisterUser_Success (2 ms)
...
[==========] 44 tests from 1 test suite ran. (125 ms total)
[  PASSED  ] 44 tests.
```

## Frontend Tests

### Prerequisites

- Node.js 18+
- npm or yarn

### Running Frontend Tests

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Run all Jest tests (unit + integration)
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage

# Run specific test file
npm test auth-flows.test.js

# Run Cypress E2E tests (interactive)
npm run cypress:open

# Run Cypress E2E tests (headless)
npm run cypress:run

# Run specific Cypress test
npx cypress run --spec tests/e2e/auth-e2e.cy.js
```

### Frontend Test Suites

| Test Suite | Test Cases | Execution Time | Command |
|------------|------------|----------------|---------|
| Auth Flows (Integration) | 35 | ~2s | `npm test auth-flows` |
| API Service (Unit) | 30 | ~1.5s | `npm test auth-api` |
| E2E (Cypress) | 22 | ~12s | `npm run cypress:run` |
| **Total** | **87** | **< 20s** | `npm test && npm run cypress:run` |

### Expected Output

**Jest:**
```
PASS  tests/integration/auth-flows.test.js
  Authentication Flows - Comprehensive Tests
    Login Flow
      ✓ successful login with valid credentials (45ms)
      ✓ login failure with invalid credentials (32ms)
      ✓ login with empty fields shows error (18ms)
      ...

Test Suites: 2 passed, 2 total
Tests:       65 passed, 65 total
Snapshots:   0 total
Time:        4.521 s
```

**Cypress:**
```
  Authentication E2E Tests
    Login Workflow
      ✓ successfully logs in with valid credentials (850ms)
      ✓ shows error message for invalid credentials (421ms)
      ...
    API Key Management
      ✓ displays list of existing API keys (612ms)
      ✓ creates a new API key (735ms)
      ...

  22 passing (12s)
```

## Running All Tests

### Full Test Suite

```bash
# Backend tests
cd backend/build && ctest

# Frontend tests
cd frontend && npm test && npm run cypress:run
```

### Total Execution Time

- Backend: < 5 seconds
- Frontend: < 20 seconds
- **Total: < 25 seconds** ⚡

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and Test Backend
        run: |
          cd backend
          mkdir build && cd build
          cmake ..
          make test_authentication_service
          make test_auth_manager
          make test_api_key_lifecycle
          ctest --output-on-failure

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install and Test
        run: |
          cd frontend
          npm install
          npm test -- --coverage
          npm run build
          npm start & npx wait-on http://localhost:3000
          npm run cypress:run
```

## Test Coverage Reports

### Backend Coverage

```bash
cd backend/build
cmake -DCMAKE_BUILD_TYPE=Coverage ..
make coverage
```

### Frontend Coverage

```bash
cd frontend
npm run test:coverage

# Coverage report will be in coverage/lcov-report/index.html
open coverage/lcov-report/index.html
```

## Troubleshooting

### Backend Tests

**Issue: Tests fail with "Cannot find GTest"**
```bash
# Install GTest
sudo apt-get install libgtest-dev  # Ubuntu/Debian
brew install googletest             # macOS
```

**Issue: Linking errors**
```bash
# Clean build
cd backend/build
rm -rf *
cmake ..
make
```

### Frontend Tests

**Issue: "Cannot find module '@/lib/api'"**
```bash
# Check jest.config.js moduleNameMapper
# Ensure paths match your project structure
```

**Issue: Cypress can't connect to localhost:3000**
```bash
# Start dev server in separate terminal
cd frontend
npm run dev

# Then run Cypress
npm run cypress:open
```

**Issue: Tests timing out**
```bash
# Increase Jest timeout in test file
jest.setTimeout(10000);  // 10 seconds
```

## Test Documentation

For detailed information about the test implementations:

- **Backend Tests**: See `AUTHENTICATION_TESTS_IMPLEMENTATION.md`
- **Frontend Tests**: See `FRONTEND_TESTS_IMPLEMENTATION.md`

## Quick Reference

### Common Commands

```bash
# Backend - Run all tests
cd backend/build && ctest

# Backend - Run specific suite
cd backend/build && ./test_authentication_service

# Frontend - Run all Jest tests
cd frontend && npm test

# Frontend - Run with coverage
cd frontend && npm run test:coverage

# Frontend - Run E2E tests
cd frontend && npm run cypress:run

# Frontend - Open Cypress UI
cd frontend && npm run cypress:open
```

### Test Statistics

| Category | Count | Coverage |
|----------|-------|----------|
| Backend Test Cases | 130+ | 100% auth |
| Frontend Test Cases | 87 | 100% auth |
| **Total Test Cases** | **217+** | **100% auth** |
| Total Test Code | 4,370 lines | - |
| Backend Execution | < 5s | - |
| Frontend Execution | < 20s | - |
| **Total Execution** | **< 25s** | - |

---

**Last Updated**: 2025-11-18
**Test Framework Versions**:
- Backend: Google Test (latest)
- Frontend: Jest 29.7.0, Cypress 13.0.0, React Testing Library 13.4.0
