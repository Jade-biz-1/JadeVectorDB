# Coverage Measurement Setup

This document explains how to set up and run code coverage measurements for the JadeVectorDB project.

## Prerequisites

Before you can run coverage measurements, you need to install the necessary tools:

```bash
./scripts/setup-coverage-tools.sh
```

This script will install `lcov` which is required for generating code coverage reports.

## Building with Coverage Enabled

To build the project with coverage instrumentation enabled, follow these steps:

```bash
cd backend
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_COVERAGE=ON ..
make
```

## Running Coverage Analysis

Once built with coverage enabled, you can run the coverage analysis:

```bash
make coverage
```

This will:
1. Zero all coverage counters
2. Run all tests
3. Capture coverage data
4. Generate an HTML report in the `coverage_report` directory

## Excluding Files from Coverage

The coverage script automatically excludes:
- System directories (`/usr/*`)
- Test files (`*/tests/*`, `*_test*`, `test_*`)
- This helps focus the coverage report on the actual production code

## Viewing the Report

After running `make coverage`, open the generated report:

```bash
open coverage_report/index.html
```

Or navigate to the `coverage_report/index.html` file in your browser.

## Troubleshooting

If you encounter issues:
- Ensure you're building with Debug configuration (`-DCMAKE_BUILD_TYPE=Debug`)
- Make sure all test executables pass before running coverage
- Check that `lcov` and `genhtml` are properly installed

## Integration with Test Enhancement Task

This coverage setup is part of the broader Test Coverage Enhancement (T185) task, which aims to:
- Implement comprehensive test cases for all services
- Set up code coverage measurement
- Achieve 90%+ coverage target

With this infrastructure in place, we can now:
1. Run tests and measure current coverage
2. Identify areas that need more tests
3. Work toward the 90%+ coverage target