#!/bin/bash

# Script to generate comprehensive test coverage report for JadeVectorDB
# This script builds the project with coverage enabled, runs all tests, and generates a coverage report

set -e  # Exit on any error

echo "Generating comprehensive test coverage report for JadeVectorDB..."

# Function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build-only    Only build the project with coverage enabled (don't run tests)"
    echo "  --run-tests     Only run tests (don't rebuild)"
    echo "  --generate-report  Only generate the report from existing coverage data"
    echo "  --help          Show this help message"
}

# Parse command line options
BUILD_ONLY=false
RUN_TESTS_ONLY=false
GENERATE_REPORT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --run-tests)
            RUN_TESTS_ONLY=true
            shift
            ;;
        --generate-report)
            GENERATE_REPORT_ONLY=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if lcov is available
if ! command -v lcov &> /dev/null; then
    echo "lcov could not be found. Please install lcov:"
    echo "  Ubuntu/Debian: sudo apt-get install lcov"
    echo "  CentOS/RHEL: sudo yum install lcov"
    exit 1
fi

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/backend/build_coverage"
COVERAGE_INFO="$BUILD_DIR/coverage.info"
COVERAGE_REPORT_DIR="$BUILD_DIR/coverage_report"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Build the project with coverage enabled if not in run-tests or generate-report mode
if [ "$RUN_TESTS_ONLY" = false ] && [ "$GENERATE_REPORT_ONLY" = false ]; then
    echo "Building project with coverage enabled..."
    cd "$BUILD_DIR"
    cmake -DBUILD_COVERAGE=ON ..
    make -j$(nproc)
    echo "Build completed successfully."
fi

# Only run tests if not in build-only or generate-report-only mode
if [ "$BUILD_ONLY" = false ] && [ "$GENERATE_REPORT_ONLY" = false ]; then
    echo "Initializing coverage counters..."
    lcov --directory . --zerocounters

    echo "Running all tests to collect coverage data..."
    
    # Run all test executables in the build directory
    for test_executable in $(find . -name "jadevectordb_tests*" -type f -executable); do
        echo "Running $test_executable..."
        "$test_executable" || echo "Test $test_executable failed, continuing with others..."
    done
    
    # If benchmark executables exist, run them too (though they may not be typical unit tests)
    for bench_executable in $(find . -name "*benchmarks" -type f -executable); do
        echo "Running benchmark $bench_executable for coverage..."
        timeout 30s "$bench_executable" || echo "Benchmark $bench_executable timed out or failed, continuing..."
    done

    echo "Capturing coverage data..."
    lcov --directory . --capture --output-file "$COVERAGE_INFO"

    echo "Removing system and test files from coverage report..."
    lcov --remove "$COVERAGE_INFO" '/usr/*' '*/tests/*' '*/test_*' '*_test*' '*/build/*' \
         '*/external/*' '*/third_party/*' --output-file "$COVERAGE_INFO"
fi

# Only generate report if not in build-only or run-tests-only mode
if [ "$BUILD_ONLY" = false ] && [ "$RUN_TESTS_ONLY" = false ]; then
    echo "Generating HTML coverage report..."
    genhtml "$COVERAGE_INFO" --output-directory "$COVERAGE_REPORT_DIR"

    # Calculate overall coverage percentage
    total_lines=$(lcov --list "$COVERAGE_INFO" 2>/dev/null | tail -n +5 | head -n -3 | grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9.]+%' | awk '{sum+=$2} END{print sum+0}')
    covered_lines=$(lcov --list "$COVERAGE_INFO" 2>/dev/null | tail -n +5 | head -n -3 | grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9.]+%' | awk '{sum+=$3} END{print sum+0}')

    if [ -n "$total_lines" ] && [ "$total_lines" -gt 0 ]; then
        coverage_percentage=$(echo "scale=2; $covered_lines * 100 / $total_lines" | bc)
        echo "Overall coverage: $coverage_percentage% ($covered_lines of $total_lines lines)"
    else
        echo "Could not calculate overall coverage percentage"
    fi

    echo "Coverage report generated successfully at: $COVERAGE_REPORT_DIR/index.html"
    echo "You can open this in your browser to view detailed coverage information."
fi

echo "Coverage report generation completed!"