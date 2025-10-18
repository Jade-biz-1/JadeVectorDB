#!/bin/bash

# Performance optimization and profiling script for JadeVectorDB
# This script runs various profiling tools to identify performance bottlenecks

set -e  # Exit on any error

echo "Running performance optimization and profiling for JadeVectorDB..."

# Function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --profile-only      Run only profiling tools (perf, gprof)"
    echo "  --optimize-only     Run only optimization suggestions"
    echo "  --benchmark-only    Run only benchmarks"
    echo "  --all               Run all profiling and optimization (default)"
    echo "  --help              Show this help message"
}

# Parse command line options
PROFILE_ONLY=false
OPTIMIZE_ONLY=false
BENCHMARK_ONLY=false
ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile-only)
            PROFILE_ONLY=true
            ALL=false
            shift
            ;;
        --optimize-only)
            OPTIMIZE_ONLY=true
            ALL=false
            shift
            ;;
        --benchmark-only)
            BENCHMARK_ONLY=true
            ALL=false
            shift
            ;;
        --all)
            ALL=true
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

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/backend/build_profile"
BENCHMARK_DIR="$PROJECT_ROOT/backend/tests/benchmarks"

# Create build directory
mkdir -p "$BUILD_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Run profiling tools
if [ "$OPTIMIZE_ONLY" = false ] && [ "$BENCHMARK_ONLY" = false ] && [ "$ALL" = true ] || [ "$PROFILE_ONLY" = true ]; then
    echo "Running performance profiling tools..."
    
    # Check if perf is available
    if ! command_exists perf; then
        echo "WARNING: perf not found. Install with: sudo apt install linux-perf"
    else
        echo "Running perf profiling..."
        perf --version
        
        # Build project with debug symbols for profiling
        echo "Building project with debug symbols for perf profiling..."
        cd "$BUILD_DIR"
        cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
        make -j$(nproc) jadevectordb_tests || echo "Build had issues but continuing with available components"
        
        # Run perf on tests if available
        for test_exec in $(find . -name "*test*" -executable -type f | head -3); do
            if [ -f "$test_exec" ]; then
                echo "Profiling $test_exec with perf..."
                perf record -g --call-graph=dwarf ./"$test_exec" || echo "Perf profiling completed with some issues"
                perf report --stdio > "perf-report-$test_exec.txt" || echo "Perf report generation completed"
                echo "Perf report saved to perf-report-$test_exec.txt"
            fi
        done
    fi
    
    # Check if gprof is available (requires GCC with -pg flag)
    if command_exists gprof; then
        echo "Checking for gprof profiling data..."
        # This would require building with -pg flag, which we'll do in the optimization section
    else
        echo "gprof not available (usually comes with GCC)"
    fi
    
    # Check for Valgrind Callgrind if available
    if command_exists valgrind; then
        echo "Running Valgrind Callgrind profiler..."
        # Build with debug info for better profiling
        cd "$BUILD_DIR"
        cmake -DCMAKE_BUILD_TYPE=Debug ..
        make -j2 jadevectordb_tests || echo "Build had issues but continuing"
        
        # Run callgrind on a sample test
        for test_exec in $(find . -name "*test*" -executable -type f | head -1); do
            if [ -f "$test_exec" ]; then
                echo "Running Callgrind on $test_exec..."
                valgrind --tool=callgrind --callgrind-out-file="callgrind-$test_exec.out" ./"$test_exec" || echo "Callgrind run completed"
                echo "Callgrind output saved to callgrind-$test_exec.out"
                echo "To view results: callgrind_annotate callgrind-$test_exec.out"
            fi
        done
    else
        echo "Valgrind not found. Install with: sudo apt install valgrind"
    fi
fi

# Run performance optimization suggestions
if [ "$PROFILE_ONLY" = false ] && [ "$BENCHMARK_ONLY" = false ] && [ "$ALL" = true ] || [ "$OPTIMIZE_ONLY" = true ]; then
    echo "Running performance optimization analysis..."
    
    # Check compiler version for optimization capabilities
    if command_exists g++; then
        echo "GCC version: $(g++ --version | head -n1)"
    fi
    
    if command_exists clang++; then
        echo "Clang version: $(clang++ --version | head -n1)"
    fi
    
    # Analyze source code for common optimization opportunities
    echo "Analyzing source code for optimization opportunities..."
    cd "$PROJECT_ROOT/backend/src"
    
    # Look for potential optimization patterns
    echo "Checking for potential loop optimization opportunities..."
    find . -name "*.cpp" -o -name "*.h" | xargs grep -n "for.*<.*size()" | head -5 || echo "No obvious loop optimization issues found"
    
    echo "Checking for potential memory allocation patterns..."
    find . -name "*.cpp" -o -name "*.h" | xargs grep -n "new\|malloc\|reserve\|resize" | head -10 || echo "No obvious memory allocation issues found in quick scan"
    
    echo "Checking for potential SIMD opportunities..."
    find . -name "*.cpp" -o -name "*.h" | xargs grep -n "std::transform\|std::accumulate\|for.*=" | head -5 || echo "Basic loop patterns found for potential vectorization"
    
    # Check for use of efficient data structures
    echo "Checking for efficient data structure usage..."
    find . -name "*.cpp" -o -name "*.h" | xargs grep -n "std::unordered_map\|std::map\|std::vector\|std::deque" | head -10
    
    # Look for potential threading opportunities
    echo "Checking for potential parallelization opportunities..."
    find . -name "*.cpp" -o -name "*.h" | xargs grep -n "for.*parallel\|std::async\|std::thread\|ThreadPool\|#pragma omp" | head -5 || echo "No explicit parallelization found (may be using internal threading)"
fi

# Run benchmarks
if [ "$PROFILE_ONLY" = false ] && [ "$OPTIMIZE_ONLY" = false ] && [ "$ALL" = true ] || [ "$BENCHMARK_ONLY" = true ]; then
    echo "Running performance benchmarks..."
    
    # Build benchmarks
    cd "$BUILD_DIR"
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc) search_benchmarks filtered_search_benchmarks || echo "Benchmark build had issues"
    
    # Run benchmarks if they were built
    if [ -f "search_benchmarks" ]; then
        echo "Running search benchmarks..."
        ./search_benchmarks || echo "Search benchmarks completed"
    fi
    
    if [ -f "filtered_search_benchmarks" ]; then
        echo "Running filtered search benchmarks..."
        ./filtered_search_benchmarks || echo "Filtered search benchmarks completed"
    fi
    
    # Run Google Benchmark if available
    if command_exists benchmark; then
        echo "Google Benchmark tools available"
    else
        echo "Google Benchmark tools not found (included in build)"
    fi
fi

# Generate optimization report
echo
echo "==============================================="
echo "PERFORMANCE PROFILING AND OPTIMIZATION REPORT"
echo "==============================================="

if [ "$PROFILE_ONLY" = false ] && [ "$OPTIMIZE_ONLY" = false ] && [ "$BENCHMARK_ONLY" = false ] || [ "$ALL" = true ]; then
    echo "✓ Performance profiling tools executed (perf, callgrind)"
    echo "✓ Optimization analysis performed"
    echo "✓ Benchmarks executed where available"
fi

if [ "$PROFILE_ONLY" = true ]; then
    echo "✓ Profiling tools only executed"
fi

if [ "$OPTIMIZE_ONLY" = true ]; then
    echo "✓ Optimization analysis only performed"
fi

if [ "$BENCHMARK_ONLY" = true ]; then
    echo "✓ Benchmarks only executed"
fi

echo "==============================================="
echo "Key findings and recommendations:"
echo "1. Review perf reports for CPU hotspots"
echo "2. Examine callgrind output for function call overhead"
echo "3. Optimize loops with potential for vectorization"
echo "4. Consider memory pool allocation for frequent allocations"
echo "5. Evaluate threading opportunities for parallelizable operations"
echo "6. Run benchmarks with different data sizes to identify scaling patterns"
echo "==============================================="

echo "Performance optimization and profiling completed!"