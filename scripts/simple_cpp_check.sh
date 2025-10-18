#!/bin/bash

# Simple C++ Implementation Standard Compliance Check
# Basic verification of C++ standards compliance

set -e

PROJECT_ROOT="/home/deepak/Public/JadeVectorDB"
BACKEND_SRC="${PROJECT_ROOT}/backend/src"
DOCS_DIR="${PROJECT_ROOT}/docs"
CPP_COMPLIANCE_REPORT="${DOCS_DIR}/cpp_compliance_report.md"

echo "Starting C++ Implementation Standard Compliance Check..."

# Create necessary directories
mkdir -p "${DOCS_DIR}"

# Initialize the compliance report
cat > "${CPP_COMPLIANCE_REPORT}" << EOF
# C++ Implementation Standard Compliance Check
## Report

**Date**: $(date)
**Project**: JadeVectorDB

EOF

# Check for C++20 features usage
echo "## C++20 Features Usage Check" >> "${CPP_COMPLIANCE_REPORT}"

echo "Checking for std::expected usage..." >> "${CPP_COMPLIANCE_REPORT}"
expected_count=$(find "${BACKEND_SRC}" -name "*.cpp" -o -name "*.h" | xargs grep -r "std::expected\|Result<" 2>/dev/null | wc -l || echo "0")
echo "  Found ${expected_count} std::expected/Result usages" >> "${CPP_COMPLIANCE_REPORT}"

echo "Checking for smart pointer usage..." >> "${CPP_COMPLIANCE_REPORT}"
unique_ptr_count=$(find "${BACKEND_SRC}" -name "*.cpp" -o -name "*.h" | xargs grep -r "std::unique_ptr" 2>/dev/null | wc -l || echo "0")
shared_ptr_count=$(find "${BACKEND_SRC}" -name "*.cpp" -o -name "*.h" | xargs grep -r "std::shared_ptr" 2>/dev/null | wc -l || echo "0")
weak_ptr_count=$(find "${BACKEND_SRC}" -name "*.cpp" -o -name "*.h" | xargs grep -r "std::weak_ptr" 2>/dev/null | wc -l || echo "0")
echo "  Found ${unique_ptr_count} std::unique_ptr usages" >> "${CPP_COMPLIANCE_REPORT}"
echo "  Found ${shared_ptr_count} std::shared_ptr usages" >> "${CPP_COMPLIANCE_REPORT}"
echo "  Found ${weak_ptr_count} std::weak_ptr usages" >> "${CPP_COMPLIANCE_REPORT}"

echo "Checking for modern C++ concurrency..." >> "${CPP_COMPLIANCE_REPORT}"
thread_count=$(find "${BACKEND_SRC}" -name "*.cpp" -o -name "*.h" | xargs grep -r "std::thread" 2>/dev/null | wc -l || echo "0")
mutex_count=$(find "${BACKEND_SRC}" -name "*.cpp" -o -name "*.h" | xargs grep -r "std::mutex\|std::shared_mutex" 2>/dev/null | wc -l || echo "0")
atomic_count=$(find "${BACKEND_SRC}" -name "*.cpp" -name "*.h" | xargs grep -r "std::atomic" 2>/dev/null | wc -l || echo "0")
async_count=$(find "${BACKEND_SRC}" -name "*.cpp" -o -name "*.h" | xargs grep -r "std::async" 2>/dev/null | wc -l || echo "0")
echo "  Found ${thread_count} std::thread usages" >> "${CPP_COMPLIANCE_REPORT}"
echo "  Found ${mutex_count} mutex usages" >> "${CPP_COMPLIANCE_REPORT}"
echo "  Found ${atomic_count} std::atomic usages" >> "${CPP_COMPLIANCE_REPORT}"
echo "  Found ${async_count} std::async usages" >> "${CPP_COMPLIANCE_REPORT}"

echo ""
echo "=== COMPLIANCE CHECK COMPLETED ==="
echo "C++ Implementation Standard Compliance Check completed successfully!"
echo "See ${CPP_COMPLIANCE_REPORT} for detailed findings."

# Update the tasks.md file to mark T189 as in progress
sed -i 's/### T189: Ensure C++ implementation standard compliance/**Status**: \[In Progress\]\n### T189: Ensure C++ implementation standard compliance/' "${PROJECT_ROOT}/specs/002-check-if-we/tasks.md"

echo "Task T189 marked as In Progress."