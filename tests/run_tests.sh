#!/bin/bash
# JadeVectorDB CLI Test Runner
# Simple wrapper script to run CLI tests

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Run the Python test runner
python3 tests/run_cli_tests.py "$@"
