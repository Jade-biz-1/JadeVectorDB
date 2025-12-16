#!/usr/bin/env python3
"""
JadeVectorDB CLI Test Runner

This script runs comprehensive tests for both Python and Shell CLI tools.
Test data is loaded from test_data.json and all tests use consistent data.

Usage:
    python3 tests/run_cli_tests.py

    Or from project root:
    python3 -m tests.run_cli_tests
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class CLITestRunner:
    def __init__(self, test_data_path: str = None):
        """Initialize test runner with test data."""
        if test_data_path is None:
            test_data_path = PROJECT_ROOT / "tests" / "test_data.json"

        with open(test_data_path, 'r') as f:
            self.test_data = json.load(f)

        self.server_url = self.test_data['server']['url']
        self.token = None
        self.results = []
        self.db_ids = {}

    def generate_vector_values(self, dimension: int, seed: float = 0.1) -> List[float]:
        """Generate test vector values."""
        return [seed + (i * 0.01) for i in range(dimension)]

    def setup_auth(self) -> bool:
        """Set up authentication and get token."""
        auth_data = self.test_data['auth']['test_user']

        try:
            # Register user (may fail if already exists, that's OK)
            requests.post(
                f"{self.server_url}/v1/auth/register",
                json={
                    "username": auth_data['username'],
                    "password": auth_data['password'],
                    "email": auth_data['email']
                },
                timeout=10
            )

            # Login to get token
            login_resp = requests.post(
                f"{self.server_url}/v1/auth/login",
                json={
                    "username": auth_data['username'],
                    "password": auth_data['password']
                },
                timeout=10
            )

            if login_resp.status_code == 200:
                self.token = login_resp.json().get('token', '')
                return bool(self.token)
            else:
                print(f"Login failed: {login_resp.status_code} - {login_resp.text}")
                return False

        except Exception as e:
            print(f"Auth setup failed: {e}")
            return False

    def check_server(self) -> bool:
        """Check if server is running."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def run_python_cli_test(self, test_name: str, args: List[str]) -> Tuple[bool, str]:
        """Run a Python CLI command and return success status and output."""
        try:
            cmd = ['python3', '-m', 'jadevectordb.cli', '--url', self.server_url]
            if self.token and '--api-key' not in args:
                # Add token for authenticated endpoints
                if test_name not in ['Health Check']:
                    cmd.extend(['--api-key', self.token])
            cmd.extend(args)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT / 'cli' / 'python',
                timeout=30
            )

            output = result.stdout if result.returncode == 0 else result.stderr
            return result.returncode == 0, output
        except Exception as e:
            return False, str(e)

    def run_shell_cli_test(self, test_name: str, args: List[str]) -> Tuple[bool, str]:
        """Run a Shell CLI command and return success status and output."""
        try:
            cmd = ['bash', 'scripts/jade-db.sh', '--url', self.server_url]
            if self.token and '--api-key' not in args:
                # Add token for authenticated endpoints
                if test_name not in ['Health Check']:
                    cmd.extend(['--api-key', self.token])
            cmd.extend(args)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT / 'cli' / 'shell',
                timeout=30
            )

            output = result.stdout if result.returncode == 0 else result.stderr
            return result.returncode == 0, output
        except Exception as e:
            return False, str(e)

    def extract_database_id(self, output: str) -> str:
        """Extract database ID from CLI output."""
        # Try JSON format first
        try:
            data = json.loads(output)
            return data.get('database_id', data.get('databaseId', ''))
        except:
            pass

        # Try text format: "Created database with ID: db_xxx"
        match = re.search(r'ID:\s*(\S+)', output)
        if match:
            return match.group(1)

        return ""

    def run_all_tests(self):
        """Run all CLI tests."""
        print("\n" + "="*80)
        print("JadeVectorDB CLI Test Suite")
        print("="*80)

        # Check server
        print("\n[1/4] Checking server connectivity...")
        if not self.check_server():
            print("❌ Server is not running at", self.server_url)
            print("\nPlease start the server first:")
            print("  cd backend/build && ./jadevectordb")
            return False
        print("✓ Server is running")

        # Setup authentication
        print("\n[2/4] Setting up authentication...")
        if not self.setup_auth():
            print("❌ Authentication setup failed")
            return False
        print("✓ Authentication successful")

        # Run tests
        print("\n[3/4] Running CLI tests...")
        self.run_python_cli_tests()
        self.run_shell_cli_tests()

        # Print results
        print("\n[4/4] Test Results:")
        self.print_results()

        return True

    def run_python_cli_tests(self):
        """Run Python CLI tests."""
        # Test 1: Health Check
        success, output = self.run_python_cli_test('Health Check', ['health'])
        self.results.append([1, "Python CLI", "Health Check",
                           "PASS" if success and "healthy" in output.lower() else "FAIL"])

        # Test 2: Status Check
        success, output = self.run_python_cli_test('Status Check', ['--api-key', self.token, 'status'])
        self.results.append([2, "Python CLI", "Status Check", "PASS" if success else "FAIL"])

        # Test 3: List Databases
        success, output = self.run_python_cli_test('List Databases', ['--api-key', self.token, 'list-dbs'])
        self.results.append([3, "Python CLI", "List Databases", "PASS" if success else "FAIL"])

        # Test 4: Create Database
        db_config = self.test_data['databases']['python_test']
        success, output = self.run_python_cli_test('Create Database', [
            '--api-key', self.token, 'create-db',
            '--name', db_config['name'],
            '--dimension', str(db_config['dimension']),
            '--index-type', db_config['index_type']
        ])
        db_id = self.extract_database_id(output) if success else ""
        self.db_ids['python'] = db_id
        self.results.append([4, "Python CLI", "Create Database",
                           "PASS" if success and db_id else "FAIL"])

        # Test 5: Get Database
        if db_id:
            success, output = self.run_python_cli_test('Get Database', [
                '--api-key', self.token, 'get-db', '--database-id', db_id
            ])
            self.results.append([5, "Python CLI", "Get Database",
                               "PASS" if success and db_id in output else "FAIL"])
        else:
            self.results.append([5, "Python CLI", "Get Database", "SKIP"])

        # Test 6: Store Vector
        if db_id:
            vec_data = self.test_data['vectors']['test_vector_1']
            vec_dimension = vec_data['dimension']
            vec_values = self.generate_vector_values(vec_dimension)

            success, output = self.run_python_cli_test('Store Vector', [
                '--api-key', self.token, 'store',
                '--database-id', db_id,
                '--vector-id', vec_data['id'],
                '--values', json.dumps(vec_values)
            ])
            self.results.append([6, "Python CLI", "Store Vector", "PASS" if success else "FAIL"])
        else:
            self.results.append([6, "Python CLI", "Store Vector", "SKIP"])

        # Test 7: Search Vectors
        if db_id:
            search_config = self.test_data['search']
            query_vec = self.generate_vector_values(search_config['query_vector']['dimension'], 0.1)

            success, output = self.run_python_cli_test('Search Vectors', [
                '--api-key', self.token, 'search',
                '--database-id', db_id,
                '--query-vector', json.dumps(query_vec),
                '--top-k', str(search_config['top_k'])
            ])
            self.results.append([7, "Python CLI", "Search Vectors", "PASS" if success else "FAIL"])
            if not success:
                print(f"\n[DEBUG] Search test failed with output: {output[:500]}")
        else:
            self.results.append([7, "Python CLI", "Search Vectors", "SKIP"])

    def run_shell_cli_tests(self):
        """Run Shell CLI tests."""
        # Test 8: Health Check
        success, output = self.run_shell_cli_test('Health Check', ['health'])
        passed = success and "healthy" in output.lower()
        self.results.append([8, "Shell CLI", "Health Check", "PASS" if passed else "FAIL"])
        if not passed:
            print(f"\n[DEBUG] Shell Health test failed. Success={success}, Output: {output[:500]}")

        # Test 9: Status Check
        success, output = self.run_shell_cli_test('Status Check', ['--api-key', self.token, 'status'])
        self.results.append([9, "Shell CLI", "Status Check", "PASS" if success else "FAIL"])

        # Test 10: List Databases
        success, output = self.run_shell_cli_test('List Databases', ['--api-key', self.token, 'list-dbs'])
        self.results.append([10, "Shell CLI", "List Databases", "PASS" if success else "FAIL"])

        # Test 11: Create Database
        db_config = self.test_data['databases']['shell_test']
        success, output = self.run_shell_cli_test('Create Database', [
            '--api-key', self.token, 'create-db',
            db_config['name'],
            db_config.get('description', ''),
            str(db_config['dimension']),
            db_config['index_type']
        ])
        db_id = self.extract_database_id(output) if success else ""
        self.db_ids['shell'] = db_id
        self.results.append([11, "Shell CLI", "Create Database",
                           "PASS" if success and db_id else "FAIL"])

        # Test 12: Get Database
        if db_id:
            success, output = self.run_shell_cli_test('Get Database', [
                '--api-key', self.token, 'get-db', db_id
            ])
            self.results.append([12, "Shell CLI", "Get Database",
                               "PASS" if success and db_id in output else "FAIL"])
        else:
            self.results.append([12, "Shell CLI", "Get Database", "SKIP"])

    def print_results(self):
        """Print test results in a formatted table."""
        print("\n" + "="*80)
        print(f"{'#':<5} {'Tool':<15} {'Test':<30} {'Result':<30}")
        print("="*80)

        for row in self.results:
            num, tool, test, result = row
            # Color code results
            if result == "PASS":
                result_str = f"✓ {result}"
            elif result == "FAIL":
                result_str = f"✗ {result}"
            else:
                result_str = f"⊘ {result}"

            print(f"{num:<5} {tool:<15} {test:<30} {result_str:<30}")

        print("="*80)

        # Summary
        passed = sum(1 for r in self.results if r[3] == "PASS")
        failed = sum(1 for r in self.results if r[3] == "FAIL")
        skipped = sum(1 for r in self.results if r[3] == "SKIP")
        total = passed + failed

        print(f"\nSummary: {passed}/{total} tests passed")
        if failed > 0:
            print(f"  Failed: {failed}")
        if skipped > 0:
            print(f"  Skipped: {skipped}")

        # Hints for failures
        if failed > 0:
            print("\n" + "-"*80)
            print("Troubleshooting Hints:")
            print("-"*80)

            for row in self.results:
                num, tool, test, result = row
                if result == "FAIL":
                    hints = self.get_failure_hints(tool, test)
                    if hints:
                        print(f"\n[Test #{num}] {tool} - {test}:")
                        for hint in hints:
                            print(f"  • {hint}")

        print("="*80 + "\n")

        return passed == total

    def get_failure_hints(self, tool: str, test: str) -> List[str]:
        """Get troubleshooting hints for failed tests."""
        hints = []

        if "Create Database" in test:
            hints.extend([
                "Check if the server is running and accessible",
                "Verify authentication token is valid",
                "Check server logs for detailed error messages"
            ])

        if "Store Vector" in test:
            hints.extend([
                "Ensure database was created successfully",
                "Verify vector dimensions match database configuration",
                "Check that vector ID is unique"
            ])

        if "Search" in test:
            hints.extend([
                "Ensure at least one vector is stored in the database",
                "Verify search service is properly initialized",
                "Check server logs for search-related errors"
            ])

        if tool == "Shell CLI":
            hints.extend([
                "Verify bash and curl are installed",
                "Check shell script has execute permissions",
                "Test shell CLI commands manually for debugging"
            ])

        return hints


def main():
    """Main entry point."""
    runner = CLITestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
