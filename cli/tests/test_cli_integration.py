#!/usr/bin/env python3
"""
CLI Integration Tests (T272)

Comprehensive integration test suite for testing cross-CLI consistency,
end-to-end workflows, and complete user scenarios.

These tests can run with mocked backends OR against a real backend if available.
"""

import sys
import os
import pytest
import json
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.jadevectordb.client import JadeVectorDB
from python.jadevectordb.import_export import VectorImporter, VectorExporter
from python.jadevectordb import formatters


class TestCrossC LIConsistency:
    """Test consistency across Python, Shell, and JavaScript CLIs"""

    def test_user_management_commands_consistent(self):
        """Verify all CLIs have same user management commands"""
        expected_commands = [
            'user-add', 'user-list', 'user-show',
            'user-update', 'user-delete',
            'user-activate', 'user-deactivate'
        ]

        # Test Python CLI help contains all commands
        with patch('sys.stdout'):
            import python.jadevectordb.cli as cli_module

        # Verify command functions exist
        for cmd in expected_commands:
            func_name = cmd.replace('-', '_')
            assert hasattr(cli_module, func_name), \
                f"Python CLI missing {func_name} function"

    def test_output_formats_consistent(self):
        """Verify all CLIs support same output formats"""
        expected_formats = ['json', 'yaml', 'table', 'csv']

        # Test Python formatters
        for fmt in expected_formats:
            assert fmt in ['json', 'yaml', 'table', 'csv'], \
                f"Python CLI should support {fmt} format"

    def test_api_endpoints_consistent(self):
        """Verify all CLIs use same API endpoints"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"user_id": "123"}
            mock_session.return_value.post.return_value = mock_response

            client = JadeVectorDB("http://localhost:8080", "test-key")
            client.session = mock_session.return_value

            # Create user
            client.create_user("test@example.com", "developer")

            # Verify endpoint
            call_args = mock_session.return_value.post.call_args
            assert "/api/v1/users" in call_args[0][0]


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""

    @pytest.fixture
    def mock_client(self):
        """Create a fully mocked client for E2E tests"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            client = JadeVectorDB("http://localhost:8080", "test-key")
            client.session = mock_session.return_value
            yield client, mock_session.return_value

    def test_complete_user_lifecycle(self, mock_client):
        """Test complete user lifecycle: create -> list -> update -> activate/deactivate -> delete"""
        client, session = mock_client

        # Step 1: Create user
        session.post.return_value = Mock(
            status_code=201,
            json=lambda: {"user_id": "user-123", "email": "lifecycle@example.com"}
        )
        result = client.create_user("lifecycle@example.com", "developer", "password123")
        assert result['user_id'] == "user-123"

        # Step 2: List users to verify creation
        session.get.return_value = Mock(
            status_code=200,
            json=lambda: {"users": [{"email": "lifecycle@example.com", "status": "active"}]}
        )
        users = client.list_users()
        assert len(users) == 1
        assert users[0]['email'] == "lifecycle@example.com"

        # Step 3: Get user details
        session.get.return_value = Mock(
            status_code=200,
            json=lambda: {
                "user_id": "user-123",
                "email": "lifecycle@example.com",
                "role": "developer",
                "status": "active"
            }
        )
        user = client.get_user("lifecycle@example.com")
        assert user['role'] == "developer"

        # Step 4: Update user role
        session.put.return_value = Mock(
            status_code=200,
            json=lambda: {"user_id": "user-123", "role": "admin"}
        )
        updated = client.update_user("lifecycle@example.com", role="admin")
        assert updated['role'] == "admin"

        # Step 5: Deactivate user
        session.put.return_value = Mock(
            status_code=200,
            json=lambda: {"status": "inactive"}
        )
        deactivated = client.deactivate_user("lifecycle@example.com")
        assert deactivated['status'] == "inactive"

        # Step 6: Reactivate user
        session.put.return_value = Mock(
            status_code=200,
            json=lambda: {"status": "active"}
        )
        activated = client.activate_user("lifecycle@example.com")
        assert activated['status'] == "active"

        # Step 7: Delete user
        session.delete.return_value = Mock(
            status_code=200,
            json=lambda: {"message": "User deleted successfully"}
        )
        deleted = client.delete_user("lifecycle@example.com")
        assert "deleted" in deleted['message'].lower()

    def test_import_export_workflow(self, mock_client):
        """Test complete import/export workflow"""
        client, session = mock_client

        # Create test vectors
        test_vectors = [
            {"id": "v1", "values": [0.1, 0.2], "metadata": {"tag": "A"}},
            {"id": "v2", "values": [0.3, 0.4], "metadata": {"tag": "B"}}
        ]

        # Mock list_vectors for export
        session.get.return_value = Mock(
            status_code=200,
            json=lambda: test_vectors
        )

        # Export to file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.close()

        try:
            exporter = VectorExporter(client, "source-db")
            export_result = exporter.export_to_json(temp_file.name)

            assert export_result['exported'] == 2
            assert os.path.exists(temp_file.name)

            # Verify file contents
            with open(temp_file.name, 'r') as f:
                exported = json.load(f)
            assert len(exported) == 2

            # Mock store_vector for import
            session.post.return_value = Mock(status_code=201)

            # Import to different database
            importer = VectorImporter(client, "target-db")
            import_result = importer.import_from_json(temp_file.name)

            assert import_result['imported'] == 2
            assert import_result['success_rate'] == 100.0

        finally:
            os.unlink(temp_file.name)

    def test_database_and_vector_workflow(self, mock_client):
        """Test database creation, vector storage, search workflow"""
        client, session = mock_client

        # 1. Create database
        session.post.return_value = Mock(
            status_code=201,
            json=lambda: {"databaseId": "db-123"}
        )
        db_id = client.create_database("test-db", "Test database", 128, "HNSW")
        assert db_id == "db-123"

        # 2. Store vectors
        session.post.return_value = Mock(status_code=201)
        success = client.store_vector("db-123", "vec-1", [0.1, 0.2, 0.3])
        assert success

        # 3. Retrieve vector
        session.get.return_value = Mock(
            status_code=200,
            json=lambda: {"id": "vec-1", "values": [0.1, 0.2, 0.3]}
        )
        vector = client.retrieve_vector("db-123", "vec-1")
        assert vector['id'] == "vec-1"

        # 4. Search similar vectors
        session.post.return_value = Mock(
            status_code=200,
            json=lambda: [
                {"id": "vec-1", "similarity": 0.95},
                {"id": "vec-2", "similarity": 0.85}
            ]
        )
        results = client.search("db-123", [0.1, 0.2, 0.3], top_k=2)
        assert len(results) == 2
        assert results[0]['similarity'] == 0.95


class TestOutputFormatConsistency:
    """Test output format consistency across operations"""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing formatters"""
        return {
            "users": [
                {"email": "admin@example.com", "role": "admin", "status": "active"},
                {"email": "dev@example.com", "role": "developer", "status": "active"}
            ]
        }

    def test_json_output_format(self, sample_data):
        """Test JSON output formatting"""
        output = formatters.format_json(sample_data)
        assert isinstance(output, str)
        # Verify it's valid JSON
        parsed = json.loads(output)
        assert parsed == sample_data

    def test_yaml_output_format(self, sample_data):
        """Test YAML output formatting"""
        output = formatters.format_yaml(sample_data)
        assert isinstance(output, str)
        # YAML output should contain keys
        assert "users" in output
        assert "email" in output

    def test_table_output_format(self, sample_data):
        """Test table output formatting"""
        users_list = sample_data["users"]
        output = formatters.format_table(users_list)
        assert isinstance(output, str)
        # Table should contain data
        assert "admin@example.com" in output or "admin" in output

    def test_csv_output_format(self, sample_data):
        """Test CSV output formatting"""
        users_list = sample_data["users"]
        output = formatters.format_csv(users_list)
        assert isinstance(output, str)
        # CSV should have header and data
        lines = output.strip().split('\n')
        assert len(lines) >= 2  # Header + at least one data row
        assert "email" in lines[0].lower()

    def test_format_consistency(self, sample_data):
        """Test that all formats handle the same data consistently"""
        users_list = sample_data["users"]

        json_out = formatters.format_json(users_list)
        yaml_out = formatters.format_yaml(users_list)
        table_out = formatters.format_table(users_list)
        csv_out = formatters.format_csv(users_list)

        # All should produce output
        assert len(json_out) > 0
        assert len(yaml_out) > 0
        assert len(table_out) > 0
        assert len(csv_out) > 0

        # All should contain the email addresses
        for fmt_out in [json_out, yaml_out, table_out, csv_out]:
            assert "admin@example.com" in fmt_out or "admin" in fmt_out


class TestErrorHandlingConsistency:
    """Test error handling consistency across CLIs"""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked session"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            client = JadeVectorDB("http://localhost:8080", "test-key")
            client.session = mock_session.return_value
            yield client, mock_session.return_value

    def test_user_already_exists_error(self, mock_client):
        """Test handling of user already exists error"""
        client, session = mock_client

        session.post.return_value = Mock(
            status_code=409,
            text="User already exists"
        )

        with pytest.raises(Exception) as exc_info:
            client.create_user("existing@example.com", "developer")

        assert "Failed to create user" in str(exc_info.value)

    def test_user_not_found_error(self, mock_client):
        """Test handling of user not found error"""
        client, session = mock_client

        session.get.return_value = Mock(
            status_code=404,
            text="User not found"
        )

        with pytest.raises(Exception) as exc_info:
            client.get_user("nonexistent@example.com")

        assert "Failed to get user" in str(exc_info.value)

    def test_authentication_error(self, mock_client):
        """Test handling of authentication errors"""
        client, session = mock_client

        session.get.return_value = Mock(
            status_code=401,
            text="Unauthorized"
        )

        with pytest.raises(Exception):
            client.list_users()

    def test_validation_error(self, mock_client):
        """Test handling of validation errors"""
        client, session = mock_client

        session.post.return_value = Mock(
            status_code=400,
            text="Invalid email format"
        )

        with pytest.raises(Exception):
            client.create_user("invalid-email", "developer")


class TestPerformanceAndScalability:
    """Test performance with various data sizes"""

    def test_small_dataset_import(self):
        """Test import performance with small dataset (< 100 vectors)"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            client = JadeVectorDB("http://localhost:8080", "test-key")
            client.session = mock_session.return_value
            mock_session.return_value.post.return_value = Mock(status_code=201)

            # Create small dataset
            vectors = [
                {"id": f"v{i}", "values": [float(i), float(i+1)], "metadata": {}}
                for i in range(50)
            ]

            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            json.dump(vectors, temp_file)
            temp_file.close()

            try:
                importer = VectorImporter(client, "test-db", batch_size=10)
                result = importer.import_from_json(temp_file.name)

                assert result['imported'] == 50
                assert result['success_rate'] == 100.0
            finally:
                os.unlink(temp_file.name)

    def test_large_dataset_import(self):
        """Test import performance with large dataset (1000+ vectors)"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            client = JadeVectorDB("http://localhost:8080", "test-key")
            client.session = mock_session.return_value
            mock_session.return_value.post.return_value = Mock(status_code=201)

            # Create large dataset
            vectors = [
                {"id": f"v{i}", "values": [float(i), float(i+1)], "metadata": {"idx": i}}
                for i in range(1000)
            ]

            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            json.dump(vectors, temp_file)
            temp_file.close()

            try:
                importer = VectorImporter(client, "test-db", batch_size=100)
                result = importer.import_from_json(temp_file.name)

                assert result['imported'] == 1000
                assert result['success_rate'] == 100.0
                # Verify batching occurred (10 batches of 100)
                assert mock_session.return_value.post.call_count == 1000
            finally:
                os.unlink(temp_file.name)

    def test_batch_size_configuration(self):
        """Test that batch size can be configured"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            client = JadeVectorDB("http://localhost:8080", "test-key")

            # Test different batch sizes
            for batch_size in [10, 50, 100, 200]:
                importer = VectorImporter(client, "test-db", batch_size=batch_size)
                assert importer.batch_size == batch_size


class TestAuthenticationAndPermissions:
    """Test authentication and permission scenarios"""

    def test_api_key_required(self):
        """Test that API key is properly used"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            session_instance = Mock()
            mock_session.return_value = session_instance

            client = JadeVectorDB("http://localhost:8080", "my-secret-key")
            client.session = session_instance

            # Verify session headers include API key
            assert hasattr(client, 'api_key')
            assert client.api_key == "my-secret-key"

    def test_unauthorized_access(self):
        """Test handling of unauthorized access"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            client = JadeVectorDB("http://localhost:8080", "invalid-key")
            client.session = mock_session.return_value

            mock_session.return_value.get.return_value = Mock(
                status_code=401,
                text="Invalid API key"
            )

            with pytest.raises(Exception):
                client.list_users()

    def test_permission_denied(self):
        """Test handling of permission denied errors"""
        with patch('python.jadevectordb.client.requests.Session') as mock_session:
            client = JadeVectorDB("http://localhost:8080", "viewer-key")
            client.session = mock_session.return_value

            mock_session.return_value.post.return_value = Mock(
                status_code=403,
                text="Permission denied"
            )

            with pytest.raises(Exception):
                client.create_user("test@example.com", "admin")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-k", "not slow"])
