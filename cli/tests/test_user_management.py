#!/usr/bin/env python3
"""
User Management CLI Tests (T264)

Comprehensive test suite for user management commands across Python CLI.
Tests user add, list, show, update, delete, activate, and deactivate operations.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.jadevectordb.client import JadeVectorDB, JadeVectorDBError


class TestUserManagementClient:
    """Test user management methods in the JadeVectorDB client"""

    @pytest.fixture
    def mock_session(self):
        """Create a mock requests session"""
        session = Mock()
        return session

    @pytest.fixture
    def client(self, mock_session):
        """Create a JadeVectorDB client with mocked session"""
        with patch('python.jadevectordb.client.requests.Session', return_value=mock_session):
            client = JadeVectorDB("http://localhost:8080", "test-api-key")
            client.session = mock_session
            return client

    def test_create_user_success(self, client, mock_session):
        """Test successful user creation"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "user_id": "user123",
            "email": "test@example.com",
            "role": "developer",
            "message": "User created successfully"
        }
        mock_session.post.return_value = mock_response

        # Call create_user
        result = client.create_user("test@example.com", "developer", "password123")

        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "/api/v1/users" in call_args[0][0]
        assert call_args[1]['json']['email'] == "test@example.com"
        assert call_args[1]['json']['role'] == "developer"
        assert call_args[1]['json']['password'] == "password123"

        # Verify result
        assert result['user_id'] == "user123"
        assert result['email'] == "test@example.com"

    def test_create_user_without_password(self, client, mock_session):
        """Test user creation without password"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"user_id": "user456"}
        mock_session.post.return_value = mock_response

        result = client.create_user("admin@example.com", "admin")

        call_args = mock_session.post.call_args
        assert 'password' not in call_args[1]['json'] or call_args[1]['json']['password'] is None

    def test_create_user_already_exists(self, client, mock_session):
        """Test creating a user that already exists"""
        mock_response = Mock()
        mock_response.status_code = 409
        mock_response.text = "User already exists"
        mock_session.post.return_value = mock_response

        with pytest.raises(JadeVectorDBError, match="Failed to create user"):
            client.create_user("existing@example.com", "developer")

    def test_list_users_success(self, client, mock_session):
        """Test listing all users"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "users": [
                {"user_id": "1", "email": "admin@example.com", "role": "admin", "status": "active"},
                {"user_id": "2", "email": "dev@example.com", "role": "developer", "status": "active"}
            ]
        }
        mock_session.get.return_value = mock_response

        users = client.list_users()

        assert len(users) == 2
        assert users[0]['email'] == "admin@example.com"
        assert users[1]['role'] == "developer"
        mock_session.get.assert_called_once()

    def test_list_users_with_filters(self, client, mock_session):
        """Test listing users with role and status filters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"users": []}
        mock_session.get.return_value = mock_response

        client.list_users(role="admin", status="active")

        call_args = mock_session.get.call_args
        assert call_args[1]['params']['role'] == "admin"
        assert call_args[1]['params']['status'] == "active"

    def test_get_user_success(self, client, mock_session):
        """Test getting user details"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "user123",
            "email": "test@example.com",
            "role": "developer",
            "status": "active"
        }
        mock_session.get.return_value = mock_response

        user = client.get_user("test@example.com")

        assert user['email'] == "test@example.com"
        assert user['role'] == "developer"

    def test_get_user_not_found(self, client, mock_session):
        """Test getting non-existent user"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "User not found"
        mock_session.get.return_value = mock_response

        with pytest.raises(JadeVectorDBError, match="Failed to get user"):
            client.get_user("nonexistent@example.com")

    def test_update_user_success(self, client, mock_session):
        """Test updating user information"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "user123",
            "email": "test@example.com",
            "role": "admin",
            "message": "User updated successfully"
        }
        mock_session.put.return_value = mock_response

        result = client.update_user("test@example.com", role="admin")

        assert result['role'] == "admin"
        mock_session.put.assert_called_once()

    def test_delete_user_success(self, client, mock_session):
        """Test deleting a user"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "User deleted successfully"}
        mock_session.delete.return_value = mock_response

        result = client.delete_user("test@example.com")

        assert result['message'] == "User deleted successfully"
        mock_session.delete.assert_called_once()

    def test_activate_user_success(self, client, mock_session):
        """Test activating a user"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "user123",
            "status": "active",
            "message": "User activated successfully"
        }
        mock_session.put.return_value = mock_response

        result = client.activate_user("test@example.com")

        assert result['status'] == "active"
        call_args = mock_session.put.call_args
        assert "/activate" in call_args[0][0]

    def test_deactivate_user_success(self, client, mock_session):
        """Test deactivating a user"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_id": "user123",
            "status": "inactive",
            "message": "User deactivated successfully"
        }
        mock_session.put.return_value = mock_response

        result = client.deactivate_user("test@example.com")

        assert result['status'] == "inactive"
        call_args = mock_session.put.call_args
        assert "/deactivate" in call_args[0][0]


class TestUserManagementCLI:
    """Test user management CLI commands"""

    @patch('python.jadevectordb.cli.JadeVectorDB')
    def test_user_add_command(self, mock_client_class):
        """Test user-add CLI command"""
        # Mock client instance
        mock_client = Mock()
        mock_client.create_user.return_value = {
            "user_id": "user123",
            "email": "test@example.com",
            "role": "developer"
        }
        mock_client_class.return_value = mock_client

        # Simulate CLI arguments
        from python.jadevectordb.cli import user_add

        args = Mock()
        args.url = "http://localhost:8080"
        args.api_key = "test-key"
        args.email = "test@example.com"
        args.role = "developer"
        args.password = "password123"

        # Call CLI function
        user_add(args)

        # Verify client was called correctly
        mock_client.create_user.assert_called_once_with(
            "test@example.com",
            "developer",
            "password123"
        )

    @patch('python.jadevectordb.cli.JadeVectorDB')
    @patch('python.jadevectordb.cli.print_formatted')
    def test_user_list_command(self, mock_print, mock_client_class):
        """Test user-list CLI command"""
        mock_client = Mock()
        mock_client.list_users.return_value = [
            {"email": "admin@example.com", "role": "admin"},
            {"email": "dev@example.com", "role": "developer"}
        ]
        mock_client_class.return_value = mock_client

        from python.jadevectordb.cli import user_list

        args = Mock()
        args.url = "http://localhost:8080"
        args.api_key = "test-key"
        args.role = None
        args.status = None
        args.format = "json"

        user_list(args)

        mock_client.list_users.assert_called_once()
        mock_print.assert_called_once()

    @patch('python.jadevectordb.cli.JadeVectorDB')
    def test_user_delete_command(self, mock_client_class):
        """Test user-delete CLI command"""
        mock_client = Mock()
        mock_client.delete_user.return_value = {"message": "User deleted"}
        mock_client_class.return_value = mock_client

        from python.jadevectordb.cli import user_delete

        args = Mock()
        args.url = "http://localhost:8080"
        args.api_key = "test-key"
        args.email = "test@example.com"

        user_delete(args)

        mock_client.delete_user.assert_called_once_with("test@example.com")


def test_user_management_workflow():
    """Integration test: Complete user management workflow"""
    with patch('python.jadevectordb.client.requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        # Create client
        client = JadeVectorDB("http://localhost:8080", "test-key")
        client.session = mock_session

        # Test workflow: Create -> List -> Update -> Activate -> Deactivate -> Delete

        # 1. Create user
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"user_id": "user123"}
        mock_session.post.return_value = mock_response

        client.create_user("workflow@example.com", "developer", "pass123")

        # 2. List users
        mock_response.status_code = 200
        mock_response.json.return_value = {"users": [{"email": "workflow@example.com"}]}
        mock_session.get.return_value = mock_response

        users = client.list_users()
        assert len(users) == 1

        # 3. Update user
        mock_response.json.return_value = {"role": "admin"}
        mock_session.put.return_value = mock_response

        client.update_user("workflow@example.com", role="admin")

        # 4. Deactivate user
        mock_response.json.return_value = {"status": "inactive"}
        client.deactivate_user("workflow@example.com")

        # 5. Activate user
        mock_response.json.return_value = {"status": "active"}
        client.activate_user("workflow@example.com")

        # 6. Delete user
        mock_response.json.return_value = {"message": "deleted"}
        mock_session.delete.return_value = mock_response

        client.delete_user("workflow@example.com")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
