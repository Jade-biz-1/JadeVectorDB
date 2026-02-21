"""Tests for user management CLI commands."""

import json
import uuid

from .conftest import run_cli, parse_json_output


def _extract_user_id(output: str) -> str | None:
    """Extract user_id from CLI output (JSON after the success message)."""
    try:
        # Output is: "Successfully created user: ...\n{json}"
        idx = output.find("{")
        if idx != -1:
            data = json.loads(output[idx:])
            return data.get("user_id", data.get("id"))
    except (json.JSONDecodeError, ValueError):
        pass
    return None


class TestUserAdd:
    def test_add_user(self, auth_token):
        username = f"pytest_u_{uuid.uuid4().hex[:8]}"
        result = run_cli(
            "user-add", username,
            "--role", "user",
            "--password", "TestPass123@",
            token=auth_token,
        )
        assert result.returncode == 0
        assert "Successfully created user" in result.stdout

        # Cleanup
        user_id = _extract_user_id(result.stdout)
        if user_id:
            run_cli("user-delete", user_id, token=auth_token)


class TestUserList:
    def test_list_includes_created_user(self, auth_token):
        username = f"pytest_u_{uuid.uuid4().hex[:8]}"
        create = run_cli(
            "user-add", username,
            "--role", "user",
            "--password", "TestPass123@",
            token=auth_token,
        )
        user_id = _extract_user_id(create.stdout)

        result = run_cli("user-list", token=auth_token)
        assert result.returncode == 0
        assert username in result.stdout

        # Cleanup
        if user_id:
            run_cli("user-delete", user_id, token=auth_token)


class TestUserShow:
    def test_show_user_details(self, auth_token):
        username = f"pytest_u_{uuid.uuid4().hex[:8]}"
        create = run_cli(
            "user-add", username,
            "--role", "user",
            "--password", "TestPass123@",
            token=auth_token,
        )
        user_id = _extract_user_id(create.stdout)
        assert user_id, f"Could not extract user_id from: {create.stdout}"

        result = run_cli("user-show", user_id, token=auth_token)
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None

        # Cleanup
        run_cli("user-delete", user_id, token=auth_token)


class TestUserDeactivateActivate:
    def test_deactivate_user(self, auth_token):
        username = f"pytest_u_{uuid.uuid4().hex[:8]}"
        create = run_cli(
            "user-add", username,
            "--role", "user",
            "--password", "TestPass123@",
            token=auth_token,
        )
        user_id = _extract_user_id(create.stdout)
        assert user_id

        result = run_cli("user-deactivate", user_id, token=auth_token)
        assert result.returncode == 0
        assert "deactivated" in result.stdout.lower()

        # Cleanup
        run_cli("user-delete", user_id, token=auth_token)

    def test_activate_user(self, auth_token):
        username = f"pytest_u_{uuid.uuid4().hex[:8]}"
        create = run_cli(
            "user-add", username,
            "--role", "user",
            "--password", "TestPass123@",
            token=auth_token,
        )
        user_id = _extract_user_id(create.stdout)
        assert user_id

        # Deactivate first, then activate
        run_cli("user-deactivate", user_id, token=auth_token)
        result = run_cli("user-activate", user_id, token=auth_token)
        assert result.returncode == 0
        assert "activated" in result.stdout.lower()

        # Cleanup
        run_cli("user-delete", user_id, token=auth_token)


class TestUserUpdate:
    def test_update_role(self, auth_token):
        username = f"pytest_u_{uuid.uuid4().hex[:8]}"
        create = run_cli(
            "user-add", username,
            "--role", "user",
            "--password", "TestPass123@",
            token=auth_token,
        )
        user_id = _extract_user_id(create.stdout)
        assert user_id

        result = run_cli(
            "user-update", user_id,
            "--role", "developer",
            token=auth_token,
        )
        assert result.returncode == 0
        assert "updated" in result.stdout.lower()

        # Cleanup
        run_cli("user-delete", user_id, token=auth_token)


class TestUserDelete:
    def test_delete_user(self, auth_token):
        username = f"pytest_u_{uuid.uuid4().hex[:8]}"
        create = run_cli(
            "user-add", username,
            "--role", "user",
            "--password", "TestPass123@",
            token=auth_token,
        )
        user_id = _extract_user_id(create.stdout)
        assert user_id

        result = run_cli("user-delete", user_id, token=auth_token)
        assert result.returncode == 0
        assert "deleted" in result.stdout.lower()
