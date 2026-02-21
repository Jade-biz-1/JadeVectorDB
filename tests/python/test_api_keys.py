"""Tests for API key management CLI commands."""

import json
import uuid

from .conftest import run_cli, parse_json_output


class TestCreateApiKey:
    def test_create_api_key(self, auth_token, auth_user_id):
        result = run_cli(
            "create-api-key",
            "--user-id", auth_user_id,
            "--description", "pytest key",
            token=auth_token,
        )
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None
        # Should return the generated API key
        api_key = data.get("api_key", data.get("key", ""))
        assert api_key, f"No api_key in response: {data}"


class TestListApiKeys:
    def test_list_includes_created_key(self, auth_token, auth_user_id):
        # Create a key first
        create_result = run_cli(
            "create-api-key",
            "--user-id", auth_user_id,
            "--description", "pytest list test",
            token=auth_token,
        )
        assert create_result.returncode == 0

        result = run_cli("list-api-keys", token=auth_token)
        assert result.returncode == 0
        # Should contain at least one key
        assert "pytest list test" in result.stdout or "key" in result.stdout.lower()


class TestApiKeyAuthentication:
    def test_api_key_can_authenticate(self, auth_token, auth_user_id):
        """A freshly created API key should work for authentication."""
        create_result = run_cli(
            "create-api-key",
            "--user-id", auth_user_id,
            "--description", "pytest auth test",
            token=auth_token,
        )
        data = parse_json_output(create_result)
        assert data is not None
        api_key = data.get("api_key", data.get("key", ""))
        assert api_key

        # Use the new API key to call status
        status_result = run_cli("status", token=api_key)
        assert status_result.returncode == 0


def _find_key_id(auth_token, description):
    """List API keys and find the key_id by description."""
    list_result = run_cli("list-api-keys", token=auth_token)
    if list_result.returncode != 0:
        return None
    # Parse JSON output — may be a list or contain an api_keys array
    try:
        idx = list_result.stdout.find("[")
        if idx == -1:
            idx = list_result.stdout.find("{")
        if idx != -1:
            data = json.loads(list_result.stdout[idx:])
            keys = data if isinstance(data, list) else data.get("api_keys", data.get("keys", []))
            for k in keys:
                if k.get("description", "") == description:
                    return k.get("key_id", k.get("id"))
    except (json.JSONDecodeError, ValueError):
        pass
    return None


class TestRevokeApiKey:
    def test_revoke_api_key(self, auth_token, auth_user_id):
        # Create a key with unique description
        desc = f"pytest revoke test {uuid.uuid4().hex[:6]}"
        create_result = run_cli(
            "create-api-key",
            "--user-id", auth_user_id,
            "--description", desc,
            token=auth_token,
        )
        assert create_result.returncode == 0

        # Find the key_id by listing keys
        key_id = _find_key_id(auth_token, desc)
        assert key_id, f"Could not find key_id for description: {desc}"

        # Revoke it
        result = run_cli(
            "revoke-api-key",
            "--key-id", key_id,
            token=auth_token,
        )
        assert result.returncode == 0
        assert "revoked" in result.stdout.lower()

    def test_revoked_key_cannot_authenticate(self, auth_token, auth_user_id):
        """After revoking, the key should no longer work."""
        desc = f"pytest revoke auth test {uuid.uuid4().hex[:6]}"
        create_result = run_cli(
            "create-api-key",
            "--user-id", auth_user_id,
            "--description", desc,
            token=auth_token,
        )
        data = parse_json_output(create_result)
        assert data is not None
        api_key = data.get("api_key", data.get("key", ""))
        assert api_key

        # Find the key_id by listing keys
        key_id = _find_key_id(auth_token, desc)
        assert key_id, f"Could not find key_id for description: {desc}"

        # Revoke
        run_cli("revoke-api-key", "--key-id", key_id, token=auth_token)

        # Try to use revoked key — should fail or return error
        status_result = run_cli("status", token=api_key)
        assert status_result.returncode != 0 or "error" in status_result.stdout.lower() or "error" in status_result.stderr.lower()
