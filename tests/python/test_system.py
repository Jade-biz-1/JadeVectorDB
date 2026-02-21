"""Tests for health and status CLI commands."""

import json

from .conftest import run_cli, parse_json_output


class TestHealth:
    def test_health_returns_healthy(self, auth_token):
        result = run_cli("health", token=auth_token)
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None
        # Check for a healthy indicator in the response
        stdout_lower = result.stdout.lower()
        assert "healthy" in stdout_lower or data.get("status") in ("healthy", "ok")

    def test_health_without_token(self):
        """Health endpoint should work without authentication."""
        result = run_cli("health")
        assert result.returncode == 0
        assert "healthy" in result.stdout.lower() or "ok" in result.stdout.lower()


class TestStatus:
    def test_status_returns_info(self, auth_token):
        result = run_cli("status", token=auth_token)
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None

    def test_status_contains_server_info(self, auth_token):
        result = run_cli("status", token=auth_token)
        assert result.returncode == 0
        stdout = result.stdout.lower()
        # Status should contain some server information
        assert any(keyword in stdout for keyword in ("version", "uptime", "status", "server"))

    def test_status_without_token_fails_or_succeeds_gracefully(self):
        """Status without auth should either succeed or fail with a clear error."""
        result = run_cli("status")
        # Either it works (public endpoint) or it fails with auth error
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
