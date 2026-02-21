"""Tests for audit-log CLI command."""

from .conftest import run_cli, parse_json_output


class TestAuditLog:
    def test_audit_log_returns_events(self, auth_token):
        result = run_cli("audit-log", token=auth_token)
        assert result.returncode == 0
        # Should return some output (events or empty list)
        assert result.stdout.strip()

    def test_audit_log_limit(self, auth_token):
        result = run_cli("audit-log", "--limit", "1", token=auth_token)
        assert result.returncode == 0
        data = parse_json_output(result)
        if data is not None:
            events = data if isinstance(data, list) else data.get("events", data.get("entries", []))
            assert len(events) <= 1
