"""
Shared fixtures for JadeVectorDB Python CLI tests.

All tests invoke the real CLI via subprocess:
    python3 -m jadevectordb.cli --url <url> [--api-key <token>] <command> [args...]
"""

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLI_CWD = PROJECT_ROOT / "cli" / "python"

SERVER_URL = os.environ.get("JADEVECTORDB_TEST_URL", "http://localhost:8080")
TEST_USERNAME = f"pytest_user_{uuid.uuid4().hex[:8]}"
TEST_PASSWORD = "PyTest_Secure123@"
TEST_EMAIL = f"{TEST_USERNAME}@test.jadevectordb.com"

VECTOR_DIMENSION = 128


def _server_is_reachable() -> bool:
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def run_cli(*args: str, token: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run the Python CLI as a subprocess and return the CompletedProcess."""
    cmd = ["python3", "-m", "jadevectordb.cli", "--url", SERVER_URL]
    if token:
        cmd.extend(["--api-key", token])
    cmd.extend(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(CLI_CWD),
        timeout=timeout,
    )


def parse_json_output(result: subprocess.CompletedProcess) -> dict | list | None:
    """Attempt to parse JSON from stdout. Returns None on failure."""
    text = result.stdout.strip()
    if not text:
        return None
    # Some commands print a message before JSON; try to find JSON
    for start_char in ("{", "["):
        idx = text.find(start_char)
        if idx != -1:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Session-scoped: register + login once, reuse token everywhere
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _skip_if_server_down():
    if not _server_is_reachable():
        pytest.skip(f"JadeVectorDB server not reachable at {SERVER_URL}")


@pytest.fixture(scope="session")
def auth_token() -> str:
    """Register a test user and return a valid auth token for the session."""
    # Register (may already exist — that's fine)
    requests.post(
        f"{SERVER_URL}/v1/auth/register",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD, "email": TEST_EMAIL},
        timeout=10,
    )
    # Login
    resp = requests.post(
        f"{SERVER_URL}/v1/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        timeout=10,
    )
    assert resp.status_code == 200, f"Login failed: {resp.status_code} {resp.text}"
    data = resp.json()
    token = data.get("token", "")
    assert token, "No token returned from login"
    return token


@pytest.fixture(scope="session")
def auth_user_id(auth_token) -> str:
    """Return the user_id of the session test user."""
    resp = requests.post(
        f"{SERVER_URL}/v1/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        timeout=10,
    )
    return resp.json().get("user_id", "")


# ---------------------------------------------------------------------------
# Function-scoped: create / destroy a temporary database per test
# ---------------------------------------------------------------------------

@pytest.fixture
def test_database(auth_token) -> str:
    """Create a temp database, yield its ID, delete on teardown."""
    db_name = f"pytest_db_{uuid.uuid4().hex[:8]}"
    result = run_cli(
        "create-db",
        "--name", db_name,
        "--dimension", str(VECTOR_DIMENSION),
        "--index-type", "hnsw",
        token=auth_token,
    )
    assert result.returncode == 0, f"create-db failed: {result.stdout} {result.stderr}"
    # Output: "Created database with ID: <id>"
    db_id = result.stdout.strip().split("ID: ")[-1].strip()
    assert db_id, f"Could not extract database ID from: {result.stdout}"

    yield db_id

    # Teardown
    run_cli("delete-db", "--database-id", db_id, token=auth_token)


def generate_vector(dimension: int = VECTOR_DIMENSION, seed: float = 0.1) -> list[float]:
    """Generate deterministic vector values."""
    return [round(seed + i * 0.01, 4) for i in range(dimension)]
