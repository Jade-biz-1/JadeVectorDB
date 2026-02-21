"""Tests for database CRUD CLI commands."""

import json
import uuid

from .conftest import run_cli, parse_json_output, VECTOR_DIMENSION


class TestCreateDatabase:
    def test_create_and_get_database(self, auth_token):
        db_name = f"pytest_db_{uuid.uuid4().hex[:8]}"
        result = run_cli(
            "create-db",
            "--name", db_name,
            "--dimension", str(VECTOR_DIMENSION),
            "--index-type", "hnsw",
            token=auth_token,
        )
        assert result.returncode == 0
        assert "Created database with ID:" in result.stdout
        db_id = result.stdout.strip().split("ID: ")[-1].strip()
        assert db_id

        # Verify via get-db
        get_result = run_cli("get-db", "--database-id", db_id, token=auth_token)
        assert get_result.returncode == 0
        data = parse_json_output(get_result)
        assert data is not None

        # Cleanup
        run_cli("delete-db", "--database-id", db_id, token=auth_token)

    def test_create_db_missing_name_fails(self, auth_token):
        """create-db without --name should fail (argparse required arg)."""
        result = run_cli("create-db", "--dimension", "128", token=auth_token)
        assert result.returncode != 0


class TestListDatabases:
    def test_list_dbs_includes_created(self, auth_token, test_database):
        result = run_cli("list-dbs", token=auth_token)
        assert result.returncode == 0
        # The created database should appear somewhere in the output
        assert test_database in result.stdout


class TestGetDatabase:
    def test_get_db_returns_info(self, auth_token, test_database):
        result = run_cli("get-db", "--database-id", test_database, token=auth_token)
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None

    def test_get_db_nonexistent_fails(self, auth_token):
        result = run_cli("get-db", "--database-id", "nonexistent_db_id", token=auth_token)
        assert result.returncode != 0 or "error" in result.stdout.lower()


class TestUpdateDatabase:
    def test_update_description(self, auth_token, test_database):
        new_desc = "Updated by pytest"
        result = run_cli(
            "update-db",
            "--database-id", test_database,
            "--description", new_desc,
            token=auth_token,
        )
        assert result.returncode == 0

        # Verify the update
        get_result = run_cli("get-db", "--database-id", test_database, token=auth_token)
        assert new_desc in get_result.stdout


class TestDeleteDatabase:
    def test_delete_database(self, auth_token):
        # Create a database to delete
        db_name = f"pytest_del_{uuid.uuid4().hex[:8]}"
        create_result = run_cli(
            "create-db", "--name", db_name, "--dimension", "128",
            token=auth_token,
        )
        assert create_result.returncode == 0
        db_id = create_result.stdout.strip().split("ID: ")[-1].strip()

        # Delete it
        del_result = run_cli("delete-db", "--database-id", db_id, token=auth_token)
        assert del_result.returncode == 0
        assert "Successfully deleted" in del_result.stdout

    def test_get_deleted_database_fails(self, auth_token):
        # Create and immediately delete
        db_name = f"pytest_del2_{uuid.uuid4().hex[:8]}"
        create_result = run_cli(
            "create-db", "--name", db_name, "--dimension", "128",
            token=auth_token,
        )
        db_id = create_result.stdout.strip().split("ID: ")[-1].strip()
        run_cli("delete-db", "--database-id", db_id, token=auth_token)

        # Now get should fail
        get_result = run_cli("get-db", "--database-id", db_id, token=auth_token)
        assert get_result.returncode != 0 or "error" in get_result.stdout.lower()
