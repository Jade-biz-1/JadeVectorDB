"""Tests for vector CRUD CLI commands."""

import json
import uuid

from .conftest import run_cli, parse_json_output, generate_vector, VECTOR_DIMENSION


class TestStoreVector:
    def test_store_vector(self, auth_token, test_database):
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        values = generate_vector()
        result = run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(values),
            token=auth_token,
        )
        assert result.returncode == 0
        assert "Successfully stored" in result.stdout

    def test_store_with_metadata(self, auth_token, test_database):
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        values = generate_vector()
        metadata = json.dumps({"source": "pytest", "category": "test"})
        result = run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(values),
            "--metadata", metadata,
            token=auth_token,
        )
        assert result.returncode == 0
        assert "Successfully stored" in result.stdout

    def test_store_wrong_dimension_fails(self, auth_token, test_database):
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        wrong_values = generate_vector(dimension=5)  # DB expects 128
        result = run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(wrong_values),
            token=auth_token,
        )
        assert result.returncode != 0 or "error" in result.stdout.lower()


class TestRetrieveVector:
    def test_retrieve_stored_vector(self, auth_token, test_database):
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        values = generate_vector(seed=0.5)
        run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(values),
            token=auth_token,
        )

        result = run_cli(
            "retrieve",
            "--database-id", test_database,
            "--vector-id", vec_id,
            token=auth_token,
        )
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None
        assert data["id"] == vec_id
        assert len(data["values"]) == VECTOR_DIMENSION

    def test_retrieve_nonexistent_vector(self, auth_token, test_database):
        result = run_cli(
            "retrieve",
            "--database-id", test_database,
            "--vector-id", "nonexistent_vec",
            token=auth_token,
        )
        # Should either return non-zero or print "not found"
        assert result.returncode != 0 or "not found" in result.stdout.lower()


class TestListVectors:
    def test_list_vectors(self, auth_token, test_database):
        # Store a vector first
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(generate_vector()),
            token=auth_token,
        )

        result = run_cli(
            "list-vectors",
            "--database-id", test_database,
            token=auth_token,
        )
        assert result.returncode == 0
        assert vec_id in result.stdout


class TestUpdateVector:
    def test_update_vector_values(self, auth_token, test_database):
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        original = generate_vector(seed=0.1)
        run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(original),
            token=auth_token,
        )

        updated = generate_vector(seed=0.9)
        result = run_cli(
            "update-vector",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(updated),
            token=auth_token,
        )
        assert result.returncode == 0
        assert "Successfully updated" in result.stdout

        # Verify the update
        retrieve = run_cli(
            "retrieve",
            "--database-id", test_database,
            "--vector-id", vec_id,
            token=auth_token,
        )
        data = parse_json_output(retrieve)
        assert data is not None
        # First value should be close to 0.9 (the updated seed)
        assert abs(data["values"][0] - 0.9) < 0.01


class TestDeleteVector:
    def test_delete_vector(self, auth_token, test_database):
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(generate_vector()),
            token=auth_token,
        )

        result = run_cli(
            "delete",
            "--database-id", test_database,
            "--vector-id", vec_id,
            token=auth_token,
        )
        assert result.returncode == 0
        assert "Successfully deleted" in result.stdout

    def test_retrieve_after_delete_fails(self, auth_token, test_database):
        vec_id = f"vec_{uuid.uuid4().hex[:8]}"
        run_cli(
            "store",
            "--database-id", test_database,
            "--vector-id", vec_id,
            "--values", json.dumps(generate_vector()),
            token=auth_token,
        )
        run_cli(
            "delete",
            "--database-id", test_database,
            "--vector-id", vec_id,
            token=auth_token,
        )

        result = run_cli(
            "retrieve",
            "--database-id", test_database,
            "--vector-id", vec_id,
            token=auth_token,
        )
        assert result.returncode != 0 or "not found" in result.stdout.lower()


class TestBatchGet:
    def test_batch_get_multiple_vectors(self, auth_token, test_database):
        vec_ids = []
        for i in range(3):
            vec_id = f"vec_batch_{uuid.uuid4().hex[:8]}"
            vec_ids.append(vec_id)
            run_cli(
                "store",
                "--database-id", test_database,
                "--vector-id", vec_id,
                "--values", json.dumps(generate_vector(seed=0.1 * (i + 1))),
                token=auth_token,
            )

        result = run_cli(
            "batch-get",
            "--database-id", test_database,
            "--vector-ids", json.dumps(vec_ids),
            token=auth_token,
        )
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None
