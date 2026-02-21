"""Tests for search and advanced-search CLI commands."""

import json
import uuid

from .conftest import run_cli, parse_json_output, generate_vector, VECTOR_DIMENSION


def _populate_database(auth_token: str, db_id: str, count: int = 5):
    """Store several vectors in the database for search testing."""
    for i in range(count):
        vec_id = f"search_vec_{i}_{uuid.uuid4().hex[:6]}"
        run_cli(
            "store",
            "--database-id", db_id,
            "--vector-id", vec_id,
            "--values", json.dumps(generate_vector(seed=0.1 * (i + 1))),
            "--metadata", json.dumps({"index": i, "category": "even" if i % 2 == 0 else "odd"}),
            token=auth_token,
        )


class TestSearch:
    def test_search_returns_results(self, auth_token, test_database):
        _populate_database(auth_token, test_database)
        query = generate_vector(seed=0.15)
        result = run_cli(
            "search",
            "--database-id", test_database,
            "--query-vector", json.dumps(query),
            "--top-k", "5",
            token=auth_token,
        )
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None

    def test_search_top_k_limits_results(self, auth_token, test_database):
        _populate_database(auth_token, test_database)
        query = generate_vector(seed=0.15)
        result = run_cli(
            "search",
            "--database-id", test_database,
            "--query-vector", json.dumps(query),
            "--top-k", "1",
            token=auth_token,
        )
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None
        # The result should contain at most 1 item
        results_list = data if isinstance(data, list) else data.get("results", data.get("matches", []))
        assert len(results_list) <= 1

    def test_search_on_empty_database(self, auth_token, test_database):
        """Search on a database with no vectors should return empty results."""
        query = generate_vector(seed=0.5)
        result = run_cli(
            "search",
            "--database-id", test_database,
            "--query-vector", json.dumps(query),
            "--top-k", "5",
            token=auth_token,
        )
        # Should succeed but return empty or zero results
        assert result.returncode == 0
        data = parse_json_output(result)
        if data is not None:
            results_list = data if isinstance(data, list) else data.get("results", data.get("matches", []))
            assert len(results_list) == 0


class TestAdvancedSearch:
    def test_advanced_search_with_filters(self, auth_token, test_database):
        _populate_database(auth_token, test_database)
        query = generate_vector(seed=0.15)
        filters = json.dumps({"category": "even"})
        result = run_cli(
            "advanced-search",
            "--database-id", test_database,
            "--query-vector", json.dumps(query),
            "--top-k", "5",
            "--filters", filters,
            token=auth_token,
        )
        assert result.returncode == 0
        data = parse_json_output(result)
        assert data is not None
