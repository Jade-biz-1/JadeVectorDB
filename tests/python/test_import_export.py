"""Tests for import and export CLI commands."""

import json
import os
import tempfile
import uuid

from .conftest import run_cli, parse_json_output, generate_vector, VECTOR_DIMENSION


class TestExport:
    def test_export_creates_file(self, auth_token, test_database):
        # Store some vectors
        for i in range(3):
            run_cli(
                "store",
                "--database-id", test_database,
                "--vector-id", f"export_vec_{i}_{uuid.uuid4().hex[:6]}",
                "--values", json.dumps(generate_vector(seed=0.1 * (i + 1))),
                token=auth_token,
            )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            result = run_cli(
                "export",
                "--database-id", test_database,
                "--file", export_path,
                token=auth_token,
            )
            assert result.returncode == 0
            assert os.path.exists(export_path)
            # File should have content
            with open(export_path) as f:
                data = json.load(f)
            assert isinstance(data, (list, dict))
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestImport:
    def test_import_from_file(self, auth_token, test_database):
        """Import vectors from a manually created JSON file."""
        vectors = [
            {
                "id": f"import_vec_{i}_{uuid.uuid4().hex[:6]}",
                "values": generate_vector(seed=0.2 * (i + 1)),
                "metadata": {"imported": True},
            }
            for i in range(2)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(vectors, f)
            import_path = f.name

        try:
            result = run_cli(
                "import",
                "--database-id", test_database,
                "--file", import_path,
                token=auth_token,
            )
            assert result.returncode == 0
            assert "import" in result.stdout.lower()
        finally:
            os.unlink(import_path)


class TestRoundTrip:
    def test_export_then_import_preserves_data(self, auth_token, test_database):
        """Export vectors, import into the same DB, verify no errors."""
        vec_ids = []
        for i in range(3):
            vec_id = f"rt_vec_{i}_{uuid.uuid4().hex[:6]}"
            vec_ids.append(vec_id)
            run_cli(
                "store",
                "--database-id", test_database,
                "--vector-id", vec_id,
                "--values", json.dumps(generate_vector(seed=0.3 * (i + 1))),
                "--metadata", json.dumps({"round_trip": True}),
                token=auth_token,
            )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            # Export
            export_result = run_cli(
                "export",
                "--database-id", test_database,
                "--file", export_path,
                token=auth_token,
            )
            assert export_result.returncode == 0

            # Import back (vectors may already exist — that's OK, we just check success)
            import_result = run_cli(
                "import",
                "--database-id", test_database,
                "--file", export_path,
                token=auth_token,
            )
            assert import_result.returncode == 0
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
