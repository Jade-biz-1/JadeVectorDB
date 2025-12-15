#!/usr/bin/env python3
"""
Import/Export CLI Tests (T268)

Comprehensive test suite for bulk import/export functionality in Python CLI.
Tests import from JSON/CSV, export to JSON/CSV, progress tracking, and error handling.
"""

import sys
import os
import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.jadevectordb.import_export import (
    VectorImporter,
    VectorExporter,
    ImportExportError,
    simple_progress_callback
)
from python.jadevectordb.client import JadeVectorDB


class TestVectorImporter:
    """Test VectorImporter class"""

    @pytest.fixture
    def mock_client(self):
        """Create a mock JadeVectorDB client"""
        client = Mock(spec=JadeVectorDB)
        return client

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vector data for testing"""
        return [
            {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"type": "test"}},
            {"id": "vec2", "values": [0.4, 0.5, 0.6], "metadata": {"type": "test"}},
            {"id": "vec3", "values": [0.7, 0.8, 0.9], "metadata": {"type": "test"}}
        ]

    @pytest.fixture
    def temp_json_file(self, sample_vectors):
        """Create a temporary JSON file with vector data"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(sample_vectors, temp_file)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)

    def test_importer_initialization(self, mock_client):
        """Test VectorImporter initialization"""
        importer = VectorImporter(mock_client, "test-db", batch_size=50)

        assert importer.client == mock_client
        assert importer.database_id == "test-db"
        assert importer.batch_size == 50
        assert importer.total_imported == 0
        assert importer.total_errors == 0

    def test_import_from_json_success(self, mock_client, temp_json_file):
        """Test successful import from JSON file"""
        mock_client.store_vector.return_value = True

        importer = VectorImporter(mock_client, "test-db", batch_size=100)
        result = importer.import_from_json(temp_json_file)

        assert result['total'] == 3
        assert result['imported'] == 3
        assert result['errors'] == 0
        assert result['success_rate'] == 100.0
        assert mock_client.store_vector.call_count == 3

    def test_import_from_json_with_progress(self, mock_client, temp_json_file):
        """Test import with progress callback"""
        mock_client.store_vector.return_value = True
        progress_calls = []

        def track_progress(imported, total, operation):
            progress_calls.append((imported, total, operation))

        importer = VectorImporter(mock_client, "test-db")
        result = importer.import_from_json(temp_json_file, progress_callback=track_progress)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3, "importing")
        assert result['imported'] == 3

    def test_import_from_json_with_errors(self, mock_client, temp_json_file):
        """Test import with some vectors failing"""
        # First vector succeeds, second fails, third succeeds
        mock_client.store_vector.side_effect = [True, Exception("Store failed"), True]

        importer = VectorImporter(mock_client, "test-db")
        result = importer.import_from_json(temp_json_file)

        assert result['total'] == 3
        assert result['imported'] == 2
        assert result['errors'] == 1
        assert result['success_rate'] == pytest.approx(66.67, rel=0.1)

    def test_import_from_json_file_not_found(self, mock_client):
        """Test import with non-existent file"""
        importer = VectorImporter(mock_client, "test-db")

        with pytest.raises(FileNotFoundError):
            importer.import_from_json("nonexistent.json")

    def test_import_from_json_invalid_json(self, mock_client):
        """Test import with invalid JSON file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.write("{ invalid json")
        temp_file.close()

        try:
            importer = VectorImporter(mock_client, "test-db")
            with pytest.raises(json.JSONDecodeError):
                importer.import_from_json(temp_file.name)
        finally:
            os.unlink(temp_file.name)

    def test_import_from_csv_success(self, mock_client):
        """Test successful import from CSV file"""
        mock_client.store_vector.return_value = True

        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        temp_file.write('id,values,metadata\n')
        temp_file.write('vec1,"[0.1, 0.2, 0.3]","{""type"": ""test""}"\n')
        temp_file.write('vec2,"[0.4, 0.5, 0.6]","{""type"": ""test""}"\n')
        temp_file.close()

        try:
            importer = VectorImporter(mock_client, "test-db")
            result = importer.import_from_csv(temp_file.name)

            assert result['total'] == 2
            assert result['imported'] == 2
            assert result['errors'] == 0
        finally:
            os.unlink(temp_file.name)

    def test_import_batch_processing(self, mock_client, sample_vectors):
        """Test that vectors are processed in batches"""
        mock_client.store_vector.return_value = True

        # Create file with 10 vectors
        large_vector_set = sample_vectors * 4  # 12 vectors
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(large_vector_set, temp_file)
        temp_file.close()

        try:
            importer = VectorImporter(mock_client, "test-db", batch_size=5)
            result = importer.import_from_json(temp_file.name)

            assert result['total'] == 12
            assert result['imported'] == 12
            # Verify batching by checking store calls
            assert mock_client.store_vector.call_count == 12
        finally:
            os.unlink(temp_file.name)

    def test_import_empty_file(self, mock_client):
        """Test import with empty JSON array"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump([], temp_file)
        temp_file.close()

        try:
            importer = VectorImporter(mock_client, "test-db")
            result = importer.import_from_json(temp_file.name)

            assert result['total'] == 0
            assert result['imported'] == 0
            assert result['errors'] == 0
        finally:
            os.unlink(temp_file.name)


class TestVectorExporter:
    """Test VectorExporter class"""

    @pytest.fixture
    def mock_client(self):
        """Create a mock JadeVectorDB client"""
        client = Mock(spec=JadeVectorDB)
        return client

    @pytest.fixture
    def sample_vectors(self):
        """Create sample vector data for testing"""
        return [
            {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"type": "test"}},
            {"id": "vec2", "values": [0.4, 0.5, 0.6], "metadata": {"type": "test"}},
            {"id": "vec3", "values": [0.7, 0.8, 0.9], "metadata": {"type": "test"}}
        ]

    def test_exporter_initialization(self, mock_client):
        """Test VectorExporter initialization"""
        exporter = VectorExporter(mock_client, "test-db")

        assert exporter.client == mock_client
        assert exporter.database_id == "test-db"
        assert exporter.total_exported == 0

    def test_export_to_json_success(self, mock_client, sample_vectors):
        """Test successful export to JSON file"""
        mock_client.list_vectors.return_value = sample_vectors

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.close()

        try:
            exporter = VectorExporter(mock_client, "test-db")
            result = exporter.export_to_json(temp_file.name)

            assert result['total'] == 3
            assert result['exported'] == 3
            assert result['success_rate'] == 100.0

            # Verify file contents
            with open(temp_file.name, 'r') as f:
                exported_data = json.load(f)
            assert len(exported_data) == 3
            assert exported_data[0]['id'] == 'vec1'
        finally:
            os.unlink(temp_file.name)

    def test_export_to_json_with_progress(self, mock_client, sample_vectors):
        """Test export with progress callback"""
        mock_client.list_vectors.return_value = sample_vectors
        progress_calls = []

        def track_progress(exported, total, operation):
            progress_calls.append((exported, total, operation))

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.close()

        try:
            exporter = VectorExporter(mock_client, "test-db")
            exporter.export_to_json(temp_file.name, progress_callback=track_progress)

            assert len(progress_calls) > 0
            assert progress_calls[-1][2] == "exporting"
        finally:
            os.unlink(temp_file.name)

    def test_export_to_csv_success(self, mock_client, sample_vectors):
        """Test successful export to CSV file"""
        mock_client.list_vectors.return_value = sample_vectors

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        temp_file.close()

        try:
            exporter = VectorExporter(mock_client, "test-db")
            result = exporter.export_to_csv(temp_file.name)

            assert result['total'] == 3
            assert result['exported'] == 3

            # Verify file has CSV header
            with open(temp_file.name, 'r') as f:
                first_line = f.readline()
            assert 'id' in first_line.lower()
            assert 'values' in first_line.lower()
        finally:
            os.unlink(temp_file.name)

    def test_export_empty_database(self, mock_client):
        """Test export with no vectors"""
        mock_client.list_vectors.return_value = []

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.close()

        try:
            exporter = VectorExporter(mock_client, "test-db")
            result = exporter.export_to_json(temp_file.name)

            assert result['total'] == 0
            assert result['exported'] == 0

            # Verify file has empty array
            with open(temp_file.name, 'r') as f:
                data = json.load(f)
            assert data == []
        finally:
            os.unlink(temp_file.name)

    def test_export_with_filter(self, mock_client, sample_vectors):
        """Test export with vector ID filter"""
        # Only return filtered vectors
        filtered_vectors = [v for v in sample_vectors if v['id'] in ['vec1', 'vec3']]
        mock_client.list_vectors.return_value = filtered_vectors

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.close()

        try:
            exporter = VectorExporter(mock_client, "test-db")
            result = exporter.export_to_json(temp_file.name, vector_ids=['vec1', 'vec3'])

            assert result['total'] == 2
            assert result['exported'] == 2
        finally:
            os.unlink(temp_file.name)


class TestProgressCallback:
    """Test progress callback function"""

    def test_simple_progress_callback(self, capsys):
        """Test simple progress callback output"""
        simple_progress_callback(50, 100, "importing")

        captured = capsys.readouterr()
        assert "importing" in captured.out.lower()
        assert "50" in captured.out or "50%" in captured.out

    def test_progress_callback_completion(self, capsys):
        """Test progress callback at completion"""
        simple_progress_callback(100, 100, "importing")

        captured = capsys.readouterr()
        assert "100" in captured.out


class TestImportExportIntegration:
    """Integration tests for import/export workflow"""

    @pytest.fixture
    def mock_client(self):
        """Create a mock JadeVectorDB client"""
        client = Mock(spec=JadeVectorDB)
        return client

    def test_export_then_import(self, mock_client):
        """Test complete export-import round trip"""
        # Setup initial vectors
        original_vectors = [
            {"id": "v1", "values": [1.0, 2.0], "metadata": {"tag": "A"}},
            {"id": "v2", "values": [3.0, 4.0], "metadata": {"tag": "B"}}
        ]
        mock_client.list_vectors.return_value = original_vectors
        mock_client.store_vector.return_value = True

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_file.close()

        try:
            # Export
            exporter = VectorExporter(mock_client, "source-db")
            export_result = exporter.export_to_json(temp_file.name)
            assert export_result['exported'] == 2

            # Import to different database
            importer = VectorImporter(mock_client, "target-db")
            import_result = importer.import_from_json(temp_file.name)
            assert import_result['imported'] == 2

            # Verify vectors were stored
            assert mock_client.store_vector.call_count == 2
        finally:
            os.unlink(temp_file.name)

    def test_large_dataset_import_export(self, mock_client):
        """Test with large dataset (1000+ vectors)"""
        # Generate large dataset
        large_dataset = [
            {"id": f"vec{i}", "values": [float(i), float(i+1)], "metadata": {"idx": i}}
            for i in range(1000)
        ]

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(large_dataset, temp_file)
        temp_file.close()

        try:
            mock_client.store_vector.return_value = True

            importer = VectorImporter(mock_client, "test-db", batch_size=100)
            result = importer.import_from_json(temp_file.name)

            assert result['total'] == 1000
            assert result['imported'] == 1000
            assert result['success_rate'] == 100.0
        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
