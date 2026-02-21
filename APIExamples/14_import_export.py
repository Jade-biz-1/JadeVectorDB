#!/usr/bin/env python3
"""
JadeVectorDB API Example — Import & Export
=============================================

For bulk data operations, JadeVectorDB provides import/export utilities
that handle large volumes of vectors efficiently with batching, progress
tracking, and error handling.

Supported formats:
  - JSON: Array of {id, values, metadata} objects
  - CSV:  id, values (JSON array), metadata (JSON string)

This example demonstrates:
  1. Export vectors from a database to a JSON file
  2. Export vectors to a CSV file
  3. Import vectors from a JSON file
  4. Import vectors from a CSV file
  5. Selective export (specific vector IDs)
  6. Progress tracking with callbacks

APIs covered:
  - VectorExporter(client, database_id)
  - exporter.export_to_json(file_path, vector_ids, progress_callback)
  - exporter.export_to_csv(file_path, vector_ids, progress_callback)
  - VectorImporter(client, database_id, batch_size)
  - importer.import_from_json(file_path, progress_callback)
  - importer.import_from_csv(file_path, progress_callback)
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError
from jadevectordb.import_export import (
    VectorImporter,
    VectorExporter,
    ImportExportError,
    simple_progress_callback,
)

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# --- Setup: create database and populate with sample vectors ---
DIMENSION = 128
db_id = client.create_database(
    name="import-export-example",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created database: {db_id}")


def make_vector(seed: float) -> list:
    return [round(seed + i * 0.001, 6) for i in range(DIMENSION)]


# Populate with sample data
for i in range(20):
    client.store_vector(
        db_id,
        f"product-{i:03d}",
        make_vector(0.01 * i),
        {"name": f"Product {i}", "category": "electronics" if i % 2 == 0 else "clothing"},
    )
print(f"Stored 20 vectors\n")

# Create a temp directory for our export files
tmp_dir = tempfile.mkdtemp(prefix="jadevectordb_export_")
json_path = os.path.join(tmp_dir, "vectors.json")
csv_path = os.path.join(tmp_dir, "vectors.csv")

# ---------------------------------------------------------------------------
# 1. Export to JSON
# ---------------------------------------------------------------------------
# Exports all vectors (or a subset) to a JSON file. The output format is:
# [
#   {"id": "vec-1", "values": [0.1, 0.2, ...], "metadata": {...}},
#   ...
# ]

print("=== 1. Export to JSON ===")
try:
    exporter = VectorExporter(client, db_id)
    stats = exporter.export_to_json(
        json_path,
        progress_callback=simple_progress_callback,
    )
    print(f"\nExport stats:")
    print(f"  Total    : {stats['total']}")
    print(f"  Exported : {stats['exported']}")
    print(f"  Errors   : {stats['errors']}")
    print(f"  File     : {stats['file_path']}")
except ImportExportError as e:
    print(f"JSON export: {e}")

# ---------------------------------------------------------------------------
# 2. Export to CSV
# ---------------------------------------------------------------------------
# CSV format: id, values (as JSON string), metadata (as JSON string)

print("\n=== 2. Export to CSV ===")
try:
    exporter = VectorExporter(client, db_id)
    stats = exporter.export_to_csv(
        csv_path,
        progress_callback=simple_progress_callback,
    )
    print(f"\nCSV export: {stats['exported']} vectors to {stats['file_path']}")
except ImportExportError as e:
    print(f"CSV export: {e}")

# ---------------------------------------------------------------------------
# 3. Selective export (specific IDs)
# ---------------------------------------------------------------------------
# Export only specific vectors by providing a list of IDs.

print("\n=== 3. Selective Export ===")
selective_path = os.path.join(tmp_dir, "selective.json")
try:
    exporter = VectorExporter(client, db_id)
    stats = exporter.export_to_json(
        selective_path,
        vector_ids=["product-000", "product-005", "product-010"],
    )
    print(f"Exported {stats['exported']} selected vectors")
except ImportExportError as e:
    print(f"Selective export: {e}")

# ---------------------------------------------------------------------------
# 4. Import from JSON
# ---------------------------------------------------------------------------
# Import vectors from a JSON file into a database. Vectors are imported
# in configurable batches for efficiency.

print("\n=== 4. Import from JSON ===")
# First, create a fresh database to import into
import_db_id = client.create_database(
    name="import-target",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created import target database: {import_db_id}")

try:
    importer = VectorImporter(
        client,
        import_db_id,
        batch_size=10,           # Import 10 vectors per batch
    )
    stats = importer.import_from_json(
        json_path,
        progress_callback=simple_progress_callback,
    )
    print(f"\nImport stats:")
    print(f"  Total    : {stats['total']}")
    print(f"  Imported : {stats['imported']}")
    print(f"  Errors   : {stats['errors']}")
    print(f"  Rate     : {stats['success_rate']:.1f}%")
except ImportExportError as e:
    print(f"JSON import: {e}")

# ---------------------------------------------------------------------------
# 5. Import from CSV
# ---------------------------------------------------------------------------

print("\n=== 5. Import from CSV ===")
csv_import_db = client.create_database(
    name="csv-import-target",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created CSV import target: {csv_import_db}")

try:
    importer = VectorImporter(client, csv_import_db, batch_size=10)
    stats = importer.import_from_csv(
        csv_path,
        progress_callback=simple_progress_callback,
    )
    print(f"\nCSV import: {stats['imported']} vectors imported")
except ImportExportError as e:
    print(f"CSV import: {e}")

# ---------------------------------------------------------------------------
# 6. Custom progress callback
# ---------------------------------------------------------------------------
# You can provide your own progress callback for integration with progress
# bars, logging frameworks, or web UIs.

print("\n=== 6. Custom Progress Callback ===")

def my_progress(current, total, status):
    """Custom progress callback for integration with logging systems."""
    pct = (current / total * 100) if total > 0 else 0
    print(f"  [{pct:5.1f}%] {current}/{total} — {status}")

try:
    exporter = VectorExporter(client, db_id)
    stats = exporter.export_to_json(
        os.path.join(tmp_dir, "progress_demo.json"),
        progress_callback=my_progress,
    )
    print(f"  Done: {stats['exported']} vectors")
except ImportExportError as e:
    print(f"Progress demo: {e}")

# --- Cleanup ---
for did in [db_id, import_db_id, csv_import_db]:
    try:
        client.delete_database(database_id=did)
    except JadeVectorDBError:
        pass

# Clean up temp files
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"\nCleaned up databases and temp files")
print("Import/export examples complete.")
