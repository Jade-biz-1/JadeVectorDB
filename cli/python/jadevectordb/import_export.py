"""
Bulk import/export functionality for JadeVectorDB

This module provides utilities for efficiently importing and exporting
large volumes of vector data in various formats (JSON, CSV).
"""

import json
import csv
import sys
from typing import List, Dict, Optional, Iterator
from pathlib import Path
from .client import JadeVectorDB, JadeVectorDBError

class ImportExportError(Exception):
    """Custom exception for import/export errors"""
    pass

class VectorImporter:
    """
    Handles bulk import of vectors from files
    """

    def __init__(self, client: JadeVectorDB, database_id: str, batch_size: int = 100):
        """
        Initialize the importer

        :param client: JadeVectorDB client instance
        :param database_id: Target database ID
        :param batch_size: Number of vectors to import in each batch
        """
        self.client = client
        self.database_id = database_id
        self.batch_size = batch_size
        self.total_imported = 0
        self.total_errors = 0

    def import_from_json(self, file_path: str, progress_callback=None) -> Dict:
        """
        Import vectors from JSON file

        Expected JSON format:
        [
          {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {...}},
          {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {...}}
        ]

        :param file_path: Path to JSON file
        :param progress_callback: Optional callback function(current, total, status)
        :return: Import statistics dictionary
        """
        try:
            with open(file_path, 'r') as f:
                vectors = json.load(f)

            if not isinstance(vectors, list):
                raise ImportExportError("JSON file must contain an array of vectors")

            return self._import_vectors(vectors, progress_callback)

        except json.JSONDecodeError as e:
            raise ImportExportError(f"Invalid JSON format: {e}")
        except FileNotFoundError:
            raise ImportExportError(f"File not found: {file_path}")
        except Exception as e:
            raise ImportExportError(f"Import failed: {e}")

    def import_from_csv(self, file_path: str, progress_callback=None) -> Dict:
        """
        Import vectors from CSV file

        Expected CSV format:
        id,values,metadata
        vec1,"[0.1,0.2,0.3]","{""key"":""value""}"
        vec2,"[0.4,0.5,0.6]","{""key"":""value""}"

        :param file_path: Path to CSV file
        :param progress_callback: Optional callback function(current, total, status)
        :return: Import statistics dictionary
        """
        vectors = []

        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        vector = {
                            'id': row['id'],
                            'values': json.loads(row['values'])
                        }

                        if 'metadata' in row and row['metadata']:
                            vector['metadata'] = json.loads(row['metadata'])

                        vectors.append(vector)
                    except (json.JSONDecodeError, KeyError) as e:
                        self.total_errors += 1
                        if progress_callback:
                            progress_callback(len(vectors), len(vectors), f"Error parsing row: {e}")

            return self._import_vectors(vectors, progress_callback)

        except FileNotFoundError:
            raise ImportExportError(f"File not found: {file_path}")
        except Exception as e:
            raise ImportExportError(f"Import failed: {e}")

    def _import_vectors(self, vectors: List[Dict], progress_callback=None) -> Dict:
        """
        Import a list of vectors in batches

        :param vectors: List of vector dictionaries
        :param progress_callback: Optional callback function
        :return: Import statistics
        """
        total = len(vectors)
        imported = 0
        errors = 0

        for i in range(0, total, self.batch_size):
            batch = vectors[i:i + self.batch_size]

            for vector in batch:
                try:
                    self.client.store_vector(
                        database_id=self.database_id,
                        vector_id=vector['id'],
                        values=vector['values'],
                        metadata=vector.get('metadata')
                    )
                    imported += 1

                    if progress_callback:
                        progress_callback(imported, total, "importing")

                except JadeVectorDBError as e:
                    errors += 1
                    if progress_callback:
                        progress_callback(imported, total, f"Error: {e}")

        self.total_imported = imported
        self.total_errors = errors

        return {
            'total': total,
            'imported': imported,
            'errors': errors,
            'success_rate': (imported / total * 100) if total > 0 else 0
        }

class VectorExporter:
    """
    Handles bulk export of vectors to files
    """

    def __init__(self, client: JadeVectorDB, database_id: str):
        """
        Initialize the exporter

        :param client: JadeVectorDB client instance
        :param database_id: Source database ID
        """
        self.client = client
        self.database_id = database_id

    def export_to_json(
        self,
        file_path: str,
        vector_ids: Optional[List[str]] = None,
        progress_callback=None
    ) -> Dict:
        """
        Export vectors to JSON file

        :param file_path: Path to output JSON file
        :param vector_ids: Optional list of specific vector IDs to export
        :param progress_callback: Optional callback function(current, total, status)
        :return: Export statistics dictionary
        """
        vectors = []
        exported = 0
        errors = 0

        # If specific IDs provided, export those
        if vector_ids:
            total = len(vector_ids)
            for vector_id in vector_ids:
                try:
                    vector = self.client.retrieve_vector(self.database_id, vector_id)
                    if vector:
                        vectors.append({
                            'id': vector.id,
                            'values': vector.values,
                            'metadata': vector.metadata
                        })
                        exported += 1
                    else:
                        errors += 1

                    if progress_callback:
                        progress_callback(exported, total, "exporting")

                except JadeVectorDBError as e:
                    errors += 1
                    if progress_callback:
                        progress_callback(exported, total, f"Error: {e}")

        # Write to file
        try:
            with open(file_path, 'w') as f:
                json.dump(vectors, f, indent=2)
        except Exception as e:
            raise ImportExportError(f"Failed to write file: {e}")

        return {
            'total': len(vector_ids) if vector_ids else 0,
            'exported': exported,
            'errors': errors,
            'file_path': file_path
        }

    def export_to_csv(
        self,
        file_path: str,
        vector_ids: Optional[List[str]] = None,
        progress_callback=None
    ) -> Dict:
        """
        Export vectors to CSV file

        :param file_path: Path to output CSV file
        :param vector_ids: Optional list of specific vector IDs to export
        :param progress_callback: Optional callback function(current, total, status)
        :return: Export statistics dictionary
        """
        exported = 0
        errors = 0

        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'values', 'metadata'])

                if vector_ids:
                    total = len(vector_ids)
                    for vector_id in vector_ids:
                        try:
                            vector = self.client.retrieve_vector(self.database_id, vector_id)
                            if vector:
                                writer.writerow([
                                    vector.id,
                                    json.dumps(vector.values),
                                    json.dumps(vector.metadata) if vector.metadata else ''
                                ])
                                exported += 1
                            else:
                                errors += 1

                            if progress_callback:
                                progress_callback(exported, total, "exporting")

                        except JadeVectorDBError as e:
                            errors += 1
                            if progress_callback:
                                progress_callback(exported, total, f"Error: {e}")

        except Exception as e:
            raise ImportExportError(f"Failed to write file: {e}")

        return {
            'total': len(vector_ids) if vector_ids else 0,
            'exported': exported,
            'errors': errors,
            'file_path': file_path
        }

def simple_progress_callback(current: int, total: int, status: str):
    """
    Simple progress callback that prints to stdout

    :param current: Current count
    :param total: Total count
    :param status: Status message
    """
    percentage = (current / total * 100) if total > 0 else 0
    print(f"\rProgress: {current}/{total} ({percentage:.1f}%) - {status}", end='', file=sys.stderr)

    if current >= total:
        print(file=sys.stderr)  # New line when complete
