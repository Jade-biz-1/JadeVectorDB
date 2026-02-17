"""
Command-line interface for JadeVectorDB
"""

import argparse
import sys
import json
import os
from typing import Dict, List
from .client import JadeVectorDB, Vector, JadeVectorDBError
from .curl_generator import CurlCommandGenerator
from .import_export import VectorImporter, VectorExporter, ImportExportError, simple_progress_callback
from .formatters import print_formatted

def create_database(args: argparse.Namespace):
    """Create a new database"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.create_database(
            name=args.name,
            description=args.description,
            vector_dimension=args.dimension,
            index_type=args.index_type
        )
        print(curl_cmd)
        return
        
    client = JadeVectorDB(args.url, args.api_key)
    try:
        db_id = client.create_database(
            name=args.name,
            description=args.description,
            vector_dimension=args.dimension,
            index_type=args.index_type
        )
        print(f"Created database with ID: {db_id}")
    except JadeVectorDBError as e:
        print(f"Error creating database: {e}")
        sys.exit(1)

def list_databases(args: argparse.Namespace):
    """List all databases"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.list_databases()
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        databases = client.list_databases()
        print_formatted(databases, args.format)
    except JadeVectorDBError as e:
        print(f"Error listing databases: {e}")
        sys.exit(1)

def store_vector(args: argparse.Namespace):
    """Store a vector in the database"""
    if args.curl_only:
        # Generate cURL command instead of executing
        # Parse JSON values if provided as string
        if args.values.startswith('[') and args.values.endswith(']'):
            values = json.loads(args.values)
        else:
            values = [float(x) for x in args.values.split(',')]
            
        # Parse metadata if provided
        metadata = None
        if args.metadata:
            metadata = json.loads(args.metadata)
        
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.store_vector(
            database_id=args.database_id,
            vector_id=args.vector_id,
            values=values,
            metadata=metadata
        )
        print(curl_cmd)
        return
        
    client = JadeVectorDB(args.url, args.api_key)
    try:
        # Parse JSON values if provided as string
        if args.values.startswith('[') and args.values.endswith(']'):
            values = json.loads(args.values)
        else:
            values = [float(x) for x in args.values.split(',')]
            
        # Parse metadata if provided
        metadata = None
        if args.metadata:
            metadata = json.loads(args.metadata)
        
        success = client.store_vector(
            database_id=args.database_id,
            vector_id=args.vector_id,
            values=values,
            metadata=metadata
        )
        
        if success:
            print(f"Successfully stored vector with ID: {args.vector_id}")
        else:
            print("Failed to store vector")
    except JadeVectorDBError as e:
        print(f"Error storing vector: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

def retrieve_vector(args: argparse.Namespace):
    """Retrieve a vector from the database"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.retrieve_vector(
            database_id=args.database_id,
            vector_id=args.vector_id
        )
        print(curl_cmd)
        return
        
    client = JadeVectorDB(args.url, args.api_key)
    try:
        vector = client.retrieve_vector(
            database_id=args.database_id,
            vector_id=args.vector_id
        )
        
        if vector:
            print(json.dumps({
                'id': vector.id,
                'values': vector.values,
                'metadata': vector.metadata
            }, indent=2))
        else:
            print(f"Vector with ID {args.vector_id} not found")
    except JadeVectorDBError as e:
        print(f"Error retrieving vector: {e}")
        sys.exit(1)

def search(args: argparse.Namespace):
    """Perform similarity search"""
    if args.curl_only:
        # Generate cURL command instead of executing
        # Parse JSON values if provided as string
        if args.query_vector.startswith('[') and args.query_vector.endswith(']'):
            query_vector = json.loads(args.query_vector)
        else:
            query_vector = [float(x) for x in args.query_vector.split(',')]
        
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.similarity_search(
            database_id=args.database_id,
            query_vector=query_vector,
            top_k=args.top_k,
            threshold=args.threshold
        )
        print(curl_cmd)
        return
        
    client = JadeVectorDB(args.url, args.api_key)
    try:
        # Parse JSON values if provided as string
        if args.query_vector.startswith('[') and args.query_vector.endswith(']'):
            query_vector = json.loads(args.query_vector)
        else:
            query_vector = [float(x) for x in args.query_vector.split(',')]
        
        results = client.search(
            database_id=args.database_id,
            query_vector=query_vector,
            top_k=args.top_k,
            threshold=args.threshold
        )
        
        print(json.dumps(results, indent=2))
    except JadeVectorDBError as e:
        print(f"Error performing search: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

def get_status(args: argparse.Namespace):
    """Get system status"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.get_status()
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        status = client.get_status()
        print_formatted(status, args.format)
    except JadeVectorDBError as e:
        print(f"Error getting status: {e}")
        sys.exit(1)

def get_health(args: argparse.Namespace):
    """Get system health"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.get_health()
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        health = client.get_health()
        print(json.dumps(health, indent=2))
    except JadeVectorDBError as e:
        print(f"Error getting health: {e}")
        sys.exit(1)

def delete_vector(args: argparse.Namespace):
    """Delete a vector from the database"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.delete_vector(
            database_id=args.database_id,
            vector_id=args.vector_id
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        success = client.delete_vector(
            database_id=args.database_id,
            vector_id=args.vector_id
        )

        if success:
            print(f"Successfully deleted vector with ID: {args.vector_id}")
        else:
            print("Failed to delete vector")
    except JadeVectorDBError as e:
        print(f"Error deleting vector: {e}")
        sys.exit(1)

def get_database(args: argparse.Namespace):
    """Get information about a specific database"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.get_database(database_id=args.database_id)
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        db_info = client.get_database(database_id=args.database_id)
        print(json.dumps(db_info, indent=2))
    except JadeVectorDBError as e:
        print(f"Error getting database: {e}")
        sys.exit(1)

def delete_database(args: argparse.Namespace):
    """Delete a database"""
    if args.curl_only:
        # Generate cURL command instead of executing
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.delete_database(database_id=args.database_id)
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        success = client.delete_database(database_id=args.database_id)

        if success:
            print(f"Successfully deleted database with ID: {args.database_id}")
        else:
            print("Failed to delete database")
    except JadeVectorDBError as e:
        print(f"Error deleting database: {e}")
        sys.exit(1)

# User Management Commands

def user_add(args: argparse.Namespace):
    """Add a new user"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.create_user(
            username=args.username,
            password=args.password if hasattr(args, 'password') and args.password else None,
            roles=[args.role] if args.role else ["user"],
            email=args.email if hasattr(args, 'email') and args.email else None
        )
        print(f"Successfully created user: {args.username}")
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error creating user: {e}")
        sys.exit(1)

def user_list(args: argparse.Namespace):
    """List all users"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        users = client.list_users(
            role=args.role if hasattr(args, 'role') and args.role else None,
            status=args.status if hasattr(args, 'status') and args.status else None
        )
        print_formatted(users, args.format)
    except JadeVectorDBError as e:
        print(f"Error listing users: {e}")
        sys.exit(1)

def user_show(args: argparse.Namespace):
    """Show user details"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.get_user(args.user_id)
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error retrieving user: {e}")
        sys.exit(1)

def user_update(args: argparse.Namespace):
    """Update user information"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        # Convert status string to boolean if provided
        is_active = None
        if hasattr(args, 'status') and args.status:
            is_active = args.status.lower() == 'active'

        # Convert role to roles list if provided
        roles = None
        if hasattr(args, 'role') and args.role:
            roles = [args.role]

        user = client.update_user(
            user_id=args.user_id,
            is_active=is_active,
            roles=roles
        )
        print(f"Successfully updated user: {args.user_id}")
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error updating user: {e}")
        sys.exit(1)

def user_delete(args: argparse.Namespace):
    """Delete a user"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        success = client.delete_user(args.user_id)
        if success:
            print(f"Successfully deleted user: {args.user_id}")
        else:
            print("Failed to delete user")
    except JadeVectorDBError as e:
        print(f"Error deleting user: {e}")
        sys.exit(1)

def user_activate(args: argparse.Namespace):
    """Activate a user"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.activate_user(args.user_id)
        print(f"Successfully activated user: {args.user_id}")
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error activating user: {e}")
        sys.exit(1)

def user_deactivate(args: argparse.Namespace):
    """Deactivate a user"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.deactivate_user(args.user_id)
        print(f"Successfully deactivated user: {args.user_id}")
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error deactivating user: {e}")
        sys.exit(1)

# Import/Export Commands

def import_vectors(args: argparse.Namespace):
    """Import vectors from file"""
    client = JadeVectorDB(args.url, args.api_key)

    try:
        batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size else 100
        importer = VectorImporter(client, args.database_id, batch_size=batch_size)

        print(f"Importing vectors from {args.file}...")

        # Determine format from file extension or explicit format
        file_format = args.format if hasattr(args, 'format') and args.format else None
        if not file_format:
            if args.file.endswith('.json'):
                file_format = 'json'
            elif args.file.endswith('.csv'):
                file_format = 'csv'
            else:
                print("Error: Cannot determine file format. Please specify --format")
                sys.exit(1)

        # Import based on format
        if file_format == 'json':
            stats = importer.import_from_json(args.file, progress_callback=simple_progress_callback)
        elif file_format == 'csv':
            stats = importer.import_from_csv(args.file, progress_callback=simple_progress_callback)
        else:
            print(f"Error: Unsupported format: {file_format}")
            sys.exit(1)

        # Print results
        print(f"\nImport completed:")
        print(f"  Total vectors: {stats['total']}")
        print(f"  Successfully imported: {stats['imported']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")

    except ImportExportError as e:
        print(f"Import error: {e}")
        sys.exit(1)
    except JadeVectorDBError as e:
        print(f"Database error: {e}")
        sys.exit(1)

def export_vectors(args: argparse.Namespace):
    """Export vectors to file"""
    client = JadeVectorDB(args.url, args.api_key)

    try:
        exporter = VectorExporter(client, args.database_id)

        # Parse vector IDs if provided
        vector_ids = None
        if hasattr(args, 'vector_ids') and args.vector_ids:
            if args.vector_ids.startswith('['):
                vector_ids = json.loads(args.vector_ids)
            else:
                vector_ids = [v.strip() for v in args.vector_ids.split(',')]

        print(f"Exporting vectors to {args.file}...")

        # Determine format from file extension or explicit format
        file_format = args.format if hasattr(args, 'format') and args.format else None
        if not file_format:
            if args.file.endswith('.json'):
                file_format = 'json'
            elif args.file.endswith('.csv'):
                file_format = 'csv'
            else:
                print("Error: Cannot determine file format. Please specify --format")
                sys.exit(1)

        # Export based on format
        if file_format == 'json':
            stats = exporter.export_to_json(args.file, vector_ids=vector_ids, progress_callback=simple_progress_callback)
        elif file_format == 'csv':
            stats = exporter.export_to_csv(args.file, vector_ids=vector_ids, progress_callback=simple_progress_callback)
        else:
            print(f"Error: Unsupported format: {file_format}")
            sys.exit(1)

        # Print results
        print(f"\nExport completed:")
        print(f"  Total vectors: {stats['total']}")
        print(f"  Successfully exported: {stats['exported']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Output file: {stats['file_path']}")

    except ImportExportError as e:
        print(f"Export error: {e}")
        sys.exit(1)
    except JadeVectorDBError as e:
        print(f"Database error: {e}")
        sys.exit(1)

# Hybrid Search Commands

def hybrid_search_query(args: argparse.Namespace):
    """Perform hybrid search combining vector and keyword search"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        # Parse query vector if provided
        query_vector = None
        if hasattr(args, 'query_vector') and args.query_vector:
            if args.query_vector.startswith('[') and args.query_vector.endswith(']'):
                query_vector = json.loads(args.query_vector)
            else:
                query_vector = [float(x) for x in args.query_vector.split(',')]

        # Parse query text if provided
        query_text = args.query_text if hasattr(args, 'query_text') and args.query_text else None

        # Validate that at least one query is provided
        if not query_text and not query_vector:
            print("Error: At least one of --query-text or --query-vector must be provided")
            sys.exit(1)

        # Parse filters if provided
        filters = None
        if hasattr(args, 'filters') and args.filters:
            filters = json.loads(args.filters)

        results = client.hybrid_search(
            database_id=args.database_id,
            query_text=query_text,
            query_vector=query_vector,
            top_k=args.top_k if hasattr(args, 'top_k') else 10,
            fusion_method=args.fusion_method if hasattr(args, 'fusion_method') else 'rrf',
            alpha=args.alpha if hasattr(args, 'alpha') else 0.7,
            filters=filters
        )

        print(json.dumps(results, indent=2))
    except JadeVectorDBError as e:
        print(f"Error performing hybrid search: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

def hybrid_search_build(args: argparse.Namespace):
    """Build BM25 index for hybrid search"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        text_field = args.text_field if hasattr(args, 'text_field') and args.text_field else 'text'
        incremental = args.incremental if hasattr(args, 'incremental') else False

        result = client.build_bm25_index(
            database_id=args.database_id,
            text_field=text_field,
            incremental=incremental
        )

        print(f"BM25 index build initiated:")
        print(json.dumps(result, indent=2))
    except JadeVectorDBError as e:
        print(f"Error building BM25 index: {e}")
        sys.exit(1)

def hybrid_search_status(args: argparse.Namespace):
    """Get BM25 index build status"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        status = client.get_bm25_index_status(database_id=args.database_id)

        print("BM25 Index Status:")
        print(json.dumps(status, indent=2))
    except JadeVectorDBError as e:
        print(f"Error getting BM25 index status: {e}")
        sys.exit(1)

def hybrid_search_rebuild(args: argparse.Namespace):
    """Rebuild BM25 index from scratch"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        text_field = args.text_field if hasattr(args, 'text_field') and args.text_field else 'text'

        result = client.rebuild_bm25_index(
            database_id=args.database_id,
            text_field=text_field
        )

        print(f"BM25 index rebuild initiated:")
        print(json.dumps(result, indent=2))
    except JadeVectorDBError as e:
        print(f"Error rebuilding BM25 index: {e}")
        sys.exit(1)

# Database & Vector Additional Commands

def update_database_cmd(args: argparse.Namespace):
    """Update a database"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.update_database(
            database_id=args.database_id,
            name=args.name,
            description=args.description,
            vector_dimension=args.dimension,
            index_type=args.index_type
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        result = client.update_database(
            database_id=args.database_id,
            name=args.name,
            description=args.description,
            vector_dimension=args.dimension,
            index_type=args.index_type
        )
        print(json.dumps(result, indent=2))
    except JadeVectorDBError as e:
        print(f"Error updating database: {e}")
        sys.exit(1)

def list_vectors_cmd(args: argparse.Namespace):
    """List vectors in a database"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.list_vectors(
            database_id=args.database_id,
            limit=args.limit,
            offset=args.offset
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        result = client.list_vectors(
            database_id=args.database_id,
            limit=args.limit,
            offset=args.offset
        )
        print_formatted(result, args.format)
    except JadeVectorDBError as e:
        print(f"Error listing vectors: {e}")
        sys.exit(1)

def update_vector_cmd(args: argparse.Namespace):
    """Update a vector"""
    if args.curl_only:
        if args.values.startswith('[') and args.values.endswith(']'):
            values = json.loads(args.values)
        else:
            values = [float(x) for x in args.values.split(',')]
        metadata = json.loads(args.metadata) if args.metadata else None
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.update_vector(
            database_id=args.database_id,
            vector_id=args.vector_id,
            values=values,
            metadata=metadata
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        if args.values.startswith('[') and args.values.endswith(']'):
            values = json.loads(args.values)
        else:
            values = [float(x) for x in args.values.split(',')]
        metadata = json.loads(args.metadata) if args.metadata else None
        success = client.update_vector(
            database_id=args.database_id,
            vector_id=args.vector_id,
            values=values,
            metadata=metadata
        )
        if success:
            print(f"Successfully updated vector: {args.vector_id}")
    except JadeVectorDBError as e:
        print(f"Error updating vector: {e}")
        sys.exit(1)

def batch_get_cmd(args: argparse.Namespace):
    """Batch get vectors"""
    if args.curl_only:
        vector_ids = json.loads(args.vector_ids) if args.vector_ids.startswith('[') else [v.strip() for v in args.vector_ids.split(',')]
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.batch_get_vectors(
            database_id=args.database_id,
            vector_ids=vector_ids
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        vector_ids = json.loads(args.vector_ids) if args.vector_ids.startswith('[') else [v.strip() for v in args.vector_ids.split(',')]
        results = client.batch_get_vectors(
            database_id=args.database_id,
            vector_ids=vector_ids
        )
        print(json.dumps(results, indent=2))
    except JadeVectorDBError as e:
        print(f"Error batch getting vectors: {e}")
        sys.exit(1)

# Search & Reranking Commands

def advanced_search_cmd(args: argparse.Namespace):
    """Perform advanced search"""
    if args.curl_only:
        if args.query_vector.startswith('[') and args.query_vector.endswith(']'):
            query_vector = json.loads(args.query_vector)
        else:
            query_vector = [float(x) for x in args.query_vector.split(',')]
        filters = json.loads(args.filters) if hasattr(args, 'filters') and args.filters else None
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.advanced_search(
            database_id=args.database_id,
            query_vector=query_vector,
            top_k=args.top_k,
            threshold=args.threshold,
            filters=filters,
            include_metadata=args.include_metadata,
            include_values=args.include_values
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        if args.query_vector.startswith('[') and args.query_vector.endswith(']'):
            query_vector = json.loads(args.query_vector)
        else:
            query_vector = [float(x) for x in args.query_vector.split(',')]
        filters = json.loads(args.filters) if hasattr(args, 'filters') and args.filters else None
        results = client.advanced_search(
            database_id=args.database_id,
            query_vector=query_vector,
            top_k=args.top_k,
            threshold=args.threshold,
            filters=filters,
            include_metadata=args.include_metadata,
            include_values=args.include_values
        )
        print(json.dumps(results, indent=2))
    except JadeVectorDBError as e:
        print(f"Error performing advanced search: {e}")
        sys.exit(1)

def rerank_search_cmd(args: argparse.Namespace):
    """Perform rerank search"""
    if args.curl_only:
        query_vector = None
        if hasattr(args, 'query_vector') and args.query_vector:
            if args.query_vector.startswith('['):
                query_vector = json.loads(args.query_vector)
            else:
                query_vector = [float(x) for x in args.query_vector.split(',')]
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.rerank_search(
            database_id=args.database_id,
            query_text=args.query_text,
            query_vector=query_vector,
            top_k=args.top_k
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        query_vector = None
        if hasattr(args, 'query_vector') and args.query_vector:
            if args.query_vector.startswith('['):
                query_vector = json.loads(args.query_vector)
            else:
                query_vector = [float(x) for x in args.query_vector.split(',')]
        results = client.rerank_search(
            database_id=args.database_id,
            query_text=args.query_text,
            query_vector=query_vector,
            top_k=args.top_k
        )
        print(json.dumps(results, indent=2))
    except JadeVectorDBError as e:
        print(f"Error performing rerank search: {e}")
        sys.exit(1)

def rerank_cmd(args: argparse.Namespace):
    """Standalone reranking"""
    if args.curl_only:
        documents = json.loads(args.documents)
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.rerank(
            query=args.query,
            documents=documents,
            top_k=args.top_k
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        documents = json.loads(args.documents)
        results = client.rerank(
            query=args.query,
            documents=documents,
            top_k=args.top_k
        )
        print(json.dumps(results, indent=2))
    except JadeVectorDBError as e:
        print(f"Error reranking: {e}")
        sys.exit(1)

# Index Management Commands

def create_index_cmd(args: argparse.Namespace):
    """Create an index"""
    if args.curl_only:
        parameters = json.loads(args.parameters) if hasattr(args, 'parameters') and args.parameters else None
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.create_index(
            database_id=args.database_id,
            index_type=args.index_type,
            name=args.name,
            parameters=parameters
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        parameters = json.loads(args.parameters) if hasattr(args, 'parameters') and args.parameters else None
        result = client.create_index(
            database_id=args.database_id,
            index_type=args.index_type,
            name=args.name,
            parameters=parameters
        )
        print(json.dumps(result, indent=2))
    except JadeVectorDBError as e:
        print(f"Error creating index: {e}")
        sys.exit(1)

def list_indexes_cmd(args: argparse.Namespace):
    """List indexes"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.list_indexes(database_id=args.database_id)
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        results = client.list_indexes(database_id=args.database_id)
        print_formatted(results, args.format)
    except JadeVectorDBError as e:
        print(f"Error listing indexes: {e}")
        sys.exit(1)

def delete_index_cmd(args: argparse.Namespace):
    """Delete an index"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.delete_index(
            database_id=args.database_id,
            index_id=args.index_id
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        success = client.delete_index(
            database_id=args.database_id,
            index_id=args.index_id
        )
        if success:
            print(f"Successfully deleted index: {args.index_id}")
    except JadeVectorDBError as e:
        print(f"Error deleting index: {e}")
        sys.exit(1)

# Embedding Commands

def generate_embedding_cmd(args: argparse.Namespace):
    """Generate embeddings"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.generate_embeddings(
            text=args.text,
            input_type=args.input_type,
            model=args.model,
            provider=args.provider
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        result = client.generate_embeddings(
            text=args.text,
            input_type=args.input_type,
            model=args.model,
            provider=args.provider
        )
        print(json.dumps(result, indent=2))
    except JadeVectorDBError as e:
        print(f"Error generating embeddings: {e}")
        sys.exit(1)

# API Key Management Commands

def create_api_key_cmd(args: argparse.Namespace):
    """Create an API key"""
    if args.curl_only:
        permissions = json.loads(args.permissions) if hasattr(args, 'permissions') and args.permissions else None
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.create_api_key(
            user_id=args.user_id,
            description=args.description,
            permissions=permissions,
            validity_days=args.validity_days
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        permissions = json.loads(args.permissions) if hasattr(args, 'permissions') and args.permissions else None
        result = client.create_api_key(
            user_id=args.user_id,
            description=args.description,
            permissions=permissions,
            validity_days=args.validity_days
        )
        print(json.dumps(result, indent=2))
    except JadeVectorDBError as e:
        print(f"Error creating API key: {e}")
        sys.exit(1)

def list_api_keys_cmd(args: argparse.Namespace):
    """List API keys"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.list_api_keys(
            user_id=args.user_id if hasattr(args, 'user_id') and args.user_id else None
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        results = client.list_api_keys(
            user_id=args.user_id if hasattr(args, 'user_id') and args.user_id else None
        )
        print_formatted(results, args.format)
    except JadeVectorDBError as e:
        print(f"Error listing API keys: {e}")
        sys.exit(1)

def revoke_api_key_cmd(args: argparse.Namespace):
    """Revoke an API key"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.revoke_api_key(key_id=args.key_id)
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        success = client.revoke_api_key(key_id=args.key_id)
        if success:
            print(f"Successfully revoked API key: {args.key_id}")
    except JadeVectorDBError as e:
        print(f"Error revoking API key: {e}")
        sys.exit(1)

# Password Management Commands

def change_password_cmd(args: argparse.Namespace):
    """Change user password"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.change_password(
            user_id=args.user_id,
            old_password=args.old_password,
            new_password=args.new_password
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        success = client.change_password(
            user_id=args.user_id,
            old_password=args.old_password,
            new_password=args.new_password
        )
        if success:
            print("Password changed successfully")
    except JadeVectorDBError as e:
        print(f"Error changing password: {e}")
        sys.exit(1)

# Audit & Analytics Commands

def audit_log_cmd(args: argparse.Namespace):
    """View audit log"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.get_audit_log(
            user_id=args.user_id if hasattr(args, 'user_id') and args.user_id else None,
            event_type=args.event_type if hasattr(args, 'event_type') and args.event_type else None,
            limit=args.limit
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        result = client.get_audit_log(
            user_id=args.user_id if hasattr(args, 'user_id') and args.user_id else None,
            event_type=args.event_type if hasattr(args, 'event_type') and args.event_type else None,
            limit=args.limit
        )
        print_formatted(result, args.format)
    except JadeVectorDBError as e:
        print(f"Error getting audit log: {e}")
        sys.exit(1)

def analytics_stats_cmd(args: argparse.Namespace):
    """View analytics stats"""
    if args.curl_only:
        generator = CurlCommandGenerator(args.url, args.api_key)
        curl_cmd = generator.get_analytics_stats(
            database_id=args.database_id,
            granularity=args.granularity,
            start_time=args.start_time if hasattr(args, 'start_time') and args.start_time else None,
            end_time=args.end_time if hasattr(args, 'end_time') and args.end_time else None
        )
        print(curl_cmd)
        return

    client = JadeVectorDB(args.url, args.api_key)
    try:
        result = client.get_analytics_stats(
            database_id=args.database_id,
            start_time=args.start_time if hasattr(args, 'start_time') and args.start_time else None,
            end_time=args.end_time if hasattr(args, 'end_time') and args.end_time else None,
            granularity=args.granularity
        )
        print_formatted(result, args.format)
    except JadeVectorDBError as e:
        print(f"Error getting analytics stats: {e}")
        sys.exit(1)

def setup_parser():
    """Set up the argument parser"""
    parser = argparse.ArgumentParser(description="JadeVectorDB CLI")

    # Get default values from environment variables
    default_url = os.environ.get('JADEVECTORDB_URL', 'http://localhost:8080')
    default_api_key = os.environ.get('JADEVECTORDB_API_KEY')

    parser.add_argument("--url", default=default_url, help=f"JadeVectorDB API URL (default: {default_url}, can be set via JADEVECTORDB_URL env var)")
    parser.add_argument("--api-key", default=default_api_key, help="API key for authentication (can be set via JADEVECTORDB_API_KEY env var)")
    parser.add_argument("--curl-only", action="store_true", help="Generate cURL commands instead of executing")
    parser.add_argument("--format", choices=['json', 'yaml', 'table', 'csv'], default='json', help="Output format (default: json)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create database subcommand
    create_db_parser = subparsers.add_parser("create-db", help="Create a new database")
    create_db_parser.add_argument("--name", required=True, help="Name of the database")
    create_db_parser.add_argument("--description", default="", help="Description of the database")
    create_db_parser.add_argument("--dimension", type=int, default=128, help="Vector dimension (default: 128)")
    create_db_parser.add_argument("--index-type", default="hnsw", help="Index type (default: hnsw)")
    create_db_parser.set_defaults(func=create_database)
    
    # List databases subcommand
    list_db_parser = subparsers.add_parser("list-dbs", help="List all databases")
    list_db_parser.set_defaults(func=list_databases)
    
    # Store vector subcommand
    store_parser = subparsers.add_parser("store", help="Store a vector")
    store_parser.add_argument("--database-id", required=True, help="Database ID")
    store_parser.add_argument("--vector-id", required=True, help="Vector ID")
    store_parser.add_argument("--values", required=True, help="Vector values as JSON array or comma-separated string")
    store_parser.add_argument("--metadata", help="Metadata as JSON string")
    store_parser.set_defaults(func=store_vector)
    
    # Retrieve vector subcommand
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve a vector")
    retrieve_parser.add_argument("--database-id", required=True, help="Database ID")
    retrieve_parser.add_argument("--vector-id", required=True, help="Vector ID")
    retrieve_parser.set_defaults(func=retrieve_vector)
    
    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Perform similarity search")
    search_parser.add_argument("--database-id", required=True, help="Database ID")
    search_parser.add_argument("--query-vector", required=True, help="Query vector as JSON array or comma-separated string")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return (default: 10)")
    search_parser.add_argument("--threshold", type=float, help="Similarity threshold")
    search_parser.set_defaults(func=search)
    
    # Status subcommand
    status_parser = subparsers.add_parser("status", help="Get system status")
    status_parser.set_defaults(func=get_status)
    
    # Health subcommand
    health_parser = subparsers.add_parser("health", help="Get system health")
    health_parser.set_defaults(func=get_health)

    # Delete vector subcommand
    delete_parser = subparsers.add_parser("delete", help="Delete a vector")
    delete_parser.add_argument("--database-id", required=True, help="Database ID")
    delete_parser.add_argument("--vector-id", required=True, help="Vector ID")
    delete_parser.set_defaults(func=delete_vector)

    # Get database subcommand
    get_db_parser = subparsers.add_parser("get-db", help="Get database information")
    get_db_parser.add_argument("--database-id", required=True, help="Database ID")
    get_db_parser.set_defaults(func=get_database)

    # Delete database subcommand
    delete_db_parser = subparsers.add_parser("delete-db", help="Delete a database")
    delete_db_parser.add_argument("--database-id", required=True, help="Database ID")
    delete_db_parser.set_defaults(func=delete_database)

    # User management subcommands
    user_add_parser = subparsers.add_parser("user-add", help="Add a new user")
    user_add_parser.add_argument("username", help="Username")
    user_add_parser.add_argument("--role", required=True, help="User role (admin, developer, user, etc.)")
    user_add_parser.add_argument("--password", help="Password (required)")
    user_add_parser.add_argument("--email", help="Email address (optional)")
    user_add_parser.set_defaults(func=user_add)

    user_list_parser = subparsers.add_parser("user-list", help="List all users")
    user_list_parser.add_argument("--role", help="Filter by role")
    user_list_parser.add_argument("--status", help="Filter by status (active, inactive)")
    user_list_parser.set_defaults(func=user_list)

    user_show_parser = subparsers.add_parser("user-show", help="Show user details")
    user_show_parser.add_argument("user_id", help="User ID")
    user_show_parser.set_defaults(func=user_show)

    user_update_parser = subparsers.add_parser("user-update", help="Update user information")
    user_update_parser.add_argument("user_id", help="User ID")
    user_update_parser.add_argument("--role", help="New role")
    user_update_parser.add_argument("--status", help="New status (active, inactive)")
    user_update_parser.set_defaults(func=user_update)

    user_delete_parser = subparsers.add_parser("user-delete", help="Delete a user")
    user_delete_parser.add_argument("user_id", help="User ID")
    user_delete_parser.set_defaults(func=user_delete)

    user_activate_parser = subparsers.add_parser("user-activate", help="Activate a user")
    user_activate_parser.add_argument("user_id", help="User ID")
    user_activate_parser.set_defaults(func=user_activate)

    user_deactivate_parser = subparsers.add_parser("user-deactivate", help="Deactivate a user")
    user_deactivate_parser.add_argument("user_id", help="User ID")
    user_deactivate_parser.set_defaults(func=user_deactivate)

    # Import/Export subcommands
    import_parser = subparsers.add_parser("import", help="Import vectors from file")
    import_parser.add_argument("--database-id", required=True, help="Target database ID")
    import_parser.add_argument("--file", required=True, help="Input file path")
    import_parser.add_argument("--format", choices=['json', 'csv'], help="File format (auto-detected if not specified)")
    import_parser.add_argument("--batch-size", type=int, default=100, help="Number of vectors per batch (default: 100)")
    import_parser.set_defaults(func=import_vectors)

    export_parser = subparsers.add_parser("export", help="Export vectors to file")
    export_parser.add_argument("--database-id", required=True, help="Source database ID")
    export_parser.add_argument("--file", required=True, help="Output file path")
    export_parser.add_argument("--format", choices=['json', 'csv'], help="File format (auto-detected if not specified)")
    export_parser.add_argument("--vector-ids", help="Comma-separated list or JSON array of vector IDs to export")
    export_parser.set_defaults(func=export_vectors)

    # Hybrid Search subcommands
    hybrid_search_query_parser = subparsers.add_parser("hybrid-search", help="Perform hybrid search combining vector and keyword search")
    hybrid_search_query_parser.add_argument("--database-id", required=True, help="Database ID")
    hybrid_search_query_parser.add_argument("--query-text", help="Query text for BM25 keyword search")
    hybrid_search_query_parser.add_argument("--query-vector", help="Query vector as JSON array or comma-separated string")
    hybrid_search_query_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return (default: 10)")
    hybrid_search_query_parser.add_argument("--fusion-method", choices=['rrf', 'weighted_linear'], default='rrf', help="Score fusion method (default: rrf)")
    hybrid_search_query_parser.add_argument("--alpha", type=float, default=0.7, help="Weight for weighted linear fusion (default: 0.7)")
    hybrid_search_query_parser.add_argument("--filters", help="Metadata filters as JSON string")
    hybrid_search_query_parser.set_defaults(func=hybrid_search_query)

    hybrid_build_parser = subparsers.add_parser("hybrid-build", help="Build BM25 index for hybrid search")
    hybrid_build_parser.add_argument("--database-id", required=True, help="Database ID")
    hybrid_build_parser.add_argument("--text-field", default="text", help="Metadata field to index (default: text)")
    hybrid_build_parser.add_argument("--incremental", action="store_true", help="Perform incremental indexing")
    hybrid_build_parser.set_defaults(func=hybrid_search_build)

    hybrid_status_parser = subparsers.add_parser("hybrid-status", help="Get BM25 index build status")
    hybrid_status_parser.add_argument("--database-id", required=True, help="Database ID")
    hybrid_status_parser.set_defaults(func=hybrid_search_status)

    hybrid_rebuild_parser = subparsers.add_parser("hybrid-rebuild", help="Rebuild BM25 index from scratch")
    hybrid_rebuild_parser.add_argument("--database-id", required=True, help="Database ID")
    hybrid_rebuild_parser.add_argument("--text-field", default="text", help="Metadata field to index (default: text)")
    hybrid_rebuild_parser.set_defaults(func=hybrid_search_rebuild)

    # Update database subcommand
    update_db_parser = subparsers.add_parser("update-db", help="Update a database")
    update_db_parser.add_argument("--database-id", required=True, help="Database ID")
    update_db_parser.add_argument("--name", help="New database name")
    update_db_parser.add_argument("--description", help="New description")
    update_db_parser.add_argument("--dimension", type=int, dest="dimension", help="New vector dimension")
    update_db_parser.add_argument("--index-type", help="New index type")
    update_db_parser.set_defaults(func=update_database_cmd)

    # List vectors subcommand
    list_vectors_parser = subparsers.add_parser("list-vectors", help="List vectors in a database")
    list_vectors_parser.add_argument("--database-id", required=True, help="Database ID")
    list_vectors_parser.add_argument("--limit", type=int, default=50, help="Max vectors to return (default: 50)")
    list_vectors_parser.add_argument("--offset", type=int, default=0, help="Number of vectors to skip (default: 0)")
    list_vectors_parser.set_defaults(func=list_vectors_cmd)

    # Update vector subcommand
    update_vector_parser = subparsers.add_parser("update-vector", help="Update a vector")
    update_vector_parser.add_argument("--database-id", required=True, help="Database ID")
    update_vector_parser.add_argument("--vector-id", required=True, help="Vector ID")
    update_vector_parser.add_argument("--values", required=True, help="New vector values as JSON array or comma-separated")
    update_vector_parser.add_argument("--metadata", help="New metadata as JSON string")
    update_vector_parser.set_defaults(func=update_vector_cmd)

    # Batch get vectors subcommand
    batch_get_parser = subparsers.add_parser("batch-get", help="Batch retrieve vectors by IDs")
    batch_get_parser.add_argument("--database-id", required=True, help="Database ID")
    batch_get_parser.add_argument("--vector-ids", required=True, help="Vector IDs as JSON array or comma-separated")
    batch_get_parser.set_defaults(func=batch_get_cmd)

    # Advanced search subcommand
    adv_search_parser = subparsers.add_parser("advanced-search", help="Perform advanced filtered search")
    adv_search_parser.add_argument("--database-id", required=True, help="Database ID")
    adv_search_parser.add_argument("--query-vector", required=True, help="Query vector as JSON array or comma-separated")
    adv_search_parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    adv_search_parser.add_argument("--threshold", type=float, help="Similarity threshold")
    adv_search_parser.add_argument("--filters", help="Metadata filters as JSON string")
    adv_search_parser.add_argument("--include-metadata", action="store_true", default=True, help="Include metadata (default: true)")
    adv_search_parser.add_argument("--include-values", action="store_true", default=False, help="Include vector values")
    adv_search_parser.set_defaults(func=advanced_search_cmd)

    # Rerank search subcommand
    rerank_search_parser = subparsers.add_parser("rerank-search", help="Search with reranking")
    rerank_search_parser.add_argument("--database-id", required=True, help="Database ID")
    rerank_search_parser.add_argument("--query-text", required=True, help="Query text for reranking")
    rerank_search_parser.add_argument("--query-vector", help="Optional query vector")
    rerank_search_parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    rerank_search_parser.set_defaults(func=rerank_search_cmd)

    # Standalone rerank subcommand
    rerank_parser = subparsers.add_parser("rerank", help="Standalone document reranking")
    rerank_parser.add_argument("--query", required=True, help="Query text")
    rerank_parser.add_argument("--documents", required=True, help="Documents as JSON array of {id, text} objects")
    rerank_parser.add_argument("--top-k", type=int, help="Number of results to return")
    rerank_parser.set_defaults(func=rerank_cmd)

    # Index management subcommands
    create_index_parser = subparsers.add_parser("create-index", help="Create an index")
    create_index_parser.add_argument("--database-id", required=True, help="Database ID")
    create_index_parser.add_argument("--index-type", required=True, help="Index type (HNSW, IVF, LSH, FLAT)")
    create_index_parser.add_argument("--name", help="Index name")
    create_index_parser.add_argument("--parameters", help="Index parameters as JSON string")
    create_index_parser.set_defaults(func=create_index_cmd)

    list_indexes_parser = subparsers.add_parser("list-indexes", help="List indexes for a database")
    list_indexes_parser.add_argument("--database-id", required=True, help="Database ID")
    list_indexes_parser.set_defaults(func=list_indexes_cmd)

    delete_index_parser = subparsers.add_parser("delete-index", help="Delete an index")
    delete_index_parser.add_argument("--database-id", required=True, help="Database ID")
    delete_index_parser.add_argument("--index-id", required=True, help="Index ID")
    delete_index_parser.set_defaults(func=delete_index_cmd)

    # Embedding generation subcommand
    gen_embed_parser = subparsers.add_parser("generate-embedding", help="Generate vector embeddings from text")
    gen_embed_parser.add_argument("--text", required=True, help="Input text to embed")
    gen_embed_parser.add_argument("--input-type", default="text", help="Input type (default: text)")
    gen_embed_parser.add_argument("--model", default="default", help="Embedding model (default: default)")
    gen_embed_parser.add_argument("--provider", default="default", help="Embedding provider (default: default)")
    gen_embed_parser.set_defaults(func=generate_embedding_cmd)

    # API key management subcommands
    create_key_parser = subparsers.add_parser("create-api-key", help="Create a new API key")
    create_key_parser.add_argument("--user-id", required=True, help="User ID")
    create_key_parser.add_argument("--description", default="", help="Key description")
    create_key_parser.add_argument("--permissions", help="Permissions as JSON array")
    create_key_parser.add_argument("--validity-days", type=int, default=0, help="Validity in days (0 = no expiry)")
    create_key_parser.set_defaults(func=create_api_key_cmd)

    list_keys_parser = subparsers.add_parser("list-api-keys", help="List API keys")
    list_keys_parser.add_argument("--user-id", help="Filter by user ID")
    list_keys_parser.set_defaults(func=list_api_keys_cmd)

    revoke_key_parser = subparsers.add_parser("revoke-api-key", help="Revoke an API key")
    revoke_key_parser.add_argument("--key-id", required=True, help="API key ID to revoke")
    revoke_key_parser.set_defaults(func=revoke_api_key_cmd)

    # Password management subcommand
    change_pw_parser = subparsers.add_parser("change-password", help="Change user password")
    change_pw_parser.add_argument("--user-id", required=True, help="User ID")
    change_pw_parser.add_argument("--old-password", required=True, help="Current password")
    change_pw_parser.add_argument("--new-password", required=True, help="New password")
    change_pw_parser.set_defaults(func=change_password_cmd)

    # Audit log subcommand
    audit_parser = subparsers.add_parser("audit-log", help="View audit log")
    audit_parser.add_argument("--user-id", help="Filter by user ID")
    audit_parser.add_argument("--event-type", help="Filter by event type")
    audit_parser.add_argument("--limit", type=int, default=100, help="Max entries (default: 100)")
    audit_parser.set_defaults(func=audit_log_cmd)

    # Analytics stats subcommand
    analytics_parser = subparsers.add_parser("analytics-stats", help="View analytics statistics")
    analytics_parser.add_argument("--database-id", required=True, help="Database ID")
    analytics_parser.add_argument("--granularity", default="hourly", choices=['hourly', 'daily', 'weekly'], help="Time granularity (default: hourly)")
    analytics_parser.add_argument("--start-time", help="Start time (ISO 8601)")
    analytics_parser.add_argument("--end-time", help="End time (ISO 8601)")
    analytics_parser.set_defaults(func=analytics_stats_cmd)

    return parser

def main():
    """Main entry point for the CLI"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()