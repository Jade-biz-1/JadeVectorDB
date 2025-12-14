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
        print(json.dumps(databases, indent=2))
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
        print(json.dumps(status, indent=2))
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
            email=args.email,
            role=args.role,
            password=args.password if hasattr(args, 'password') else None
        )
        print(f"Successfully created user: {args.email}")
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
        print(json.dumps(users, indent=2))
    except JadeVectorDBError as e:
        print(f"Error listing users: {e}")
        sys.exit(1)

def user_show(args: argparse.Namespace):
    """Show user details"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.get_user(email=args.email)
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error retrieving user: {e}")
        sys.exit(1)

def user_update(args: argparse.Namespace):
    """Update user information"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.update_user(
            email=args.email,
            role=args.role if hasattr(args, 'role') and args.role else None,
            status=args.status if hasattr(args, 'status') and args.status else None
        )
        print(f"Successfully updated user: {args.email}")
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error updating user: {e}")
        sys.exit(1)

def user_delete(args: argparse.Namespace):
    """Delete a user"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        success = client.delete_user(email=args.email)
        if success:
            print(f"Successfully deleted user: {args.email}")
        else:
            print("Failed to delete user")
    except JadeVectorDBError as e:
        print(f"Error deleting user: {e}")
        sys.exit(1)

def user_activate(args: argparse.Namespace):
    """Activate a user"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.activate_user(email=args.email)
        print(f"Successfully activated user: {args.email}")
        print(json.dumps(user, indent=2))
    except JadeVectorDBError as e:
        print(f"Error activating user: {e}")
        sys.exit(1)

def user_deactivate(args: argparse.Namespace):
    """Deactivate a user"""
    client = JadeVectorDB(args.url, args.api_key)
    try:
        user = client.deactivate_user(email=args.email)
        print(f"Successfully deactivated user: {args.email}")
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

def setup_parser():
    """Set up the argument parser"""
    parser = argparse.ArgumentParser(description="JadeVectorDB CLI")

    # Get default values from environment variables
    default_url = os.environ.get('JADEVECTORDB_URL', 'http://localhost:8080')
    default_api_key = os.environ.get('JADEVECTORDB_API_KEY')

    parser.add_argument("--url", default=default_url, help=f"JadeVectorDB API URL (default: {default_url}, can be set via JADEVECTORDB_URL env var)")
    parser.add_argument("--api-key", default=default_api_key, help="API key for authentication (can be set via JADEVECTORDB_API_KEY env var)")
    parser.add_argument("--curl-only", action="store_true", help="Generate cURL commands instead of executing")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create database subcommand
    create_db_parser = subparsers.add_parser("create-db", help="Create a new database")
    create_db_parser.add_argument("--name", required=True, help="Name of the database")
    create_db_parser.add_argument("--description", default="", help="Description of the database")
    create_db_parser.add_argument("--dimension", type=int, default=128, help="Vector dimension (default: 128)")
    create_db_parser.add_argument("--index-type", default="HNSW", help="Index type (default: HNSW)")
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
    user_add_parser.add_argument("email", help="User email address")
    user_add_parser.add_argument("--role", required=True, help="User role (admin, developer, viewer, etc.)")
    user_add_parser.add_argument("--password", help="Optional password")
    user_add_parser.set_defaults(func=user_add)

    user_list_parser = subparsers.add_parser("user-list", help="List all users")
    user_list_parser.add_argument("--role", help="Filter by role")
    user_list_parser.add_argument("--status", help="Filter by status (active, inactive)")
    user_list_parser.set_defaults(func=user_list)

    user_show_parser = subparsers.add_parser("user-show", help="Show user details")
    user_show_parser.add_argument("email", help="User email address")
    user_show_parser.set_defaults(func=user_show)

    user_update_parser = subparsers.add_parser("user-update", help="Update user information")
    user_update_parser.add_argument("email", help="User email address")
    user_update_parser.add_argument("--role", help="New role")
    user_update_parser.add_argument("--status", help="New status (active, inactive)")
    user_update_parser.set_defaults(func=user_update)

    user_delete_parser = subparsers.add_parser("user-delete", help="Delete a user")
    user_delete_parser.add_argument("email", help="User email address")
    user_delete_parser.set_defaults(func=user_delete)

    user_activate_parser = subparsers.add_parser("user-activate", help="Activate a user")
    user_activate_parser.add_argument("email", help="User email address")
    user_activate_parser.set_defaults(func=user_activate)

    user_deactivate_parser = subparsers.add_parser("user-deactivate", help="Deactivate a user")
    user_deactivate_parser.add_argument("email", help="User email address")
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