"""
Command-line interface for JadeVectorDB
"""

import argparse
import sys
import json
from typing import Dict, List
from .client import JadeVectorDB, Vector, JadeVectorDBError
from .curl_generator import CurlCommandGenerator

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

def setup_parser():
    """Set up the argument parser"""
    parser = argparse.ArgumentParser(description="JadeVectorDB CLI")
    parser.add_argument("--url", required=True, help="JadeVectorDB API URL (e.g., http://localhost:8080)")
    parser.add_argument("--api-key", help="API key for authentication")
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