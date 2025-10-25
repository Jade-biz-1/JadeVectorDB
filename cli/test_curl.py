#!/usr/bin/env python3
"""
Test script for JadeVectorDB cURL command generation
"""

import sys
import os

# Add the parent directory to the path so we can import the CLI module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from jadevectordb.curl_generator import CurlCommandGenerator

def test_curl_generation():
    """Test cURL command generation"""
    print("Testing JadeVectorDB cURL Command Generation")
    print("=" * 50)
    
    # Create a generator
    generator = CurlCommandGenerator("http://localhost:8080", "test-api-key")
    
    # Test create database
    print("\n1. Create Database:")
    curl_cmd = generator.create_database("test-db", "Test database", 128, "HNSW")
    print(curl_cmd)
    
    # Test list databases
    print("\n2. List Databases:")
    curl_cmd = generator.list_databases()
    print(curl_cmd)
    
    # Test store vector
    print("\n3. Store Vector:")
    curl_cmd = generator.store_vector("test-db", "vector-1", [0.1, 0.2, 0.3], {"category": "test"})
    print(curl_cmd)
    
    # Test retrieve vector
    print("\n4. Retrieve Vector:")
    curl_cmd = generator.retrieve_vector("test-db", "vector-1")
    print(curl_cmd)
    
    # Test similarity search
    print("\n5. Similarity Search:")
    curl_cmd = generator.similarity_search("test-db", [0.15, 0.25, 0.35], 5, 0.7)
    print(curl_cmd)
    
    # Test system status
    print("\n6. System Status:")
    curl_cmd = generator.get_status()
    print(curl_cmd)
    
    # Test system health
    print("\n7. System Health:")
    curl_cmd = generator.get_health()
    print(curl_cmd)
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")

if __name__ == "__main__":
    test_curl_generation()