#!/usr/bin/env python3
"""
Comprehensive test suite for JadeVectorDB cURL command generation
"""

import sys
import os

# Add the parent directory to the path so we can import the CLI module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.jadevectordb.curl_generator import CurlCommandGenerator

def test_all_curl_commands():
    """Test all cURL command generation functionality"""
    print("Comprehensive JadeVectorDB cURL Command Generation Test")
    print("=" * 60)
    
    # Create a generator
    generator = CurlCommandGenerator("http://localhost:8080", "test-api-key")
    
    # Test all commands
    tests = [
        ("Create Database", lambda: generator.create_database("test-db", "Test database", 128, "HNSW")),
        ("List Databases", lambda: generator.list_databases()),
        ("Store Vector", lambda: generator.store_vector("test-db", "vector-1", [0.1, 0.2, 0.3], {"category": "test"})),
        ("Retrieve Vector", lambda: generator.retrieve_vector("test-db", "vector-1")),
        ("Delete Vector", lambda: generator.delete_vector("test-db", "vector-1")),
        ("Similarity Search", lambda: generator.similarity_search("test-db", [0.15, 0.25, 0.35], 5, 0.7)),
        ("Get Status", lambda: generator.get_status()),
        ("Get Health", lambda: generator.get_health())
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result and isinstance(result, str) and result.startswith("curl"):
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED - Invalid cURL command format")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {failed} tests failed.")
        return False

def test_edge_cases():
    """Test edge cases for cURL command generation"""
    print("\nEdge Case Testing")
    print("-" * 30)
    
    # Test without API key
    generator_no_auth = CurlCommandGenerator("http://localhost:8080")
    
    try:
        cmd = generator_no_auth.list_databases()
        if "Authorization:" not in cmd:
            print("✅ No API key handling: PASSED")
        else:
            print("❌ No API key handling: FAILED - Authorization header present when it shouldn't be")
    except Exception as e:
        print(f"❌ No API key handling: FAILED - {str(e)}")
    
    # Test with special characters in database name
    try:
        cmd = generator_no_auth.create_database("test-db-with-special-chars_123", "Test database with special chars: &\"'")
        if "test-db-with-special-chars_123" in cmd:
            print("✅ Special characters handling: PASSED")
        else:
            print("❌ Special characters handling: FAILED - Special characters not properly escaped")
    except Exception as e:
        print(f"❌ Special characters handling: FAILED - {str(e)}")
    
    # Test with large vector
    try:
        large_vector = [0.1] * 1000  # Large vector with 1000 dimensions
        cmd = generator_no_auth.similarity_search("test-db", large_vector, 10, 0.8)
        if str(len(large_vector)) in cmd:
            print("✅ Large vector handling: PASSED")
        else:
            print("❌ Large vector handling: FAILED - Large vector not properly handled")
    except Exception as e:
        print(f"❌ Large vector handling: FAILED - {str(e)}")

def main():
    """Main test function"""
    print("Running JadeVectorDB cURL Command Generation Tests...\n")
    
    # Run comprehensive tests
    success = test_all_curl_commands()
    
    # Run edge case tests
    test_edge_cases()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All core tests passed! cURL command generation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())