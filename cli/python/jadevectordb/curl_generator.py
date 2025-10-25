#!/usr/bin/env python3
"""
JadeVectorDB cURL Command Generator
Utility to generate cURL commands for JadeVectorDB operations
"""

import json
import sys
from typing import Dict, Any, Optional

class CurlCommandGenerator:
    """Generates cURL commands for JadeVectorDB operations"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
    
    def _build_auth_header(self) -> str:
        """Build authorization header for cURL command"""
        if self.api_key:
            return f"-H 'Authorization: Bearer {self.api_key}'"
        return ""
    
    def _format_json_data(self, data: Dict[Any, Any]) -> str:
        """Format JSON data for cURL command"""
        if not data:
            return ""
        json_str = json.dumps(data, indent=2)
        # Escape quotes for shell
        escaped_json = json_str.replace('"', '\\"')
        return f"-d '{json_str}'"
    
    def create_database(self, name: str, description: str = "", 
                        vector_dimension: int = 128, index_type: str = "HNSW") -> str:
        """Generate cURL command for creating a database"""
        data = {
            "name": name,
            "vectorDimension": vector_dimension,
            "indexType": index_type
        }
        if description:
            data["description"] = description
            
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        
        return f"""curl -X POST {self.base_url}/v1/databases \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""
    
    def list_databases(self) -> str:
        """Generate cURL command for listing databases"""
        auth_header = self._build_auth_header()
        return f"""curl -X GET {self.base_url}/v1/databases \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""
    
    def store_vector(self, database_id: str, vector_id: str, values: list, 
                    metadata: Optional[Dict] = None) -> str:
        """Generate cURL command for storing a vector"""
        data = {
            "id": vector_id,
            "values": values
        }
        if metadata:
            data["metadata"] = metadata
            
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        
        return f"""curl -X POST {self.base_url}/v1/databases/{database_id}/vectors \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""
    
    def retrieve_vector(self, database_id: str, vector_id: str) -> str:
        """Generate cURL command for retrieving a vector"""
        auth_header = self._build_auth_header()
        return f"""curl -X GET {self.base_url}/v1/databases/{database_id}/vectors/{vector_id} \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""
    
    def delete_vector(self, database_id: str, vector_id: str) -> str:
        """Generate cURL command for deleting a vector"""
        auth_header = self._build_auth_header()
        return f"""curl -X DELETE {self.base_url}/v1/databases/{database_id}/vectors/{vector_id} \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""
    
    def similarity_search(self, database_id: str, query_vector: list, 
                         top_k: int = 10, threshold: Optional[float] = None) -> str:
        """Generate cURL command for similarity search"""
        data = {
            "queryVector": query_vector,
            "topK": top_k
        }
        if threshold is not None:
            data["threshold"] = threshold
            
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        
        return f"""curl -X POST {self.base_url}/v1/databases/{database_id}/search \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""
    
    def get_status(self) -> str:
        """Generate cURL command for getting system status"""
        auth_header = self._build_auth_header()
        return f"""curl -X GET {self.base_url}/status \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""
    
    def get_health(self) -> str:
        """Generate cURL command for getting system health"""
        auth_header = self._build_auth_header()
        return f"""curl -X GET {self.base_url}/health \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

def main():
    """Main entry point for cURL command generator"""
    if len(sys.argv) < 2:
        print("Usage: python3 curl_generator.py <base_url> [api_key]")
        print("Example: python3 curl_generator.py http://localhost:8080 my-api-key")
        sys.exit(1)
    
    base_url = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    generator = CurlCommandGenerator(base_url, api_key)
    
    # Example usage
    print("# JadeVectorDB cURL Commands")
    print("\n## Create Database")
    print(generator.create_database("my-database", "My test database", 128, "HNSW"))
    
    print("\n## List Databases")
    print(generator.list_databases())
    
    print("\n## Store Vector")
    print(generator.store_vector("my-database", "vector-1", [0.1, 0.2, 0.3], {"category": "test"}))
    
    print("\n## Retrieve Vector")
    print(generator.retrieve_vector("my-database", "vector-1"))
    
    print("\n## Search")
    print(generator.similarity_search("my-database", [0.15, 0.25, 0.35], 5, 0.7))
    
    print("\n## System Status")
    print(generator.get_status())
    
    print("\n## System Health")
    print(generator.get_health())

if __name__ == "__main__":
    main()