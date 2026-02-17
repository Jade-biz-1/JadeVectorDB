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

    def get_database(self, database_id: str) -> str:
        """Generate cURL command for getting a database"""
        auth_header = self._build_auth_header()
        return f"""curl -X GET {self.base_url}/v1/databases/{database_id} \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

    def delete_database(self, database_id: str) -> str:
        """Generate cURL command for deleting a database"""
        auth_header = self._build_auth_header()
        return f"""curl -X DELETE {self.base_url}/v1/databases/{database_id} \\
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

    def update_database(self, database_id: str, name: Optional[str] = None,
                        description: Optional[str] = None,
                        vector_dimension: Optional[int] = None,
                        index_type: Optional[str] = None) -> str:
        """Generate cURL command for updating a database"""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if vector_dimension is not None:
            data["vectorDimension"] = vector_dimension
        if index_type is not None:
            data["indexType"] = index_type
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X PUT {self.base_url}/v1/databases/{database_id} \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def list_vectors(self, database_id: str, limit: int = 50, offset: int = 0) -> str:
        """Generate cURL command for listing vectors"""
        auth_header = self._build_auth_header()
        return f"""curl -X GET '{self.base_url}/v1/databases/{database_id}/vectors?limit={limit}&offset={offset}' \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

    def update_vector(self, database_id: str, vector_id: str, values: list,
                      metadata: Optional[Dict] = None) -> str:
        """Generate cURL command for updating a vector"""
        data = {"values": values}
        if metadata:
            data["metadata"] = metadata
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X PUT {self.base_url}/v1/databases/{database_id}/vectors/{vector_id} \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def batch_get_vectors(self, database_id: str, vector_ids: list) -> str:
        """Generate cURL command for batch getting vectors"""
        data = {"ids": vector_ids}
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X POST {self.base_url}/v1/databases/{database_id}/vectors/batch-get \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def advanced_search(self, database_id: str, query_vector: list, top_k: int = 10,
                        threshold: Optional[float] = None, filters: Optional[Dict] = None,
                        include_metadata: bool = True, include_values: bool = False) -> str:
        """Generate cURL command for advanced search"""
        data = {
            "queryVector": query_vector,
            "topK": top_k,
            "includeMetadata": include_metadata,
            "includeValues": include_values
        }
        if threshold is not None:
            data["threshold"] = threshold
        if filters:
            data["filters"] = filters
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X POST {self.base_url}/v1/databases/{database_id}/search/advanced \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def rerank_search(self, database_id: str, query_text: str,
                      query_vector: Optional[list] = None, top_k: int = 10,
                      enable_reranking: bool = True, rerank_top_n: int = 100) -> str:
        """Generate cURL command for rerank search"""
        data = {
            "queryText": query_text,
            "topK": top_k,
            "enableReranking": enable_reranking,
            "rerankTopN": rerank_top_n
        }
        if query_vector:
            data["queryVector"] = query_vector
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X POST {self.base_url}/v1/databases/{database_id}/search/rerank \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def rerank(self, query: str, documents: list, top_k: Optional[int] = None) -> str:
        """Generate cURL command for standalone reranking"""
        data = {"query": query, "documents": documents}
        if top_k is not None:
            data["topK"] = top_k
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X POST {self.base_url}/v1/rerank \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def create_index(self, database_id: str, index_type: str,
                     name: Optional[str] = None, parameters: Optional[Dict] = None) -> str:
        """Generate cURL command for creating an index"""
        data = {"indexType": index_type}
        if name:
            data["name"] = name
        if parameters:
            data["parameters"] = parameters
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X POST {self.base_url}/v1/databases/{database_id}/indexes \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def list_indexes(self, database_id: str) -> str:
        """Generate cURL command for listing indexes"""
        auth_header = self._build_auth_header()
        return f"""curl -X GET {self.base_url}/v1/databases/{database_id}/indexes \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

    def delete_index(self, database_id: str, index_id: str) -> str:
        """Generate cURL command for deleting an index"""
        auth_header = self._build_auth_header()
        return f"""curl -X DELETE {self.base_url}/v1/databases/{database_id}/indexes/{index_id} \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

    def generate_embeddings(self, text: str, input_type: str = "text",
                            model: str = "default", provider: str = "default") -> str:
        """Generate cURL command for embedding generation"""
        data = {
            "input": text,
            "input_type": input_type,
            "model": model,
            "provider": provider
        }
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X POST {self.base_url}/v1/embeddings/generate \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def create_api_key(self, user_id: str, description: str = "",
                       permissions: Optional[list] = None, validity_days: int = 0) -> str:
        """Generate cURL command for creating an API key"""
        data = {
            "userId": user_id,
            "description": description,
            "validityDays": validity_days
        }
        if permissions:
            data["permissions"] = permissions
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X POST {self.base_url}/v1/api-keys \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def list_api_keys(self, user_id: Optional[str] = None) -> str:
        """Generate cURL command for listing API keys"""
        auth_header = self._build_auth_header()
        query = f"?user_id={user_id}" if user_id else ""
        return f"""curl -X GET '{self.base_url}/v1/api-keys{query}' \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

    def revoke_api_key(self, key_id: str) -> str:
        """Generate cURL command for revoking an API key"""
        auth_header = self._build_auth_header()
        return f"""curl -X DELETE {self.base_url}/v1/api-keys/{key_id} \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

    def change_password(self, user_id: str, old_password: str, new_password: str) -> str:
        """Generate cURL command for changing password"""
        data = {"oldPassword": old_password, "newPassword": new_password}
        data_flag = self._format_json_data(data)
        auth_header = self._build_auth_header()
        return f"""curl -X PUT {self.base_url}/v1/users/{user_id}/password \\
  -H 'Content-Type: application/json' \\
  {auth_header} \\
  {data_flag}"""

    def get_audit_log(self, user_id: Optional[str] = None,
                      event_type: Optional[str] = None, limit: int = 100) -> str:
        """Generate cURL command for getting audit log"""
        auth_header = self._build_auth_header()
        params = [f"limit={limit}"]
        if user_id:
            params.append(f"user_id={user_id}")
        if event_type:
            params.append(f"event_type={event_type}")
        query = "?" + "&".join(params)
        return f"""curl -X GET '{self.base_url}/v1/security/audit-log{query}' \\
  -H 'Content-Type: application/json' \\
  {auth_header}"""

    def get_analytics_stats(self, database_id: str, granularity: str = "hourly",
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None) -> str:
        """Generate cURL command for getting analytics stats"""
        auth_header = self._build_auth_header()
        params = [f"granularity={granularity}"]
        if start_time:
            params.append(f"start_time={start_time}")
        if end_time:
            params.append(f"end_time={end_time}")
        query = "?" + "&".join(params)
        return f"""curl -X GET '{self.base_url}/v1/databases/{database_id}/analytics/stats{query}' \\
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