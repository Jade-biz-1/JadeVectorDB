"""
Python client library for JadeVectorDB

This module provides a Python interface to interact with the JadeVectorDB
vector database system, allowing users to perform vector storage, retrieval,
and similarity search operations from Python applications.
"""

import requests
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class Vector:
    id: str
    values: List[float]
    metadata: Optional[Dict[str, Union[str, int, float, bool]]] = None

class JadeVectorDBError(Exception):
    """Custom exception for JadeVectorDB client errors"""
    pass

class JadeVectorDB:
    """
    Python client for JadeVectorDB
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the JadeVectorDB client
        
        :param base_url: Base URL for the JadeVectorDB API (e.g., http://localhost:8080)
        :param api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
        else:
            # If no API key provided, assume we're using API key format
            # This might be set by user later
            self.session.headers.update({
                'Content-Type': 'application/json'
            })
    
    def create_database(
        self, 
        name: str, 
        description: str = "", 
        vector_dimension: int = 128,
        index_type: str = "HNSW",
        **kwargs
    ) -> str:
        """
        Create a new vector database
        
        :param name: Name of the database
        :param description: Description of the database
        :param vector_dimension: Dimension of vectors to be stored
        :param index_type: Type of index to use (HNSW, IVF, LSH, FLAT)
        :return: Database ID
        """
        url = f"{self.base_url}/v1/databases"
        
        payload = {
            "name": name,
            "description": description,
            "vectorDimension": vector_dimension,
            "indexType": index_type
        }
        
        # Add any additional parameters from kwargs
        payload.update(kwargs)
        
        response = self.session.post(url, json=payload)
        
        if response.status_code == 201:
            result = response.json()
            return result.get('databaseId')
        else:
            raise JadeVectorDBError(f"Failed to create database: {response.text}")
    
    def get_database(self, database_id: str) -> Dict:
        """
        Get information about a specific database
        
        :param database_id: ID of the database to retrieve
        :return: Database information
        """
        url = f"{self.base_url}/v1/databases/{database_id}"
        
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get database: {response.text}")
    
    def list_databases(self) -> List[Dict]:
        """
        List all available databases
        
        :return: List of database information
        """
        url = f"{self.base_url}/v1/databases"
        
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to list databases: {response.text}")
    
    def store_vector(
        self, 
        database_id: str, 
        vector_id: str, 
        values: List[float], 
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store a vector in the database
        
        :param database_id: ID of the database to store in
        :param vector_id: Unique ID for the vector
        :param values: Vector values (list of floats)
        :param metadata: Optional metadata to store with the vector
        :return: True if successful
        """
        url = f"{self.base_url}/v1/databases/{database_id}/vectors"
        
        payload = {
            "id": vector_id,
            "values": values
        }
        
        if metadata:
            payload["metadata"] = metadata
            
        response = self.session.post(url, json=payload)
        
        if response.status_code == 201:
            return True
        else:
            raise JadeVectorDBError(f"Failed to store vector: {response.text}")
    
    def batch_store_vectors(
        self, 
        database_id: str, 
        vectors: List[Vector]
    ) -> bool:
        """
        Store multiple vectors in the database
        
        :param database_id: ID of the database to store in
        :param vectors: List of Vector objects to store
        :return: True if successful
        """
        url = f"{self.base_url}/v1/databases/{database_id}/vectors/batch"
        
        payload = {
            "vectors": [
                {
                    "id": v.id,
                    "values": v.values,
                    **({"metadata": v.metadata} if v.metadata else {})
                }
                for v in vectors
            ]
        }
        
        response = self.session.post(url, json=payload)
        
        if response.status_code == 201:
            return True
        else:
            raise JadeVectorDBError(f"Failed to batch store vectors: {response.text}")
    
    def retrieve_vector(
        self, 
        database_id: str, 
        vector_id: str
    ) -> Optional[Vector]:
        """
        Retrieve a vector from the database
        
        :param database_id: ID of the database to retrieve from
        :param vector_id: ID of the vector to retrieve
        :return: Vector object if found, None otherwise
        """
        url = f"{self.base_url}/v1/databases/{database_id}/vectors/{vector_id}"
        
        response = self.session.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return Vector(
                id=data['id'],
                values=data['values'],
                metadata=data.get('metadata')
            )
        elif response.status_code == 404:
            return None
        else:
            raise JadeVectorDBError(f"Failed to retrieve vector: {response.text}")
    
    def delete_vector(
        self, 
        database_id: str, 
        vector_id: str
    ) -> bool:
        """
        Delete a vector from the database
        
        :param database_id: ID of the database to delete from
        :param vector_id: ID of the vector to delete
        :return: True if successful
        """
        url = f"{self.base_url}/v1/databases/{database_id}/vectors/{vector_id}"
        
        response = self.session.delete(url)
        
        if response.status_code == 200:
            return True
        else:
            raise JadeVectorDBError(f"Failed to delete vector: {response.text}")
    
    def search(
        self, 
        database_id: str, 
        query_vector: List[float], 
        top_k: int = 10, 
        threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform similarity search
        
        :param database_id: ID of the database to search in
        :param query_vector: Query vector for similarity search
        :param top_k: Number of results to return
        :param threshold: Optional similarity threshold
        :param filters: Optional metadata filters
        :return: List of matching vectors with similarity scores
        """
        url = f"{self.base_url}/v1/databases/{database_id}/search"
        
        payload = {
            "queryVector": query_vector,
            "topK": top_k
        }
        
        if threshold is not None:
            payload["threshold"] = threshold
            
        if filters:
            payload["filters"] = filters
            
        response = self.session.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json().get('results', [])
        else:
            raise JadeVectorDBError(f"Failed to perform search: {response.text}")
    
    def get_status(self) -> Dict:
        """
        Get system status information
        
        :return: System status information
        """
        url = f"{self.base_url}/status"
        
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get status: {response.text}")
    
    def get_health(self) -> Dict:
        """
        Get system health information
        
        :return: System health information
        """
        url = f"{self.base_url}/health"
        
        response = self.session.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get health: {response.text}")