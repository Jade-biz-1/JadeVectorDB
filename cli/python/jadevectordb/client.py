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

    @staticmethod
    def _sanitize_metadata(metadata: Optional[Dict]) -> Optional[Dict[str, str]]:
        """
        Convert metadata values to strings.

        The backend requires all metadata values to be strings. This helper
        transparently converts int, float, and bool values so callers can
        pass natural Python types (e.g. ``{"price": 9.99, "in_stock": True}``).
        """
        if metadata is None:
            return None
        return {k: str(v) if not isinstance(v, str) else v for k, v in metadata.items()}

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

    def delete_database(self, database_id: str) -> bool:
        """
        Delete a database

        :param database_id: ID of the database to delete
        :return: True if successful
        """
        url = f"{self.base_url}/v1/databases/{database_id}"

        response = self.session.delete(url)

        if response.status_code == 200:
            return True
        else:
            raise JadeVectorDBError(f"Failed to delete database: {response.text}")
    
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
            payload["metadata"] = self._sanitize_metadata(metadata)

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
                    **({"metadata": self._sanitize_metadata(v.metadata)} if v.metadata else {})
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

    # User Management Methods

    def create_user(
        self,
        username: str,
        password: str,
        roles: Optional[List[str]] = None,
        email: Optional[str] = None
    ) -> Dict:
        """
        Create a new user

        :param username: Username for the new user
        :param password: Password for the new user
        :param roles: Optional list of roles (defaults to ["user"])
        :param email: Optional email address
        :return: Created user information
        """
        url = f"{self.base_url}/v1/users"

        payload = {
            "username": username,
            "password": password
        }

        if roles:
            payload["roles"] = roles

        if email:
            payload["email"] = email

        response = self.session.post(url, json=payload)

        if response.status_code == 201:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to create user: {response.text}")

    def list_users(
        self,
        role: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        List all users

        :param role: Optional filter by role
        :param status: Optional filter by status (active, inactive)
        :return: List of users
        """
        url = f"{self.base_url}/v1/users"

        params = {}
        if role:
            params["role"] = role
        if status:
            params["status"] = status

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json().get('users', [])
        else:
            raise JadeVectorDBError(f"Failed to list users: {response.text}")

    def get_user(self, user_id: str) -> Dict:
        """
        Get information about a specific user

        :param user_id: User ID
        :return: User information
        """
        url = f"{self.base_url}/v1/users/{user_id}"

        response = self.session.get(url)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise JadeVectorDBError(f"User not found: {user_id}")
        else:
            raise JadeVectorDBError(f"Failed to get user: {response.text}")

    def update_user(
        self,
        user_id: str,
        is_active: Optional[bool] = None,
        roles: Optional[List[str]] = None
    ) -> Dict:
        """
        Update user information

        :param user_id: User ID
        :param is_active: New active status (optional)
        :param roles: New roles list (optional)
        :return: Updated user information
        """
        url = f"{self.base_url}/v1/users/{user_id}"

        payload = {}
        if is_active is not None:
            payload["is_active"] = is_active
        if roles:
            payload["roles"] = roles

        if not payload:
            raise JadeVectorDBError("At least one of is_active or roles must be provided")

        response = self.session.put(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to update user: {response.text}")

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user

        :param user_id: User ID
        :return: True if successful
        """
        url = f"{self.base_url}/v1/users/{user_id}"

        response = self.session.delete(url)

        if response.status_code == 200:
            return True
        else:
            raise JadeVectorDBError(f"Failed to delete user: {response.text}")

    def activate_user(self, user_id: str) -> Dict:
        """
        Activate a user

        :param user_id: User ID
        :return: Updated user information
        """
        return self.update_user(user_id, is_active=True)

    def deactivate_user(self, user_id: str) -> Dict:
        """
        Deactivate a user

        :param user_id: User ID
        :return: Updated user information
        """
        return self.update_user(user_id, is_active=False)

    # Hybrid Search Methods

    def hybrid_search(
        self,
        database_id: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        fusion_method: str = "rrf",
        alpha: float = 0.7,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and BM25 keyword search

        :param database_id: ID of the database to search in
        :param query_text: Query text for BM25 search (optional if query_vector provided)
        :param query_vector: Query vector for similarity search (optional if query_text provided)
        :param top_k: Number of results to return
        :param fusion_method: Fusion method ("rrf" or "linear")
        :param alpha: Weight for weighted linear fusion (0.0-1.0, default 0.7)
        :param filters: Optional metadata filters
        :return: List of hybrid search results with scores
        """
        url = f"{self.base_url}/v1/databases/{database_id}/search/hybrid"

        payload = {
            "topK": top_k,
            "fusionMethod": fusion_method,
            "alpha": alpha
        }

        if query_text:
            payload["queryText"] = query_text

        if query_vector:
            payload["queryVector"] = query_vector

        if filters:
            payload["filters"] = filters

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json().get('results', [])
        else:
            raise JadeVectorDBError(f"Failed to perform hybrid search: {response.text}")

    def build_bm25_index(
        self,
        database_id: str,
        text_field: str = "text",
        incremental: bool = False
    ) -> Dict:
        """
        Build BM25 index for a database

        :param database_id: ID of the database
        :param text_field: Metadata field to index (default: "text")
        :param incremental: Whether to perform incremental indexing
        :return: Build status information
        """
        url = f"{self.base_url}/v1/databases/{database_id}/bm25-index/build"

        payload = {
            "textField": text_field,
            "incremental": incremental
        }

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to build BM25 index: {response.text}")

    def get_bm25_index_status(self, database_id: str) -> Dict:
        """
        Get BM25 index build status

        :param database_id: ID of the database
        :return: Index status information
        """
        url = f"{self.base_url}/v1/databases/{database_id}/bm25-index/status"

        response = self.session.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get BM25 index status: {response.text}")

    def rebuild_bm25_index(
        self,
        database_id: str,
        text_field: str = "text"
    ) -> Dict:
        """
        Rebuild BM25 index from scratch

        :param database_id: ID of the database
        :param text_field: Metadata field to index (default: "text")
        :return: Rebuild status information
        """
        url = f"{self.base_url}/v1/databases/{database_id}/bm25-index/rebuild"

        payload = {
            "textField": text_field
        }

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to rebuild BM25 index: {response.text}")

    def add_bm25_documents(
        self,
        database_id: str,
        documents: List[Dict[str, str]]
    ) -> Dict:
        """
        Add documents to BM25 index

        :param database_id: ID of the database
        :param documents: List of documents with 'doc_id' and 'text' fields
        :return: Addition status information
        """
        url = f"{self.base_url}/v1/databases/{database_id}/bm25-index/documents"

        payload = {
            "documents": documents
        }

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to add BM25 documents: {response.text}")

    # Database & Vector Operations (additional)

    def update_database(
        self,
        database_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        vector_dimension: Optional[int] = None,
        index_type: Optional[str] = None
    ) -> Dict:
        """
        Update a database's configuration

        :param database_id: ID of the database to update
        :param name: New name (optional)
        :param description: New description (optional)
        :param vector_dimension: New vector dimension (optional)
        :param index_type: New index type (optional)
        :return: Updated database information
        """
        url = f"{self.base_url}/v1/databases/{database_id}"

        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if vector_dimension is not None:
            payload["vectorDimension"] = vector_dimension
        if index_type is not None:
            payload["indexType"] = index_type

        if not payload:
            raise JadeVectorDBError("At least one field must be provided for update")

        response = self.session.put(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to update database: {response.text}")

    def list_vectors(
        self,
        database_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict:
        """
        List vectors in a database with pagination

        :param database_id: ID of the database
        :param limit: Maximum number of vectors to return (default: 50)
        :param offset: Number of vectors to skip (default: 0)
        :return: Dictionary with vectors and pagination info
        """
        url = f"{self.base_url}/v1/databases/{database_id}/vectors"

        params = {"limit": limit, "offset": offset}

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to list vectors: {response.text}")

    def update_vector(
        self,
        database_id: str,
        vector_id: str,
        values: List[float],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update an existing vector's values and/or metadata

        :param database_id: ID of the database
        :param vector_id: ID of the vector to update
        :param values: New vector values
        :param metadata: New metadata (optional)
        :return: True if successful
        """
        url = f"{self.base_url}/v1/databases/{database_id}/vectors/{vector_id}"

        payload = {"values": values}
        if metadata is not None:
            payload["metadata"] = self._sanitize_metadata(metadata)

        response = self.session.put(url, json=payload)

        if response.status_code == 200:
            return True
        else:
            raise JadeVectorDBError(f"Failed to update vector: {response.text}")

    def batch_get_vectors(
        self,
        database_id: str,
        vector_ids: List[str]
    ) -> List[Dict]:
        """
        Retrieve multiple vectors by their IDs in a single request

        :param database_id: ID of the database
        :param vector_ids: List of vector IDs to retrieve
        :return: List of vector data dictionaries
        """
        url = f"{self.base_url}/v1/databases/{database_id}/vectors/batch-get"

        payload = {"ids": vector_ids}

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json().get('vectors', [])
        else:
            raise JadeVectorDBError(f"Failed to batch get vectors: {response.text}")

    # Search & Reranking

    def advanced_search(
        self,
        database_id: str,
        query_vector: List[float],
        top_k: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        include_values: bool = False
    ) -> List[Dict]:
        """
        Perform advanced similarity search with additional options

        :param database_id: ID of the database to search in
        :param query_vector: Query vector for similarity search
        :param top_k: Number of results to return
        :param threshold: Optional similarity threshold
        :param filters: Optional metadata filters
        :param include_metadata: Whether to include metadata in results (default: True)
        :param include_values: Whether to include vector values in results (default: False)
        :return: List of matching vectors with similarity scores
        """
        url = f"{self.base_url}/v1/databases/{database_id}/search/advanced"

        payload = {
            "queryVector": query_vector,
            "topK": top_k,
            "includeMetadata": include_metadata,
            "includeValues": include_values
        }

        if threshold is not None:
            payload["threshold"] = threshold
        if filters:
            payload["filters"] = filters

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json().get('results', [])
        else:
            raise JadeVectorDBError(f"Failed to perform advanced search: {response.text}")

    def rerank_search(
        self,
        database_id: str,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        enable_reranking: bool = True,
        rerank_top_n: int = 100
    ) -> Dict:
        """
        Perform search with reranking

        :param database_id: ID of the database to search in
        :param query_text: Query text for reranking
        :param query_vector: Optional query vector for initial retrieval
        :param top_k: Number of final results to return
        :param enable_reranking: Whether to enable reranking (default: True)
        :param rerank_top_n: Number of candidates to rerank (default: 100)
        :return: Search results with reranking scores
        """
        url = f"{self.base_url}/v1/databases/{database_id}/search/rerank"

        payload = {
            "queryText": query_text,
            "topK": top_k,
            "enableReranking": enable_reranking,
            "rerankTopN": rerank_top_n
        }

        if query_vector:
            payload["queryVector"] = query_vector

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to perform rerank search: {response.text}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Standalone document reranking

        :param query: Query text to rank documents against
        :param documents: List of documents with 'id' and 'text' fields
        :param top_k: Number of top results to return (optional)
        :return: Reranked list of documents with scores
        """
        url = f"{self.base_url}/v1/rerank"

        payload = {
            "query": query,
            "documents": documents
        }

        if top_k is not None:
            payload["topK"] = top_k

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json().get('results', [])
        else:
            raise JadeVectorDBError(f"Failed to rerank documents: {response.text}")

    def get_reranking_config(self, database_id: str) -> Dict:
        """
        Get reranking configuration for a database

        :param database_id: ID of the database
        :return: Reranking configuration
        """
        url = f"{self.base_url}/v1/databases/{database_id}/reranking/config"

        response = self.session.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get reranking config: {response.text}")

    def update_reranking_config(
        self,
        database_id: str,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        score_threshold: Optional[float] = None,
        combine_scores: Optional[bool] = None,
        rerank_weight: Optional[float] = None
    ) -> Dict:
        """
        Update reranking configuration for a database

        :param database_id: ID of the database
        :param model_name: Reranking model name (optional)
        :param batch_size: Batch size for reranking (optional)
        :param score_threshold: Minimum score threshold (optional)
        :param combine_scores: Whether to combine scores (optional)
        :param rerank_weight: Weight for reranking score (optional)
        :return: Updated reranking configuration
        """
        url = f"{self.base_url}/v1/databases/{database_id}/reranking/config"

        payload = {}
        if model_name is not None:
            payload["modelName"] = model_name
        if batch_size is not None:
            payload["batchSize"] = batch_size
        if score_threshold is not None:
            payload["scoreThreshold"] = score_threshold
        if combine_scores is not None:
            payload["combineScores"] = combine_scores
        if rerank_weight is not None:
            payload["rerankWeight"] = rerank_weight

        if not payload:
            raise JadeVectorDBError("At least one config field must be provided")

        response = self.session.put(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to update reranking config: {response.text}")

    # Index Management

    def create_index(
        self,
        database_id: str,
        index_type: str,
        name: Optional[str] = None,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """
        Create an index on a database

        :param database_id: ID of the database
        :param index_type: Type of index (e.g., HNSW, IVF, LSH, FLAT)
        :param name: Optional name for the index
        :param parameters: Optional index-specific parameters
        :return: Created index information
        """
        url = f"{self.base_url}/v1/databases/{database_id}/indexes"

        payload = {"type": index_type}
        if name:
            payload["name"] = name
        if parameters:
            payload["parameters"] = parameters

        response = self.session.post(url, json=payload)

        if response.status_code in (200, 201):
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to create index: {response.text}")

    def list_indexes(self, database_id: str) -> List[Dict]:
        """
        List all indexes for a database

        :param database_id: ID of the database
        :return: List of index information
        """
        url = f"{self.base_url}/v1/databases/{database_id}/indexes"

        response = self.session.get(url)

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            return data.get('indexes', [])
        else:
            raise JadeVectorDBError(f"Failed to list indexes: {response.text}")

    def update_index(
        self,
        database_id: str,
        index_id: str,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """
        Update an index's parameters

        :param database_id: ID of the database
        :param index_id: ID of the index to update
        :param parameters: New index parameters
        :return: Updated index information
        """
        url = f"{self.base_url}/v1/databases/{database_id}/indexes/{index_id}"

        payload = {}
        if parameters:
            payload["parameters"] = parameters

        response = self.session.put(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to update index: {response.text}")

    def delete_index(self, database_id: str, index_id: str) -> bool:
        """
        Delete an index from a database

        :param database_id: ID of the database
        :param index_id: ID of the index to delete
        :return: True if successful
        """
        url = f"{self.base_url}/v1/databases/{database_id}/indexes/{index_id}"

        response = self.session.delete(url)

        if response.status_code == 200:
            return True
        else:
            raise JadeVectorDBError(f"Failed to delete index: {response.text}")

    # Embeddings

    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        input_type: str = "text",
        model: str = "default",
        provider: str = "default"
    ) -> Dict:
        """
        Generate vector embeddings from text

        :param text: Input text or list of texts to embed
        :param input_type: Type of input (default: "text")
        :param model: Embedding model to use (default: "default")
        :param provider: Embedding provider (default: "default")
        :return: Generated embeddings
        """
        url = f"{self.base_url}/v1/embeddings/generate"

        payload = {
            "input": text,
            "input_type": input_type,
            "model": model,
            "provider": provider
        }

        response = self.session.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to generate embeddings: {response.text}")

    # API Key Management

    def create_api_key(
        self,
        user_id: str,
        description: str = "",
        permissions: Optional[List[str]] = None,
        validity_days: int = 0
    ) -> Dict:
        """
        Create a new API key

        :param user_id: User ID to associate the key with
        :param description: Description of the API key
        :param permissions: List of permissions (optional)
        :param validity_days: Number of days the key is valid (0 = no expiry)
        :return: Created API key information including the key value
        """
        url = f"{self.base_url}/v1/api-keys"

        payload = {
            "user_id": user_id,
            "description": description,
            "validity_days": validity_days
        }

        if permissions:
            payload["permissions"] = permissions

        response = self.session.post(url, json=payload)

        if response.status_code in (200, 201):
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to create API key: {response.text}")

    def list_api_keys(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        List API keys

        :param user_id: Optional user ID to filter by
        :return: List of API key information
        """
        url = f"{self.base_url}/v1/api-keys"

        params = {}
        if user_id:
            params["user_id"] = user_id

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data.get('api_keys', data.get('keys', []))
        else:
            raise JadeVectorDBError(f"Failed to list API keys: {response.text}")

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke (delete) an API key

        :param key_id: ID of the API key to revoke
        :return: True if successful
        """
        url = f"{self.base_url}/v1/api-keys/{key_id}"

        response = self.session.delete(url)

        if response.status_code == 200:
            return True
        else:
            raise JadeVectorDBError(f"Failed to revoke API key: {response.text}")

    # Security & Audit

    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> Dict:
        """
        Get audit log entries

        :param user_id: Optional user ID to filter by
        :param event_type: Optional event type to filter by
        :param limit: Maximum number of entries to return (default: 100)
        :return: Audit log entries
        """
        url = f"{self.base_url}/v1/security/audit-log"

        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        if event_type:
            params["event_type"] = event_type

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get audit log: {response.text}")

    def get_sessions(self, user_id: str) -> Dict:
        """
        Get active sessions for a user

        :param user_id: User ID
        :return: Session information
        """
        url = f"{self.base_url}/v1/security/sessions"

        params = {"user_id": user_id}

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get sessions: {response.text}")

    def get_audit_stats(self) -> Dict:
        """
        Get audit statistics summary

        :return: Audit statistics
        """
        url = f"{self.base_url}/v1/security/audit-stats"

        response = self.session.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get audit stats: {response.text}")

    # Analytics

    def get_analytics_stats(
        self,
        database_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        granularity: str = "hourly"
    ) -> Dict:
        """
        Get analytics statistics for a database

        :param database_id: ID of the database
        :param start_time: Start time filter (ISO 8601 format)
        :param end_time: End time filter (ISO 8601 format)
        :param granularity: Time granularity (hourly, daily, weekly)
        :return: Analytics statistics
        """
        url = f"{self.base_url}/v1/databases/{database_id}/analytics/stats"

        params = {"granularity": granularity}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get analytics stats: {response.text}")

    def get_analytics_queries(
        self,
        database_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict:
        """
        Get analytics query log for a database

        :param database_id: ID of the database
        :param start_time: Start time filter (ISO 8601 format)
        :param end_time: End time filter (ISO 8601 format)
        :param limit: Maximum number of entries (default: 100)
        :param offset: Number of entries to skip (default: 0)
        :return: Query analytics data
        """
        url = f"{self.base_url}/v1/databases/{database_id}/analytics/queries"

        params = {"limit": limit, "offset": offset}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get analytics queries: {response.text}")

    def get_analytics_patterns(
        self,
        database_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        min_count: int = 2,
        limit: int = 100
    ) -> Dict:
        """
        Get query pattern analytics for a database

        :param database_id: ID of the database
        :param start_time: Start time filter (ISO 8601 format)
        :param end_time: End time filter (ISO 8601 format)
        :param min_count: Minimum pattern occurrence count (default: 2)
        :param limit: Maximum number of patterns to return (default: 100)
        :return: Query pattern analytics
        """
        url = f"{self.base_url}/v1/databases/{database_id}/analytics/patterns"

        params = {"min_count": min_count, "limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get analytics patterns: {response.text}")

    def get_analytics_insights(
        self,
        database_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict:
        """
        Get analytics insights for a database

        :param database_id: ID of the database
        :param start_time: Start time filter (ISO 8601 format)
        :param end_time: End time filter (ISO 8601 format)
        :return: Analytics insights
        """
        url = f"{self.base_url}/v1/databases/{database_id}/analytics/insights"

        params = {}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get analytics insights: {response.text}")

    def get_analytics_trending(
        self,
        database_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        min_growth: float = 0.5,
        limit: int = 100
    ) -> Dict:
        """
        Get trending query analytics for a database

        :param database_id: ID of the database
        :param start_time: Start time filter (ISO 8601 format)
        :param end_time: End time filter (ISO 8601 format)
        :param min_growth: Minimum growth rate to include (default: 0.5)
        :param limit: Maximum number of trends to return (default: 100)
        :return: Trending analytics data
        """
        url = f"{self.base_url}/v1/databases/{database_id}/analytics/trending"

        params = {"min_growth": min_growth, "limit": limit}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to get analytics trending: {response.text}")

    def submit_analytics_feedback(
        self,
        database_id: str,
        query_id: str,
        user_id: Optional[str] = None,
        rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        clicked_result_id: Optional[str] = None,
        clicked_rank: Optional[int] = None
    ) -> Dict:
        """
        Submit feedback for a search query

        :param database_id: ID of the database
        :param query_id: ID of the query to provide feedback for
        :param user_id: Optional user ID
        :param rating: Optional rating (e.g., 1-5)
        :param feedback_text: Optional text feedback
        :param clicked_result_id: Optional ID of clicked result
        :param clicked_rank: Optional rank of clicked result
        :return: Feedback submission confirmation
        """
        url = f"{self.base_url}/v1/databases/{database_id}/analytics/feedback"

        payload = {"queryId": query_id}
        if user_id:
            payload["userId"] = user_id
        if rating is not None:
            payload["rating"] = rating
        if feedback_text:
            payload["feedbackText"] = feedback_text
        if clicked_result_id:
            payload["clickedResultId"] = clicked_result_id
        if clicked_rank is not None:
            payload["clickedRank"] = clicked_rank

        response = self.session.post(url, json=payload)

        if response.status_code in (200, 201):
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to submit analytics feedback: {response.text}")

    def export_analytics(
        self,
        database_id: str,
        format: str = "json",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict:
        """
        Export analytics data for a database

        :param database_id: ID of the database
        :param format: Export format (json, csv) (default: "json")
        :param start_time: Start time filter (ISO 8601 format)
        :param end_time: End time filter (ISO 8601 format)
        :return: Exported analytics data
        """
        url = f"{self.base_url}/v1/databases/{database_id}/analytics/export"

        params = {"format": format}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to export analytics: {response.text}")

    # Password Management

    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change a user's password

        :param user_id: User ID
        :param old_password: Current password
        :param new_password: New password
        :return: True if successful
        """
        url = f"{self.base_url}/v1/users/{user_id}/password"

        payload = {
            "old_password": old_password,
            "new_password": new_password
        }

        response = self.session.put(url, json=payload)

        if response.status_code == 200:
            return True
        else:
            raise JadeVectorDBError(f"Failed to change password: {response.text}")

    def admin_reset_password(
        self,
        user_id: str,
        new_password: str
    ) -> Dict:
        """
        Admin reset a user's password (requires admin privileges)

        :param user_id: User ID
        :param new_password: New password to set
        :return: Reset confirmation
        """
        url = f"{self.base_url}/v1/admin/users/{user_id}/reset-password"

        payload = {"new_password": new_password}

        response = self.session.put(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise JadeVectorDBError(f"Failed to reset password: {response.text}")