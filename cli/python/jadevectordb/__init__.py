"""
JadeVectorDB Python Client Library

A Python client for interacting with the JadeVectorDB vector database system.
"""

from .client import JadeVectorDB, Vector, JadeVectorDBError

__version__ = "1.0.0"
__author__ = "JadeVectorDB Team"
__all__ = ["JadeVectorDB", "Vector", "JadeVectorDBError"]