"""
Configuration management for EnterpriseRAG
Supports both mock and production modes
"""

import os
from typing import Literal
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Operation Mode
    mode: Literal["mock", "production"] = "mock"

    # Application
    app_name: str = "EnterpriseRAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # EnterpriseRAG API key (set to enable auth; leave empty to disable in dev)
    rag_api_key: str | None = None

    # Frontend URL (for CORS)
    frontend_url: str = "http://localhost:5173"

    # JadeVectorDB (Production Mode)
    jadevectordb_url: str = "http://localhost:8080"
    jadevectordb_api_key: str | None = None
    jadevectordb_database_id: str = "maintenance_docs"

    # Ollama (Production Mode)
    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_llm_model: str = "llama3.2:3b"

    # RAG Parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.65

    # Embeddings
    embedding_dimension: int = 384  # E5-small / nomic-embed-text

    # LLM Parameters
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Document Upload
    max_upload_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".docx"]
    upload_dir: Path = Path("uploads")

    # Persistent metadata database
    metadata_db_path: Path = Path("data/rag_metadata.db")

    # Processing
    batch_size: int = 100
    max_concurrent_uploads: int = 3

    # Timeouts (seconds)
    embedding_timeout: int = 30
    llm_timeout: int = 60
    search_timeout: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def is_mock_mode() -> bool:
    """Check if running in mock mode"""
    return settings.mode == "mock"


def is_production_mode() -> bool:
    """Check if running in production mode"""
    return settings.mode == "production"


# Create upload directory if it doesn't exist
settings.upload_dir.mkdir(exist_ok=True)


# Print configuration on startup
print("\n" + "="*60)
print(f"🚀 {settings.app_name} v{settings.app_version}")
print("="*60)
print(f"Mode: {settings.mode.upper()}")
print(f"Frontend URL: {settings.frontend_url}")

if is_production_mode():
    print(f"JadeVectorDB: {settings.jadevectordb_url}")
    print(f"Ollama: {settings.ollama_url}")
    print(f"Database ID: {settings.jadevectordb_database_id}")
else:
    print("⚠️  Running in MOCK mode - using simulated responses")

print(f"Upload directory: {settings.upload_dir.absolute()}")
print("="*60 + "\n")
