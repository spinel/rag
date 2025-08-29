"""
Конфигурация приложения RAG-бота для QuantumForge Software.
"""
import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # Основные настройки
    app_name: str = "QuantumForge RAG Bot"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # OpenAI настройки
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 1000
    
    # ChromaDB настройки
    chroma_db_host: str = "localhost"
    chroma_db_port: int = 8000
    chroma_db_persist_directory: str = "./data/chroma"
    
    # Sentence Transformers настройки
    embedding_model: str = "all-mpnet-base-v2"
    embedding_device: str = "cpu"  # или "cuda" для GPU
    
    # Настройки обработки документов
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_document_size: int = 10 * 1024 * 1024  # 10MB
    
    # Настройки поиска
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    # Настройки API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Настройки безопасности
    cors_origins: list = ["*"]
    rate_limit_per_minute: int = 60
    
    # Настройки мониторинга
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = Settings()


def get_settings() -> Settings:
    """Получить настройки приложения."""
    return settings
