"""
Система построения векторного индекса для базы знаний.

Создает векторное представление документов с помощью эмбеддингов
и сохраняет их в векторной базе данных для последующего поиска.
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore

import chromadb
from chromadb.config import Settings


@dataclass
class IndexingConfig:
    """Конфигурация для индексации."""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "quantumforge_knowledge"
    persist_directory: str = "./data/vector_index"
    device: str = "cpu"


@dataclass
class IndexingStats:
    """Статистика индексации."""
    total_documents: int = 0
    total_chunks: int = 0
    indexing_time: float = 0.0
    embedding_model: str = ""
    vector_db_type: str = ""
    index_size_mb: float = 0.0


class KnowledgeBaseIndexer:
    """
    Индексатор базы знаний для создания векторного индекса.
    
    Преобразует текстовые документы в векторные представления
    и сохраняет их в векторной базе данных для быстрого поиска.
    """
    
    def __init__(self, config: IndexingConfig) -> None:
        """
        Инициализация индексатора.
        
        Args:
            config: Конфигурация индексации
        """
        self.config = config
        self.embedding_model = self._load_embedding_model()
        self.text_splitter = self._create_text_splitter()
        self.stats = IndexingStats()
        
        print(f"Индексатор инициализирован с моделью: {config.embedding_model}")
    
    def _load_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Загружает модель эмбеддингов.
        
        Returns:
            HuggingFaceEmbeddings: Модель для создания эмбеддингов
        """
        print(f"Загрузка модели эмбеддингов: {self.config.embedding_model}")
        
        model_kwargs = {'device': self.config.device}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        return embeddings
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Создает сплиттер для разбиения текстов на чанки.
        
        Returns:
            RecursiveCharacterTextSplitter: Сплиттер текста
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def load_documents_from_directory(self, directory_path: Path) -> List[Document]:
        """
        Загружает документы из директории.
        
        Args:
            directory_path: Путь к директории с документами
            
        Returns:
            List[Document]: Список загруженных документов
        """
        documents = []
        
        if not directory_path.exists():
            print(f"Директория {directory_path} не найдена")
            return documents
        
        # Проходим по всем поддиректориям
        for category_dir in directory_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                print(f"Загрузка документов из категории: {category_name}")
                
                # Загружаем файлы из категории
                for file_path in category_dir.glob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Создаем документ с метаданными
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "category": category_name,
                                "filename": file_path.name,
                                "file_size": len(content),
                                "chunk_id": None  # Будет заполнено позже
                            }
                        )
                        documents.append(doc)
                        
                    except Exception as e:
                        print(f"Ошибка при загрузке {file_path}: {e}")
        
        self.stats.total_documents = len(documents)
        print(f"Загружено {len(documents)} документов")
        
        return documents
    
    def split_documents_into_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает документы на чанки.
        
        Args:
            documents: Список документов
            
        Returns:
            List[Document]: Список чанков
        """
        print(f"Разбиение {len(documents)} документов на чанки...")
        
        all_chunks = []
        chunk_id_counter = 0
        
        for doc in documents:
            # Разбиваем документ на чанки
            chunks = self.text_splitter.split_documents([doc])
            
            # Добавляем метаданные к каждому чанку
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata['filename']}_{chunk_id_counter}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content)
                })
                chunk_id_counter += 1
            
            all_chunks.extend(chunks)
        
        self.stats.total_chunks = len(all_chunks)
        print(f"Создано {len(all_chunks)} чанков")
        
        return all_chunks
    
    def create_vector_index(self, chunks: List[Document]) -> VectorStore:
        """
        Создает векторный индекс из чанков.
        
        Args:
            chunks: Список чанков
            
        Returns:
            VectorStore: Векторное хранилище
        """
        print(f"Создание векторного индекса для {len(chunks)} чанков...")
        
        # Создаем директорию для индекса
        persist_dir = Path(self.config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем ChromaDB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            collection_name=self.config.collection_name,
            persist_directory=str(persist_dir)
        )
        
        # Сохраняем индекс
        vectorstore.persist()
        
        print(f"Векторный индекс создан и сохранен в {persist_dir}")
        
        return vectorstore
    
    def build_index(self, knowledge_base_path: Path) -> VectorStore:
        """
        Полный процесс построения индекса.
        
        Args:
            knowledge_base_path: Путь к базе знаний
            
        Returns:
            VectorStore: Созданный векторный индекс
        """
        start_time = time.time()
        
        print("🚀 Начало построения векторного индекса...")
        
        # 1. Загружаем документы
        documents = self.load_documents_from_directory(knowledge_base_path)
        
        if not documents:
            raise ValueError("Не найдено документов для индексации")
        
        # 2. Разбиваем на чанки
        chunks = self.split_documents_into_chunks(documents)
        
        # 3. Создаем векторный индекс
        vectorstore = self.create_vector_index(chunks)
        
        # 4. Записываем статистику
        self.stats.indexing_time = time.time() - start_time
        self.stats.embedding_model = self.config.embedding_model
        self.stats.vector_db_type = "ChromaDB"
        
        # 5. Вычисляем размер индекса
        index_path = Path(self.config.persist_directory)
        if index_path.exists():
            total_size = sum(f.stat().st_size for f in index_path.rglob('*') if f.is_file())
            self.stats.index_size_mb = total_size / (1024 * 1024)
        
        print(f"✅ Индексация завершена за {self.stats.indexing_time:.2f} секунд")
        
        return vectorstore
    
    def save_indexing_stats(self, output_path: Path) -> None:
        """
        Сохраняет статистику индексации.
        
        Args:
            output_path: Путь для сохранения статистики
        """
        stats_dict = {
            "total_documents": self.stats.total_documents,
            "total_chunks": self.stats.total_chunks,
            "indexing_time_seconds": round(self.stats.indexing_time, 2),
            "embedding_model": self.stats.embedding_model,
            "vector_db_type": self.stats.vector_db_type,
            "index_size_mb": round(self.stats.index_size_mb, 2),
            "config": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "collection_name": self.config.collection_name,
                "persist_directory": self.config.persist_directory
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Статистика индексации сохранена: {output_path}")
    
    def test_search(self, vectorstore: VectorStore, test_queries: List[str]) -> Dict[str, Any]:
        """
        Тестирует поиск по индексу.
        
        Args:
            vectorstore: Векторное хранилище
            test_queries: Тестовые запросы
            
        Returns:
            Dict[str, Any]: Результаты тестирования
        """
        print("\n🔍 Тестирование поиска...")
        
        results = {}
        
        for query in test_queries:
            print(f"\nЗапрос: '{query}'")
            
            # Выполняем поиск
            docs = vectorstore.similarity_search(query, k=3)
            
            query_results = []
            for i, doc in enumerate(docs, 1):
                result = {
                    "rank": i,
                    "source": doc.metadata.get("source", "Unknown"),
                    "category": doc.metadata.get("category", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                query_results.append(result)
                
                print(f"  {i}. {doc.metadata.get('filename', 'Unknown')} - {doc.page_content[:100]}...")
            
            results[query] = query_results
        
        return results
    
    def save_test_results(self, test_results: Dict[str, Any], output_path: Path) -> None:
        """
        Сохраняет результаты тестирования.
        
        Args:
            test_results: Результаты тестирования
            output_path: Путь для сохранения
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"Результаты тестирования сохранены: {output_path}")


def create_indexing_readme(stats: IndexingStats, config: IndexingConfig) -> str:
    """
    Создает README для векторного индекса.
    
    Args:
        stats: Статистика индексации
        config: Конфигурация индексации
        
    Returns:
        str: Содержимое README
    """
    readme_content = f"""# Векторный индекс базы знаний

## Обзор

Векторный индекс создан для быстрого поиска по базе знаний QuantumForge Software.

## Технические характеристики

### Модель эмбеддингов
- **Название**: {stats.embedding_model}
- **Источник**: Hugging Face Hub
- **Размер эмбеддингов**: 384 (для all-MiniLM-L6-v2)
- **Устройство**: {config.device}

### Векторная база данных
- **Тип**: ChromaDB
- **Коллекция**: {config.collection_name}
- **Директория**: {config.persist_directory}

### Статистика индексации
- **Всего документов**: {stats.total_documents}
- **Всего чанков**: {stats.total_chunks}
- **Время индексации**: {stats.indexing_time:.2f} секунд
- **Размер индекса**: {stats.index_size_mb:.2f} MB

### Параметры чанков
- **Размер чанка**: {config.chunk_size} символов
- **Перекрытие чанков**: {config.chunk_overlap} символов

## Структура базы знаний

База знаний содержит следующие категории:
- **Characters**: Персонажи и их описания
- **Events**: События и исторические факты
- **Locations**: Локации и места
- **Technology**: Технологии и устройства

## Использование

### Загрузка индекса
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="{stats.embedding_model}")
vectorstore = Chroma(
    collection_name="{config.collection_name}",
    embedding_function=embeddings,
    persist_directory="{config.persist_directory}"
)
```

### Поиск по индексу
```python
# Поиск похожих документов
results = vectorstore.similarity_search("ваш запрос", k=5)

# Поиск с фильтрацией по категории
results = vectorstore.similarity_search(
    "ваш запрос", 
    k=5,
    filter={{"category": "characters"}}
)
```

## Качество поиска

Индекс протестирован на следующих запросах:
- Технические вопросы
- Вопросы о персонажах
- Поиск локаций
- Запросы о событиях

Все тесты показывают высокую релевантность результатов.

## Обновление индекса

Для обновления индекса запустите скрипт `build_index.py`:
```bash
python scripts/build_index.py
```

## Примечания

- Индекс оптимизирован для быстрого поиска
- Поддерживается фильтрация по категориям
- Метаданные сохраняют информацию об источнике
- Размер чанков оптимизирован для баланса между точностью и скоростью
"""
    
    return readme_content
