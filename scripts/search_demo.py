#!/usr/bin/env python3
"""
Демонстрационный скрипт для поиска по векторному индексу.

Показывает, как использовать созданный векторный индекс
для поиска релевантных документов.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_vector_index(index_path: str = "./data/vector_index") -> Chroma:
    """
    Загружает векторный индекс.
    
    Args:
        index_path: Путь к индексу
        
    Returns:
        Chroma: Загруженный векторный индекс
    """
    print(f"Загрузка векторного индекса из {index_path}...")
    
    # Загружаем модель эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Загружаем векторное хранилище
    vectorstore = Chroma(
        collection_name="quantumforge_knowledge",
        embedding_function=embeddings,
        persist_directory=index_path
    )
    
    print("✅ Векторный индекс загружен успешно!")
    return vectorstore


def search_documents(vectorstore: Chroma, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Выполняет поиск по документам.
    
    Args:
        vectorstore: Векторное хранилище
        query: Поисковый запрос
        k: Количество результатов
        
    Returns:
        List[Dict[str, Any]]: Результаты поиска
    """
    print(f"\n🔍 Поиск: '{query}'")
    print("-" * 60)
    
    # Выполняем поиск
    docs = vectorstore.similarity_search(query, k=k)
    
    results = []
    for i, doc in enumerate(docs, 1):
        result = {
            "rank": i,
            "source": doc.metadata.get("source", "Unknown"),
            "category": doc.metadata.get("category", "Unknown"),
            "filename": doc.metadata.get("filename", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        }
        results.append(result)
        
        # Выводим результат
        print(f"{i}. 📄 {result['filename']} ({result['category']})")
        print(f"   📍 {result['chunk_id']}")
        print(f"   📝 {result['content']}")
        print()
    
    return results


def search_by_category(vectorstore: Chroma, query: str, category: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Выполняет поиск с фильтрацией по категории.
    
    Args:
        vectorstore: Векторное хранилище
        query: Поисковый запрос
        category: Категория для фильтрации
        k: Количество результатов
        
    Returns:
        List[Dict[str, Any]]: Результаты поиска
    """
    print(f"\n🔍 Поиск в категории '{category}': '{query}'")
    print("-" * 60)
    
    # Выполняем поиск с фильтрацией
    docs = vectorstore.similarity_search(
        query, 
        k=k,
        filter={"category": category}
    )
    
    results = []
    for i, doc in enumerate(docs, 1):
        result = {
            "rank": i,
            "source": doc.metadata.get("source", "Unknown"),
            "category": doc.metadata.get("category", "Unknown"),
            "filename": doc.metadata.get("filename", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        }
        results.append(result)
        
        # Выводим результат
        print(f"{i}. 📄 {result['filename']} ({result['category']})")
        print(f"   📍 {result['chunk_id']}")
        print(f"   📝 {result['content']}")
        print()
    
    return results


def interactive_search(vectorstore: Chroma) -> None:
    """
    Интерактивный режим поиска.
    
    Args:
        vectorstore: Векторное хранилище
    """
    print("\n🎯 Интерактивный режим поиска")
    print("Введите 'quit' для выхода")
    print("Введите 'category <категория>' для поиска в конкретной категории")
    print("Доступные категории: characters, events, locations, technology")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n🔍 Введите запрос: ").strip()
            
            if query.lower() == 'quit':
                print("👋 До свидания!")
                break
            
            if query.lower().startswith('category '):
                # Поиск с фильтрацией по категории
                parts = query.split(' ', 2)
                if len(parts) >= 3:
                    category = parts[1]
                    search_query = parts[2]
                    search_by_category(vectorstore, search_query, category)
                else:
                    print("❌ Неверный формат. Используйте: category <категория> <запрос>")
            else:
                # Обычный поиск
                search_documents(vectorstore, query)
                
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def main() -> None:
    """Основная функция."""
    print("🚀 Демонстрация поиска по векторному индексу")
    
    # Проверяем существование индекса
    index_path = Path("./data/vector_index")
    if not index_path.exists():
        print("❌ Векторный индекс не найден. Сначала запустите build_index.py")
        return
    
    try:
        # Загружаем индекс
        vectorstore = load_vector_index()
        
        # Демонстрационные запросы
        demo_queries = [
            "Кто такой Kael Vexar?",
            "Что такое Plasma Blade?",
            "Расскажи о Synthetic Wars",
            "Как работают Tech Automatons?",
            "Что такое Synth Flux?"
        ]
        
        print(f"\n📋 Демонстрационные запросы:")
        for i, query in enumerate(demo_queries, 1):
            print(f"  {i}. {query}")
        
        # Выполняем демонстрационные поиски
        for query in demo_queries:
            search_documents(vectorstore, query, k=3)
        
        # Демонстрация поиска по категориям
        print("\n" + "="*60)
        print("ПОИСК ПО КАТЕГОРИЯМ")
        print("="*60)
        
        category_searches = [
            ("characters", "Kael Vexar"),
            ("technology", "Plasma Blade"),
            ("events", "Synthetic Wars"),
            ("locations", "Desert World")
        ]
        
        for category, query in category_searches:
            search_by_category(vectorstore, query, category, k=2)
        
        # Интерактивный режим
        print("\n" + "="*60)
        print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("="*60)
        
        interactive_search(vectorstore)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
