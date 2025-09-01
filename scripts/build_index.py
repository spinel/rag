#!/usr/bin/env python3
"""
Скрипт для построения векторного индекса базы знаний.

Создает векторное представление документов и сохраняет их
в ChromaDB для последующего поиска.
"""
import sys
from pathlib import Path
from typing import List

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.vectorstore.index_builder import (
    KnowledgeBaseIndexer, 
    IndexingConfig, 
    create_indexing_readme
)


def main() -> None:
    """Основная функция построения индекса."""
    print("🚀 Запуск построения векторного индекса...")
    
    # Конфигурация индексации
    config = IndexingConfig(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        collection_name="quantumforge_knowledge",
        persist_directory="./data/vector_index",
        device="cpu"
    )
    
    print(f"\n📋 Конфигурация:")
    print(f"  Модель эмбеддингов: {config.embedding_model}")
    print(f"  Размер чанка: {config.chunk_size}")
    print(f"  Перекрытие чанков: {config.chunk_overlap}")
    print(f"  Коллекция: {config.collection_name}")
    print(f"  Директория: {config.persist_directory}")
    
    # Инициализируем индексатор
    indexer = KnowledgeBaseIndexer(config)
    
    # Путь к базе знаний
    knowledge_base_path = Path("knowledge_base")
    
    if not knowledge_base_path.exists():
        print(f"❌ База знаний не найдена: {knowledge_base_path}")
        return
    
    try:
        # Строим индекс
        vectorstore = indexer.build_index(knowledge_base_path)
        
        # Сохраняем статистику
        stats_path = Path("data/indexing_stats.json")
        indexer.save_indexing_stats(stats_path)
        
        # Тестовые запросы для проверки качества
        test_queries = [
            "Кто такой Kael Vexar?",
            "Что такое Plasma Blade?",
            "Расскажи о Synthetic Wars",
            "Какие технологии используются?",
            "Где происходили основные события?",
            "Как работают Tech Automatons?",
            "Что такое Synth Flux?",
            "Какие локации упоминаются в документах?"
        ]
        
        # Тестируем поиск
        test_results = indexer.test_search(vectorstore, test_queries)
        
        # Сохраняем результаты тестирования
        test_results_path = Path("data/search_test_results.json")
        indexer.save_test_results(test_results, test_results_path)
        
        # Создаем README
        readme_content = create_indexing_readme(indexer.stats, config)
        readme_path = Path("data/vector_index/README.md")
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"\n📝 README создан: {readme_path}")
        
        # Выводим итоговую статистику
        print(f"\n" + "="*60)
        print("ИТОГОВАЯ СТАТИСТИКА ИНДЕКСАЦИИ")
        print("="*60)
        print(f"📊 Документов обработано: {indexer.stats.total_documents}")
        print(f"📄 Чанков создано: {indexer.stats.total_chunks}")
        print(f"⏱️  Время индексации: {indexer.stats.indexing_time:.2f} секунд")
        print(f"💾 Размер индекса: {indexer.stats.index_size_mb:.2f} MB")
        print(f"🔍 Тестовых запросов: {len(test_queries)}")
        
        print(f"\n✅ Векторный индекс успешно создан!")
        print(f"📁 Расположение: {config.persist_directory}")
        print(f"🔍 Готов к использованию для поиска!")
        
    except Exception as e:
        print(f"❌ Ошибка при построении индекса: {e}")
        return


if __name__ == "__main__":
    main()
