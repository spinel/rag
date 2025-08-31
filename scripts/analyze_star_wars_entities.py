#!/usr/bin/env python3
"""
Скрипт для анализа сущностей в файлах Star Wars.

Анализирует все файлы персонажей и извлекает:
- Персонажей
- Технологии
- События
- Объекты
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.entity_extraction.star_wars_entities import StarWarsEntityExtractor, EntityType


def load_star_wars_files(data_dir: str = "starwars/characters") -> Dict[str, str]:
    """
    Загружает все файлы Star Wars.
    
    Args:
        data_dir: Директория с файлами
        
    Returns:
        Dict[str, str]: Словарь {имя_файла: содержимое}
    """
    files = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Директория {data_dir} не найдена")
        return files
    
    for file_path in data_path.glob("*.txt"):
        if file_path.stat().st_size > 0:  # Пропускаем пустые файлы
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    files[file_path.stem] = content
                print(f"Загружен файл: {file_path.name}")
            except Exception as e:
                print(f"Ошибка при загрузке {file_path.name}: {e}")
    
    return files


def analyze_entities(files: Dict[str, str]) -> Dict[str, Any]:
    """
    Анализирует сущности во всех файлах.
    
    Args:
        files: Словарь файлов
        
    Returns:
        Dict[str, Any]: Результаты анализа
    """
    extractor = StarWarsEntityExtractor()
    all_entities = []
    file_entities = {}
    
    print(f"\nАнализируем {len(files)} файлов...")
    
    for filename, content in files.items():
        print(f"Обрабатываем {filename}...")
        
        # Извлекаем сущности из файла
        entities = extractor.extract_entities_from_text(content)
        file_entities[filename] = entities
        all_entities.extend(entities)
    
    # Получаем статистику
    stats = extractor.get_entity_statistics(all_entities)
    
    return {
        "statistics": stats,
        "file_entities": file_entities,
        "all_entities": all_entities
    }


def print_analysis_results(results: Dict[str, Any]) -> None:
    """
    Выводит результаты анализа в консоль.
    
    Args:
        results: Результаты анализа
    """
    stats = results["statistics"]
    
    print("\n" + "="*60)
    print("АНАЛИЗ СУЩНОСТЕЙ STAR WARS")
    print("="*60)
    
    # Общая статистика
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"Всего сущностей: {stats['total_entities']}")
    
    # Статистика по типам
    print(f"\n📈 СУЩНОСТИ ПО ТИПАМ:")
    for entity_type, count in stats["by_type"].items():
        if count > 0:
            print(f"  {entity_type.capitalize()}: {count}")
    
    # Топ сущностей
    print(f"\n🏆 ТОП-10 САМЫХ ЧАСТЫХ СУЩНОСТЕЙ:")
    for i, (entity_name, count) in enumerate(stats["top_entities"], 1):
        print(f"  {i:2d}. {entity_name}: {count}")
    
    # Детали по файлам
    print(f"\n📁 ДЕТАЛИ ПО ФАЙЛАМ:")
    for filename, entities in results["file_entities"].items():
        if entities:
            entity_types = {}
            for entity in entities:
                entity_type = entity.entity_type.value
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            print(f"\n  {filename}:")
            for entity_type, count in entity_types.items():
                print(f"    {entity_type.capitalize()}: {count}")


def save_results_to_json(results: Dict[str, Any], output_file: str = "star_wars_entities_analysis.json") -> None:
    """
    Сохраняет результаты анализа в JSON файл.
    
    Args:
        results: Результаты анализа
        output_file: Имя выходного файла
    """
    # Подготавливаем данные для JSON
    json_data = {
        "statistics": results["statistics"],
        "file_entities": {}
    }
    
    # Конвертируем сущности в словари
    for filename, entities in results["file_entities"].items():
        json_data["file_entities"][filename] = []
        for entity in entities:
            entity_dict = {
                "name": entity.name,
                "type": entity.entity_type.value,
                "description": entity.description,
                "appearances": entity.appearances,
                "relationships": entity.relationships,
                "metadata": entity.metadata
            }
            json_data["file_entities"][filename].append(entity_dict)
    
    # Сохраняем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены в {output_file}")


def generate_entity_summary(results: Dict[str, Any]) -> None:
    """
    Генерирует краткое резюме по сущностям.
    
    Args:
        results: Результаты анализа
    """
    print(f"\n📋 КРАТКОЕ РЕЗЮМЕ ПО СУЩНОСТЯМ STAR WARS:")
    print("-" * 50)
    
    # Группируем сущности по типам
    entities_by_type = {}
    for entity in results["all_entities"]:
        entity_type = entity.entity_type.value
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity.name)
    
    # Выводим примеры для каждого типа
    for entity_type, entities in entities_by_type.items():
        if entities:
            unique_entities = list(set(entities))[:5]  # Первые 5 уникальных
            print(f"\n{entity_type.upper()}:")
            for entity in unique_entities:
                print(f"  • {entity}")
            if len(set(entities)) > 5:
                print(f"  ... и еще {len(set(entities)) - 5}")


def main() -> None:
    """Основная функция."""
    print("🚀 Запуск анализа сущностей Star Wars...")
    
    # Загружаем файлы
    files = load_star_wars_files()
    
    if not files:
        print("❌ Не найдено файлов для анализа")
        return
    
    # Анализируем сущности
    results = analyze_entities(files)
    
    # Выводим результаты
    print_analysis_results(results)
    
    # Генерируем резюме
    generate_entity_summary(results)
    
    # Сохраняем результаты
    save_results_to_json(results)
    
    print(f"\n✅ Анализ завершен успешно!")


if __name__ == "__main__":
    main()

