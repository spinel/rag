#!/usr/bin/env python3
"""
Скрипт для обработки файлов Star Wars с заменой терминов.

Заменяет все термины Star Wars на вымышленные аналоги
и создает уникальную базу знаний.
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.entity_extraction.terms_mapper import TermsMapper


def print_processing_stats(stats: Dict[str, Any]) -> None:
    """
    Выводит статистику обработки.
    
    Args:
        stats: Статистика обработки
    """
    print("\n" + "="*60)
    print("СТАТИСТИКА ОБРАБОТКИ ФАЙЛОВ")
    print("="*60)
    
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"Всего файлов: {stats['total_files']}")
    print(f"Обработано успешно: {stats['processed_files']}")
    print(f"Ошибок: {stats['failed_files']}")
    
    print(f"\n📁 ПО КАТЕГОРИЯМ:")
    for category, cat_stats in stats["categories"].items():
        if cat_stats["total"] > 0:
            print(f"\n  {category.upper()}:")
            print(f"    Всего: {cat_stats['total']}")
            print(f"    Обработано: {cat_stats['processed']}")
            print(f"    Ошибок: {cat_stats['failed']}")
    
    success_rate = (stats["processed_files"] / stats["total_files"] * 100) if stats["total_files"] > 0 else 0
    print(f"\n✅ Процент успешной обработки: {success_rate:.1f}%")


def print_mapping_stats(mapper: TermsMapper) -> None:
    """
    Выводит статистику по словарю соответствий.
    
    Args:
        mapper: Маппер терминов
    """
    stats = mapper.get_mapping_statistics()
    
    print("\n" + "="*60)
    print("СТАТИСТИКА СЛОВАРЯ СООТВЕТСТВИЙ")
    print("="*60)
    
    print(f"\n📚 ОБЩАЯ СТАТИСТИКА:")
    print(f"Всего терминов: {stats['total_terms']}")
    
    print(f"\n📈 ПО КАТЕГОРИЯМ:")
    for category, count in stats["categories"].items():
        if count > 0:
            print(f"  {category.capitalize()}: {count}")
    
    print(f"\n🔍 ПРИМЕРЫ ЗАМЕН:")
    for category, terms in stats["category_details"].items():
        if terms:
            print(f"\n  {category.upper()}:")
            for original, replacement in terms[:3]:  # Показываем первые 3
                print(f"    {original} → {replacement}")
            if len(terms) > 3:
                print(f"    ... и еще {len(terms) - 3}")


def main() -> None:
    """Основная функция."""
    print("🚀 Запуск обработки файлов Star Wars...")
    
    # Инициализируем маппер
    mapper = TermsMapper()
    
    # Пути к директориям
    input_dir = Path("starwars")
    output_dir = Path("knowledge_base")
    
    print(f"\n📂 Входная директория: {input_dir}")
    print(f"📂 Выходная директория: {output_dir}")
    
    # Проверяем существование входной директории
    if not input_dir.exists():
        print(f"❌ Входная директория {input_dir} не найдена")
        return
    
    # Обрабатываем все файлы
    print(f"\n🔄 Обработка файлов...")
    stats = mapper.process_directory(input_dir, output_dir)
    
    # Выводим статистику обработки
    print_processing_stats(stats)
    
    # Выводим статистику словаря
    print_mapping_stats(mapper)
    
    # Сохраняем словарь соответствий
    terms_map_path = output_dir / "terms_map.json"
    if mapper.save_terms_mapping(terms_map_path):
        print(f"\n💾 Словарь соответствий сохранен: {terms_map_path}")
    else:
        print(f"\n❌ Ошибка при сохранении словаря соответствий")
    
    # Создаем README для базы знаний
    create_knowledge_base_readme(output_dir, stats, mapper)
    
    print(f"\n✅ Обработка завершена успешно!")
    print(f"📁 Уникальная база знаний создана в: {output_dir}")


def create_knowledge_base_readme(output_dir: Path, stats: Dict[str, Any], mapper: TermsMapper) -> None:
    """
    Создает README файл для базы знаний.
    
    Args:
        output_dir: Директория базы знаний
        stats: Статистика обработки
        mapper: Маппер терминов
    """
    readme_content = f"""# Уникальная база знаний

Эта база знаний содержит обработанные документы с замененными терминами.

## Статистика обработки

- **Всего файлов**: {stats['total_files']}
- **Обработано успешно**: {stats['processed_files']}
- **Ошибок**: {stats['failed_files']}

## Структура базы знаний

```
knowledge_base/
├── characters/     # Персонажи
├── events/         # События
├── locations/      # Локации
├── technology/     # Технологии
└── terms_map.json  # Словарь соответствий
```

## Словарь соответствий

Основные замены терминов:
- "Darth Vader" → "Xarn Velgor"
- "Death Star" → "Void Core"
- "The Force" → "Synth Flux"
- "Luke Skywalker" → "Kael Vexar"
- "Obi-Wan Kenobi" → "Toren Vexis"

Полный словарь соответствий находится в файле `terms_map.json`.

## Использование

Все документы в этой базе знаний были обработаны для удаления
оригинальных терминов Star Wars и замены их на вымышленные аналоги.
Тексты остались логичными и читаемыми, но больше не содержат
прямых отсылок к вселенной Star Wars.

## Категории документов

"""
    
    # Добавляем информацию по категориям
    for category, cat_stats in stats["categories"].items():
        if cat_stats["total"] > 0:
            readme_content += f"- **{category.capitalize()}**: {cat_stats['processed']} файлов\n"
    
    readme_content += f"""
## Техническая информация

- Обработка выполнена с помощью TermsMapper
- Использованы регулярные выражения для точной замены
- Сохранена структура оригинальных документов
- Все файлы в кодировке UTF-8

## Примечания

- Некоторые термины могут быть пропущены, если они не были
  включены в словарь соответствий
- Контекст и логика текстов сохранены
- Имена файлов остались без изменений
"""
    
    # Сохраняем README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📝 README создан: {readme_path}")


if __name__ == "__main__":
    main()
