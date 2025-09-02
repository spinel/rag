#!/usr/bin/env python3
"""
Консольный интерфейс для RAG-бота.

Предоставляет интерактивный способ взаимодействия с RAG-ботом
через командную строку.
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.rag_bot import RAGBot, RAGConfig


def load_config() -> RAGConfig:
    """
    Загружает конфигурацию RAG-бота.
    
    Returns:
        RAGConfig: Конфигурация бота
    """
    return RAGConfig(
        vector_db_path="./data/vector_index",
        collection_name="quantumforge_knowledge",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000,
        search_k=5
    )


def save_conversation(conversation: List[Dict[str, Any]], filename: str = "conversation.json") -> None:
    """
    Сохраняет диалог в файл.
    
    Args:
        conversation: Список сообщений диалога
        filename: Имя файла для сохранения
    """
    output_path = Path("data") / filename
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Диалог сохранен: {output_path}")


def print_help() -> None:
    """Выводит справку по командам."""
    help_text = """
🎯 КОМАНДЫ RAG-БОТА:
===================

/help - Показать эту справку
/quit - Выйти из бота
/save - Сохранить текущий диалог
/clear - Очистить экран
/stats - Показать статистику диалога
/examples - Показать примеры вопросов

📝 ПРИМЕРЫ ВОПРОСОВ:
===================

• Кто такой Kael Vexar?
• Что такое Plasma Blade?
• Расскажи о Synthetic Wars
• Как работают Tech Automatons?
• Что такое Synth Flux?
• Какие технологии используются?
• Где происходили основные события?

🔍 ОСОБЕННОСТИ:
===============

• Few-shot prompting для лучших ответов
• Chain-of-Thought рассуждения
• Поиск по векторному индексу
• Указание источников информации
• Оценка уверенности в ответе

"""
    print(help_text)


def print_examples() -> None:
    """Выводит примеры вопросов."""
    examples = [
        "Кто такой Kael Vexar?",
        "Что такое Plasma Blade?",
        "Расскажи о Synthetic Wars",
        "Как работают Tech Automatons?",
        "Что такое Synth Flux?",
        "Какие технологии используются?",
        "Где происходили основные события?",
        "Кто участвовал в Void Civil Conflict?",
        "Что такое Flux Knights?",
        "Как работает Mind Control?"
    ]
    
    print("📋 ПРИМЕРЫ ВОПРОСОВ:")
    print("=" * 30)
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example}")
    print()


def print_stats(conversation: List[Dict[str, Any]]) -> None:
    """
    Выводит статистику диалога.
    
    Args:
        conversation: Список сообщений диалога
    """
    if not conversation:
        print("📊 Статистика: диалог пуст")
        return
    
    total_questions = len([msg for msg in conversation if msg["type"] == "question"])
    total_answers = len([msg for msg in conversation if msg["type"] == "answer"])
    avg_confidence = sum(msg.get("confidence", 0) for msg in conversation if msg["type"] == "answer") / max(total_answers, 1)
    avg_time = sum(msg.get("total_time", 0) for msg in conversation if msg["type"] == "answer") / max(total_answers, 1)
    
    print("📊 СТАТИСТИКА ДИАЛОГА:")
    print("=" * 25)
    print(f"Вопросов задано: {total_questions}")
    print(f"Ответов получено: {total_answers}")
    print(f"Средняя уверенность: {avg_confidence:.1%}")
    print(f"Среднее время ответа: {avg_time:.2f}с")
    print()


def interactive_mode(bot: RAGBot) -> None:
    """
    Интерактивный режим работы с ботом.
    
    Args:
        bot: Инициализированный RAG-бот
    """
    conversation = []
    
    print("🤖 RAG-БОТ QUANTUMFORGE SOFTWARE")
    print("=" * 40)
    print("Введите /help для справки")
    print("Введите /quit для выхода")
    print("-" * 40)
    
    while True:
        try:
            # Получаем вопрос от пользователя
            query = input("\n❓ Ваш вопрос: ").strip()
            
            # Обрабатываем команды
            if query.lower() == '/quit':
                print("👋 До свидания!")
                break
            elif query.lower() == '/help':
                print_help()
                continue
            elif query.lower() == '/save':
                save_conversation(conversation)
                continue
            elif query.lower() == '/clear':
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif query.lower() == '/stats':
                print_stats(conversation)
                continue
            elif query.lower() == '/examples':
                print_examples()
                continue
            elif not query:
                continue
            
            # Сохраняем вопрос
            conversation.append({
                "type": "question",
                "query": query,
                "timestamp": time.time()
            })
            
            # Получаем ответ от бота
            print(f"\n🤖 Обрабатываю вопрос...")
            response = bot.ask(query)
            
            # Выводим ответ
            print("\n" + "=" * 60)
            print(bot.format_response(response))
            print("=" * 60)
            
            # Сохраняем ответ
            conversation.append({
                "type": "answer",
                "query": query,
                "answer": response.answer,
                "reasoning": response.reasoning,
                "sources": response.sources,
                "confidence": response.confidence,
                "total_time": response.total_time,
                "timestamp": time.time()
            })
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue


def demo_mode(bot: RAGBot) -> None:
    """
    Демонстрационный режим с предустановленными вопросами.
    
    Args:
        bot: Инициализированный RAG-бот
    """
    demo_questions = [
        "Кто такой Kael Vexar?",
        "Что такое Plasma Blade?",
        "Расскажи о Synthetic Wars",
        "Как работают Tech Automatons?",
        "Что такое Synth Flux?"
    ]
    
    print("🎬 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("=" * 40)
    print(f"Будет задано {len(demo_questions)} вопросов")
    print("-" * 40)
    
    conversation = []
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. ВОПРОС: {question}")
        print("-" * 40)
        
        # Сохраняем вопрос
        conversation.append({
            "type": "question",
            "query": question,
            "timestamp": time.time()
        })
        
        # Получаем ответ
        response = bot.ask(question)
        
        # Выводим ответ
        print("\n" + "=" * 60)
        print(bot.format_response(response))
        print("=" * 60)
        
        # Сохраняем ответ
        conversation.append({
            "type": "answer",
            "query": question,
            "answer": response.answer,
            "reasoning": response.reasoning,
            "sources": response.sources,
            "confidence": response.confidence,
            "total_time": response.total_time,
            "timestamp": time.time()
        })
        
        # Пауза между вопросами
        if i < len(demo_questions):
            input("\nНажмите Enter для следующего вопроса...")
    
    # Сохраняем демонстрационный диалог
    save_conversation(conversation, "demo_conversation.json")
    
    print("\n✅ Демонстрация завершена!")
    print_stats(conversation)


def main() -> None:
    """Основная функция."""
    import time
    
    print("🚀 Запуск RAG-бота...")
    
    # Проверяем существование индекса
    index_path = Path("./data/vector_index")
    if not index_path.exists():
        print("❌ Векторный индекс не найден. Сначала запустите build_index.py")
        return
    
    try:
        # Загружаем конфигурацию
        config = load_config()
        
        # Инициализируем бота
        bot = RAGBot(config)
        
        print("\n✅ RAG-бот готов к работе!")
        
        # Выбираем режим
        print("\n🎯 Выберите режим работы:")
        print("1. Интерактивный режим")
        print("2. Демонстрационный режим")
        
        while True:
            choice = input("\nВаш выбор (1 или 2): ").strip()
            
            if choice == "1":
                interactive_mode(bot)
                break
            elif choice == "2":
                demo_mode(bot)
                break
            else:
                print("❌ Неверный выбор. Введите 1 или 2.")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return


if __name__ == "__main__":
    main()
