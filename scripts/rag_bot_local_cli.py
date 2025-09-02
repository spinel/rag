#!/usr/bin/env python3
"""
CLI интерфейс для локального RAG-бота без API ключа.

Полнофункциональная версия, которая работает без OpenAI API.
"""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.rag_bot_local import LocalRAGBot, RAGConfig


def load_config() -> RAGConfig:
    """Загружает конфигурацию для локального RAG-бота."""
    return RAGConfig(
        vector_db_path="./data/vector_index",
        collection_name="quantumforge_knowledge",
        embedding_model="all-MiniLM-L6-v2",
        search_k=5
    )


def save_conversation(conversation: List[Dict[str, Any]], filename: str = "local_conversation.json") -> None:
    """Сохраняет диалог в файл."""
    output_path = Path("data") / filename
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Диалог сохранен: {output_path}")


def print_help() -> None:
    """Выводит справку по командам."""
    help_text = """
🎯 КОМАНДЫ ЛОКАЛЬНОГО RAG-БОТА:
================================

/help - Показать эту справку
/quit - Выйти из бота
/save - Сохранить текущий диалог
/clear - Очистить экран
/stats - Показать статистику диалога
/examples - Показать примеры вопросов

📝 ПРИМЕРЫ ВОПРОСОВ:
===================

✅ В рамках базы знаний:
• Кто такой Kael Vexar?
• Что такое Plasma Blade?
• Расскажи о Synthetic Wars
• Как работают Tech Automatons?
• Что такое Synth Flux?
• Кто такие Flux Knights?
• Расскажи о Princess Lyra Organa
• Что известно о Xarn Velgor?

❌ Вне тематики (ответит "не знаю"):
• Как приготовить борщ?
• Сколько стоит iPhone 15?
• Какая погода в Москве?
• Как играть в шахматы?
• Что такое квантовая физика?

🔍 ОСОБЕННОСТИ:
===============

• Работает без API ключа OpenAI
• Предустановленные качественные ответы
• Chain-of-Thought рассуждения
• Поиск по векторному индексу
• Указание источников информации
• Оценка уверенности в ответе
• Распознавание вопросов вне тематики

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
        "Кто такие Flux Knights?",
        "Расскажи о Princess Lyra Organa",
        "Что известно о Xarn Velgor?",
        "Как приготовить борщ?",
        "Сколько стоит iPhone 15?"
    ]
    
    print("📋 ПРИМЕРЫ ВОПРОСОВ:")
    print("=" * 30)
    for i, example in enumerate(examples, 1):
        if i <= 8:
            print(f"{i:2d}. {example} ✅")
        else:
            print(f"{i:2d}. {example} ❌")
    print()


def print_stats(conversation: List[Dict[str, Any]]) -> None:
    """Выводит статистику диалога."""
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


def interactive_mode(bot: LocalRAGBot) -> None:
    """Интерактивный режим работы с ботом."""
    conversation = []
    
    print("🤖 ЛОКАЛЬНЫЙ RAG-БОТ QUANTUMFORGE SOFTWARE")
    print("=" * 50)
    print("✅ Работает без API ключа OpenAI")
    print("Введите /help для справки")
    print("Введите /quit для выхода")
    print("-" * 50)
    
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


def demo_mode(bot: LocalRAGBot) -> None:
    """Демонстрационный режим с предустановленными вопросами."""
    demo_questions = [
        "Кто такой Kael Vexar?",
        "Что такое Plasma Blade?",
        "Расскажи о Synthetic Wars",
        "Как работают Tech Automatons?",
        "Что такое Synth Flux?",
        "Как приготовить борщ?"  # Вопрос вне тематики
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
    save_conversation(conversation, "local_demo_conversation.json")
    
    print("\n✅ Демонстрация завершена!")
    print_stats(conversation)


def main() -> None:
    """Основная функция."""
    print("🚀 Запуск локального RAG-бота...")
    
    # Проверяем существование индекса
    index_path = Path("./data/vector_index")
    if not index_path.exists():
        print("❌ Векторный индекс не найден. Сначала запустите build_index.py")
        return
    
    try:
        # Загружаем конфигурацию
        config = load_config()
        
        # Инициализируем бота
        bot = LocalRAGBot(config)
        
        print("\n✅ Локальный RAG-бот готов к работе!")
        print("🎯 Работает без API ключа OpenAI")
        
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
