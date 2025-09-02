#!/usr/bin/env python3
"""
Тестовый скрипт для демонстрации работы RAG-бота.

Показывает примеры успешных диалогов и случаев "Я не знаю".
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.rag_bot import RAGBot, RAGConfig


def load_config() -> RAGConfig:
    """Загружает конфигурацию для тестирования."""
    return RAGConfig(
        vector_db_path="./data/vector_index",
        collection_name="quantumforge_knowledge",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000,
        search_k=5
    )


def test_successful_dialogs(bot: RAGBot) -> List[Dict[str, Any]]:
    """
    Тестирует успешные диалоги.
    
    Args:
        bot: RAG-бот
        
    Returns:
        List[Dict[str, Any]]: Результаты тестов
    """
    print("✅ ТЕСТИРОВАНИЕ УСПЕШНЫХ ДИАЛОГОВ")
    print("=" * 50)
    
    successful_questions = [
        "Кто такой Kael Vexar?",
        "Что такое Plasma Blade?",
        "Расскажи о Synthetic Wars",
        "Как работают Tech Automatons?",
        "Что такое Synth Flux?"
    ]
    
    results = []
    
    for i, question in enumerate(successful_questions, 1):
        print(f"\n{i}. ВОПРОС: {question}")
        print("-" * 40)
        
        try:
            response = bot.ask(question)
            
            # Выводим краткий результат
            print(f"✅ ОТВЕТ: {response.answer[:200]}...")
            print(f"📊 Уверенность: {response.confidence * 100:.0f}%")
            print(f"⏱️  Время: {response.total_time:.2f}с")
            print(f"📚 Источников: {len(response.sources)}")
            
            # Сохраняем результат
            results.append({
                "type": "successful",
                "question": question,
                "answer": response.answer,
                "reasoning": response.reasoning,
                "confidence": response.confidence,
                "total_time": response.total_time,
                "sources_count": len(response.sources)
            })
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            results.append({
                "type": "error",
                "question": question,
                "error": str(e)
            })
    
    return results


def test_unknown_questions(bot: RAGBot) -> List[Dict[str, Any]]:
    """
    Тестирует вопросы, на которые бот должен ответить "Я не знаю".
    
    Args:
        bot: RAG-бот
        
    Returns:
        List[Dict[str, Any]]: Результаты тестов
    """
    print("\n\n❓ ТЕСТИРОВАНИЕ ВОПРОСОВ 'Я НЕ ЗНАЮ'")
    print("=" * 50)
    
    unknown_questions = [
        "Как приготовить борщ?",
        "Какая погода в Москве?",
        "Сколько стоит iPhone 15?",
        "Как играть в шахматы?",
        "Что такое квантовая физика?"
    ]
    
    results = []
    
    for i, question in enumerate(unknown_questions, 1):
        print(f"\n{i}. ВОПРОС: {question}")
        print("-" * 40)
        
        try:
            response = bot.ask(question)
            
            # Проверяем, содержит ли ответ "не знаю"
            is_unknown = any(phrase in response.answer.lower() 
                           for phrase in ["не знаю", "не могу", "нет информации", "не найдено"])
            
            if is_unknown:
                print(f"✅ ПРАВИЛЬНО: Бот признал, что не знает ответ")
            else:
                print(f"⚠️  ВНИМАНИЕ: Бот дал ответ, хотя должен был сказать 'не знаю'")
            
            print(f"📝 ОТВЕТ: {response.answer[:200]}...")
            print(f"📊 Уверенность: {response.confidence * 100:.0f}%")
            print(f"⏱️  Время: {response.total_time:.2f}с")
            
            # Сохраняем результат
            results.append({
                "type": "unknown_question",
                "question": question,
                "answer": response.answer,
                "reasoning": response.reasoning,
                "confidence": response.confidence,
                "total_time": response.total_time,
                "correctly_unknown": is_unknown
            })
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            results.append({
                "type": "error",
                "question": question,
                "error": str(e)
            })
    
    return results


def save_test_results(successful_results: List[Dict[str, Any]], 
                     unknown_results: List[Dict[str, Any]]) -> None:
    """
    Сохраняет результаты тестирования.
    
    Args:
        successful_results: Результаты успешных диалогов
        unknown_results: Результаты вопросов "не знаю"
    """
    output_path = Path("data/rag_test_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    test_results = {
        "timestamp": time.time(),
        "successful_dialogs": successful_results,
        "unknown_questions": unknown_results,
        "summary": {
            "total_successful": len(successful_results),
            "total_unknown": len(unknown_results),
            "successful_errors": len([r for r in successful_results if r["type"] == "error"]),
            "unknown_errors": len([r for r in unknown_results if r["type"] == "error"]),
            "correctly_unknown": len([r for r in unknown_results if r.get("correctly_unknown", False)])
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты тестирования сохранены: {output_path}")


def print_summary(successful_results: List[Dict[str, Any]], 
                 unknown_results: List[Dict[str, Any]]) -> None:
    """
    Выводит сводку результатов тестирования.
    
    Args:
        successful_results: Результаты успешных диалогов
        unknown_results: Результаты вопросов "не знаю"
    """
    print("\n" + "=" * 60)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    # Статистика успешных диалогов
    successful_count = len([r for r in successful_results if r["type"] == "successful"])
    successful_errors = len([r for r in successful_results if r["type"] == "error"])
    
    if successful_count > 0:
        avg_confidence = sum(r["confidence"] for r in successful_results if r["type"] == "successful") / successful_count
        avg_time = sum(r["total_time"] for r in successful_results if r["type"] == "successful") / successful_count
    else:
        avg_confidence = 0
        avg_time = 0
    
    print(f"✅ Успешные диалоги: {successful_count}")
    print(f"❌ Ошибки в успешных: {successful_errors}")
    print(f"📊 Средняя уверенность: {avg_confidence:.1%}")
    print(f"⏱️  Среднее время ответа: {avg_time:.2f}с")
    
    # Статистика вопросов "не знаю"
    unknown_count = len([r for r in unknown_results if r["type"] == "unknown_question"])
    unknown_errors = len([r for r in unknown_results if r["type"] == "error"])
    correctly_unknown = len([r for r in unknown_results if r.get("correctly_unknown", False)])
    
    print(f"\n❓ Вопросы 'не знаю': {unknown_count}")
    print(f"❌ Ошибки в вопросах 'не знаю': {unknown_errors}")
    print(f"✅ Правильно ответили 'не знаю': {correctly_unknown}")
    
    if unknown_count > 0:
        accuracy = correctly_unknown / unknown_count
        print(f"🎯 Точность распознавания 'не знаю': {accuracy:.1%}")
    
    print("\n" + "=" * 60)


def main() -> None:
    """Основная функция."""
    import time
    
    print("🧪 ТЕСТИРОВАНИЕ RAG-БОТА")
    print("=" * 40)
    
    # Проверяем существование индекса
    index_path = Path("./data/vector_index")
    if not index_path.exists():
        print("❌ Векторный индекс не найден. Сначала запустите build_index.py")
        return
    
    try:
        # Загружаем конфигурацию
        config = load_config()
        
        # Инициализируем бота
        print("🤖 Инициализация RAG-бота...")
        bot = RAGBot(config)
        
        # Тестируем успешные диалоги
        successful_results = test_successful_dialogs(bot)
        
        # Тестируем вопросы "не знаю"
        unknown_results = test_unknown_questions(bot)
        
        # Выводим сводку
        print_summary(successful_results, unknown_results)
        
        # Сохраняем результаты
        save_test_results(successful_results, unknown_results)
        
        print("\n✅ Тестирование завершено!")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return


if __name__ == "__main__":
    main()
