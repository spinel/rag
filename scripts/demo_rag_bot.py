#!/usr/bin/env python3
"""
Демонстрационная версия RAG-бота без реального API ключа.

Показывает, как работает система поиска и промптинг.
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_vectorstore():
    """Загружает векторное хранилище."""
    print("Загрузка векторного хранилища...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        collection_name="quantumforge_knowledge",
        embedding_function=embeddings,
        persist_directory="./data/vector_index"
    )
    
    return vectorstore


def demo_search(vectorstore, query: str):
    """Демонстрирует поиск по векторному индексу."""
    print(f"\n🔍 ПОИСК: '{query}'")
    print("=" * 60)
    
    # Выполняем поиск
    docs = vectorstore.similarity_search(query, k=3)
    
    print("📚 НАЙДЕННЫЕ ДОКУМЕНТЫ:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. 📄 {doc.metadata.get('filename', 'Unknown')} ({doc.metadata.get('category', 'Unknown')})")
        print(f"   📍 {doc.metadata.get('chunk_id', 'Unknown')}")
        print(f"   📝 {doc.page_content[:300]}...")
    
    return docs


def demo_prompt_generation(query: str, docs: List):
    """Демонстрирует генерацию промпта."""
    print(f"\n🤖 ГЕНЕРАЦИЯ ПРОМПТА")
    print("=" * 60)
    
    # Форматируем контекст
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        context_parts.append(f"Документ {i} ({category}): {source}")
        context_parts.append(doc.page_content)
        context_parts.append("---")
    
    context = "\n".join(context_parts)
    
    # Few-shot примеры
    few_shot_examples = """Q: Кто такой Kael Vexar?
A: Kael Vexar - это Synth Flux-чувствительный Terran мужчина, легендарный Flux Knight Mentor, который сражался в Void Civil Conflict во время правления Galactic Void Empire.

Q: Что такое Plasma Blade?
A: Plasma Blade - это оружие, используемое Flux Knights. Это энергетический меч, который является жизнью воина.

Q: Что такое Synth Flux?
A: Synth Flux - это энергия, которая пронизывает вселенную. Flux Knights чувствительны к Synth Flux и могут использовать его для различных способностей."""
    
    # Системный промпт
    system_prompt = f"""Ты помощник QuantumForge Software, который отвечает на вопросы на основе предоставленной базы знаний.

ВАЖНО: Ты должен всегда объяснять свои шаги рассуждения перед тем, как дать ответ.

Правила:
1. Сначала проанализируй найденные документы
2. Объясни свои шаги рассуждения
3. Дай четкий и точный ответ
4. Если информации недостаточно, скажи "Я не знаю"
5. Используй только информацию из предоставленных документов
6. Отвечай на русском языке

Контекст:
{context}

Примеры вопросов и ответов:
{few_shot_examples}

Вопрос: {query}

Рассуждение и ответ:"""
    
    print("📋 СИСТЕМНЫЙ ПРОМПТ:")
    print(system_prompt[:1000] + "..." if len(system_prompt) > 1000 else system_prompt)
    
    return system_prompt


def demo_chain_of_thought(query: str, docs: List):
    """Демонстрирует Chain-of-Thought рассуждение."""
    print(f"\n🧠 CHAIN-OF-THOUGHT РАССУЖДЕНИЕ")
    print("=" * 60)
    
    # Анализируем найденные документы
    print("1️⃣ АНАЛИЗ ДОКУМЕНТОВ:")
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get('filename', 'Unknown')
        category = doc.metadata.get('category', 'Unknown')
        content_preview = doc.page_content[:100] + "..."
        print(f"   Документ {i}: {filename} ({category})")
        print(f"   Содержание: {content_preview}")
    
    # Определяем релевантность
    print(f"\n2️⃣ ОЦЕНКА РЕЛЕВАНТНОСТИ:")
    if docs:
        print(f"   ✅ Найдено {len(docs)} релевантных документов")
        print(f"   📊 Уверенность: {min(len(docs) / 3 * 100, 95):.0f}%")
    else:
        print("   ❌ Релевантные документы не найдены")
        print("   📊 Уверенность: 5%")
    
    # Формируем рассуждение
    print(f"\n3️⃣ ЛОГИЧЕСКОЕ РАССУЖДЕНИЕ:")
    if docs:
        print("   Шаг 1: Анализирую найденные документы")
        print("   Шаг 2: Извлекаю ключевую информацию")
        print("   Шаг 3: Формирую структурированный ответ")
        print("   Шаг 4: Проверяю точность информации")
    else:
        print("   Шаг 1: Проверяю базу знаний")
        print("   Шаг 2: Не нахожу релевантной информации")
        print("   Шаг 3: Формирую ответ 'не знаю'")
    
    print(f"\n4️⃣ ФИНАЛЬНЫЙ ОТВЕТ:")
    if docs:
        print("   ✅ Могу дать ответ на основе найденной информации")
    else:
        print("   ❌ Не могу дать ответ - информации недостаточно")


def main():
    """Основная функция демонстрации."""
    print("🎬 ДЕМОНСТРАЦИЯ RAG-БОТА")
    print("=" * 60)
    print("Показываем работу системы без реального API ключа")
    print("-" * 60)
    
    # Проверяем существование индекса
    index_path = Path("./data/vector_index")
    if not index_path.exists():
        print("❌ Векторный индекс не найден. Сначала запустите build_index.py")
        return
    
    try:
        # Загружаем векторное хранилище
        vectorstore = load_vectorstore()
        print("✅ Векторное хранилище загружено")
        
        # Демонстрационные вопросы
        demo_questions = [
            "Кто такой Kael Vexar?",
            "Что такое Plasma Blade?",
            "Расскажи о Synthetic Wars",
            "Как приготовить борщ?"  # Вопрос вне тематики
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{'='*80}")
            print(f"ДЕМОНСТРАЦИЯ {i}/{len(demo_questions)}")
            print(f"{'='*80}")
            
            # Демонстрируем поиск
            docs = demo_search(vectorstore, question)
            
            # Демонстрируем генерацию промпта
            prompt = demo_prompt_generation(question, docs)
            
            # Демонстрируем Chain-of-Thought
            demo_chain_of_thought(question, docs)
            
            if i < len(demo_questions):
                input("\nНажмите Enter для следующего вопроса...")
        
        print(f"\n{'='*80}")
        print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print(f"{'='*80}")
        print("\n📋 ЧТО БЫЛО ПОКАЗАНО:")
        print("1. Поиск по векторному индексу")
        print("2. Генерация промпта с Few-shot примерами")
        print("3. Chain-of-Thought рассуждение")
        print("4. Оценка релевантности и уверенности")
        print("\n🚀 Для полной работы нужен API ключ OpenAI")
        
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")


if __name__ == "__main__":
    main()
