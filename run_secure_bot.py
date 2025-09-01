#!/usr/bin/env python3
"""
Быстрый запуск защищенного RAG-бота с защитой от промпт-инъекций.

Просто запустите: python run_secure_bot.py
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from src.rag.rag_bot_secure import SecureRAGBot, RAGConfig, SecurityConfig


def main():
    """Запуск защищенного RAG-бота."""
    print("🛡️ Запуск защищенного RAG-бота...")
    
    # Конфигурация
    rag_config = RAGConfig(
        vector_db_path="./data/vector_index",
        collection_name="quantumforge_knowledge",
        embedding_model="all-MiniLM-L6-v2",
        search_k=5
    )
    
    security_config = SecurityConfig(
        enable_pre_filtering=True,
        enable_post_checking=True,
        enable_system_removal=True,
        enable_sensitive_filtering=True
    )
    
    # Инициализация защищенного бота
    bot = SecureRAGBot(rag_config, security_config)
    
    print("\n✅ Защищенный RAG-бот готов к работе!")
    print("🎯 Работает с защитой от промпт-инъекций")
    print("Введите /quit для выхода")
    print("-" * 50)
    
    # Интерактивный режим
    while True:
        try:
            query = input("\n❓ Ваш вопрос: ").strip()
            
            if query.lower() == '/quit':
                print("👋 До свидания!")
                break
            elif not query:
                continue
            
            # Получаем ответ
            response = bot.ask(query)
            
            # Выводим ответ
            print("\n" + "=" * 60)
            print(bot.format_response(response))
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue


if __name__ == "__main__":
    main()

