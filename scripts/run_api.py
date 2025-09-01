#!/usr/bin/env python3
"""
Скрипт для запуска REST API сервера RAG-бота.
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.api.rag_api import run_api_server


def main():
    """Основная функция."""
    print("🚀 Запуск REST API сервера для RAG-бота...")
    
    # Проверяем существование индекса
    index_path = Path("./data/vector_index")
    if not index_path.exists():
        print("❌ Векторный индекс не найден. Сначала запустите build_index.py")
        return
    
    try:
        # Запускаем API сервер
        run_api_server(host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        print("\n👋 Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")


if __name__ == "__main__":
    main()
