#!/usr/bin/env python3
"""
Быстрый запуск локального RAG-бота без API ключа.

Просто запустите: python run_local_bot.py
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from scripts.rag_bot_local_cli import main

if __name__ == "__main__":
    main()
