# RAG-бот для QuantumForge Software

Интеллектуальный бот на основе Retrieval-Augmented Generation (RAG) с техниками промптинга (Few-shot и Chain-of-Thought) для корпоративной базы знаний компании QuantumForge Software.

## Описание проекта

QuantumForge Software - финско-эстонская продуктовая компания, специализирующаяся на SaaS-платформе для моделирования промышленных объектов "Digital Twin". 

### Проблема
- Разрозненность и дублирование информации в корпоративной базе знаний
- Время поиска информации до 4 часов в неделю для новых сотрудников
- Устаревание документации через 2-3 месяца после релиза
- Сложность подготовки к аудитам (140 часов работы GRC-команды)

### Решение
RAG-бот с техниками промптинга, который:
- Быстро и точно отвечает на вопросы сотрудников
- Использует Few-shot prompting для улучшения качества ответов
- Применяет Chain-of-Thought для прозрачности рассуждений
- Выявляет пробелы в документации
- Помогает системно улучшать качество базы знаний
- Сокращает время поиска информации

## Структура проекта

```
rag/
├── research/                    # Исследования и анализ
│   ├── 01-models-and-infrastructure-analysis.md
│   └── ...
├── src/                        # Исходный код
│   ├── core/                   # Основная логика
│   ├── rag/                    # RAG-бот с промптингом
│   ├── vectorstore/            # Векторные базы данных
│   ├── entity_extraction/      # Извлечение сущностей
│   └── api/                    # API интерфейс
├── tests/                      # Тесты
├── data/                       # Данные и документы
├── config/                     # Конфигурация
└── docs/                       # Документация
```

## Технологический стек

### Рекомендуемая конфигурация (Гибридный подход)
- **LLM**: OpenAI GPT-3.5-turbo
- **Эмбеддинги**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Векторная БД**: ChromaDB
- **Техники промптинга**: Few-shot + Chain-of-Thought
- **Интерфейсы**: CLI + REST API

### Альтернативные варианты
- Полностью локальный (Llama 2 + FAISS)
- Полностью облачный (OpenAI + Pinecone)

## Установка и запуск

### Требования
- Python 3.9+
- 16GB RAM (минимально)
- 32-64GB RAM (рекомендуется для продакшена)

### Установка

1. Клонировать репозиторий:
```bash
git clone <repository-url>
cd rag
```

2. Создать виртуальное окружение:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows
```

3. Установить зависимости:
```bash
pip install -r requirements.txt
```

4. Настроить переменные окружения:
```bash
cp .env.example .env
# Отредактировать .env файл
```

5. Запустить RAG-бот:

**CLI интерфейс:**
```bash
python scripts/rag_bot_cli.py
```

**REST API:**
```bash
python scripts/run_api.py
```

**Тестирование:**
```bash
python scripts/test_rag_bot.py
```

## Конфигурация

Создайте файл `.env` со следующими переменными:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# ChromaDB
CHROMA_DB_HOST=localhost
CHROMA_DB_PORT=8000

# Настройки приложения
LOG_LEVEL=INFO
ENVIRONMENT=development
```

## Использование

### API Endpoints

- `POST /ask` - Задать вопрос RAG-боту
- `POST /ask/batch` - Задать несколько вопросов
- `GET /health` - Проверка состояния системы
- `GET /stats` - Статистика API
- `GET /examples` - Примеры вопросов

### Пример использования

```python
import requests

# Отправить вопрос
response = requests.post("http://localhost:8000/ask", json={
    "question": "Кто такой Kael Vexar?"
})

result = response.json()
print(f"Ответ: {result['answer']}")
print(f"Уверенность: {result['confidence']:.1%}")
```

## Разработка

### Структура кода
- Следуем принципам SOLID
- Используем типизацию (type hints)
- Пишем тесты для всех компонентов
- Документируем код на русском языке

### Запуск тестов
```bash
pytest tests/
```

### Форматирование кода
```bash
black src/
isort src/
flake8 src/
```

## Мониторинг и логирование

- Используем Loguru для структурированного логирования
- Метрики производительности и качества ответов
- Система обратной связи для улучшения ответов

## Лицензия

Внутренний проект QuantumForge Software

## Контакты

- Команда разработки: dev@quantumforge.software
- Техническая поддержка: support@quantumforge.software
