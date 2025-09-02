"""
REST API для RAG-бота.

Предоставляет HTTP интерфейс для взаимодействия с RAG-ботом.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import time

from src.rag.rag_bot import RAGBot, RAGConfig


# Модели данных для API
class QuestionRequest(BaseModel):
    """Модель запроса вопроса."""
    question: str
    search_k: Optional[int] = 5


class SourceInfo(BaseModel):
    """Модель информации об источнике."""
    filename: str
    category: str
    chunk_id: str
    content_preview: str


class RAGResponse(BaseModel):
    """Модель ответа RAG-бота."""
    question: str
    answer: str
    reasoning: str
    sources: List[SourceInfo]
    confidence: float
    search_time: float
    generation_time: float
    total_time: float


class HealthResponse(BaseModel):
    """Модель ответа о состоянии сервиса."""
    status: str
    model: str
    vector_db: str
    uptime: float


class StatsResponse(BaseModel):
    """Модель статистики."""
    total_requests: int
    avg_response_time: float
    avg_confidence: float
    success_rate: float


# Инициализация FastAPI приложения
app = FastAPI(
    title="RAG Bot API",
    description="API для RAG-бота QuantumForge Software",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
rag_bot: Optional[RAGBot] = None
start_time = time.time()
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "total_response_time": 0.0,
    "total_confidence": 0.0
}


def initialize_rag_bot() -> RAGBot:
    """
    Инициализирует RAG-бота.
    
    Returns:
        RAGBot: Инициализированный бот
    """
    config = RAGConfig(
        vector_db_path="./data/vector_index",
        collection_name="quantumforge_knowledge",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000,
        search_k=5
    )
    
    return RAGBot(config)


@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения."""
    global rag_bot
    print("🚀 Инициализация RAG-бота...")
    rag_bot = initialize_rag_bot()
    print("✅ RAG-бот готов к работе!")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Корневой endpoint."""
    return {
        "message": "RAG Bot API для QuantumForge Software",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса."""
    global rag_bot, start_time
    
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG-бот не инициализирован")
    
    return HealthResponse(
        status="healthy",
        model=rag_bot.config.llm_model,
        vector_db=rag_bot.config.vector_db_path,
        uptime=time.time() - start_time
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Получение статистики API."""
    global request_stats
    
    if request_stats["total_requests"] == 0:
        return StatsResponse(
            total_requests=0,
            avg_response_time=0.0,
            avg_confidence=0.0,
            success_rate=0.0
        )
    
    avg_response_time = request_stats["total_response_time"] / request_stats["total_requests"]
    avg_confidence = request_stats["total_confidence"] / request_stats["successful_requests"]
    success_rate = request_stats["successful_requests"] / request_stats["total_requests"]
    
    return StatsResponse(
        total_requests=request_stats["total_requests"],
        avg_response_time=avg_response_time,
        avg_confidence=avg_confidence,
        success_rate=success_rate
    )


@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    """
    Задает вопрос RAG-боту.
    
    Args:
        request: Запрос с вопросом
        
    Returns:
        RAGResponse: Ответ бота
    """
    global rag_bot, request_stats
    
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG-бот не инициализирован")
    
    # Обновляем статистику
    request_stats["total_requests"] += 1
    api_start_time = time.time()
    
    try:
        # Временно изменяем search_k если указан
        original_search_k = rag_bot.config.search_k
        if request.search_k:
            rag_bot.config.search_k = request.search_k
        
        # Получаем ответ от бота
        response = rag_bot.ask(request.question)
        
        # Восстанавливаем оригинальный search_k
        rag_bot.config.search_k = original_search_k
        
        # Обновляем статистику успешных запросов
        request_stats["successful_requests"] += 1
        request_stats["total_response_time"] += response.total_time
        request_stats["total_confidence"] += response.confidence
        
        # Преобразуем источники в модель API
        sources = []
        for source in response.sources:
            sources.append(SourceInfo(
                filename=source["filename"],
                category=source["category"],
                chunk_id=source["chunk_id"],
                content_preview=source["content_preview"]
            ))
        
        return RAGResponse(
            question=request.question,
            answer=response.answer,
            reasoning=response.reasoning,
            sources=sources,
            confidence=response.confidence,
            search_time=response.total_time - response.total_time * 0.3,  # Примерное время поиска
            generation_time=response.total_time * 0.7,  # Примерное время генерации
            total_time=response.total_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")


@app.post("/ask/batch", response_model=List[RAGResponse])
async def ask_batch_questions(questions: List[str]):
    """
    Задает несколько вопросов одновременно.
    
    Args:
        questions: Список вопросов
        
    Returns:
        List[RAGResponse]: Список ответов
    """
    global rag_bot
    
    if not rag_bot:
        raise HTTPException(status_code=503, detail="RAG-бот не инициализирован")
    
    if len(questions) > 10:
        raise HTTPException(status_code=400, detail="Максимум 10 вопросов за раз")
    
    responses = []
    
    for question in questions:
        try:
            response = rag_bot.ask(question)
            
            # Преобразуем источники
            sources = []
            for source in response.sources:
                sources.append(SourceInfo(
                    filename=source["filename"],
                    category=source["category"],
                    chunk_id=source["chunk_id"],
                    content_preview=source["content_preview"]
                ))
            
            responses.append(RAGResponse(
                question=question,
                answer=response.answer,
                reasoning=response.reasoning,
                sources=sources,
                confidence=response.confidence,
                search_time=response.total_time * 0.3,
                generation_time=response.total_time * 0.7,
                total_time=response.total_time
            ))
            
        except Exception as e:
            # Возвращаем ошибку для конкретного вопроса
            responses.append(RAGResponse(
                question=question,
                answer=f"Ошибка: {str(e)}",
                reasoning="Не удалось обработать запрос",
                sources=[],
                confidence=0.0,
                search_time=0.0,
                generation_time=0.0,
                total_time=0.0
            ))
    
    return responses


@app.get("/examples")
async def get_example_questions():
    """Возвращает примеры вопросов."""
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
    
    return {
        "examples": examples,
        "count": len(examples)
    }


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Запускает API сервер.
    
    Args:
        host: Хост для запуска
        port: Порт для запуска
    """
    print(f"🚀 Запуск API сервера на {host}:{port}")
    print(f"📖 Документация: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api.rag_api:app",
        host=host,
        port=port,
        reload=False
    )


if __name__ == "__main__":
    run_api_server()
