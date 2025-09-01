"""
RAG-бот с техниками промптинга для QuantumForge Software.

Реализует полный пайплайн RAG с Few-shot и Chain-of-Thought техниками.
"""
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


@dataclass
class RAGConfig:
    """Конфигурация RAG-бота."""
    # Векторная база
    vector_db_path: str = "./data/vector_index"
    collection_name: str = "quantumforge_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # LLM
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Поиск
    search_k: int = 5
    similarity_threshold: float = 0.7
    
    # Промпты
    system_prompt: str = ""
    few_shot_examples: List[Dict[str, str]] = None


@dataclass
class SearchResult:
    """Результат поиска."""
    query: str
    documents: List[Document]
    search_time: float
    total_results: int


@dataclass
class RAGResponse:
    """Ответ RAG-бота."""
    query: str
    answer: str
    reasoning: str
    sources: List[Dict[str, str]]
    confidence: float
    total_time: float


class RAGBot:
    """
    RAG-бот с техниками промптинга.
    
    Реализует полный пайплайн:
    1. Поиск релевантных документов
    2. Формирование промпта с Few-shot и CoT
    3. Генерация ответа с объяснением
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """
        Инициализация RAG-бота.
        
        Args:
            config: Конфигурация бота
        """
        self.config = config
        self.vectorstore = self._load_vectorstore()
        self.llm = self._load_llm()
        self.prompt_template = self._create_prompt_template()
        
        print(f"🤖 RAG-бот инициализирован")
        print(f"   Модель LLM: {config.llm_model}")
        print(f"   Векторная БД: {config.vector_db_path}")
        print(f"   Поиск: top-{config.search_k} документов")
    
    def _load_vectorstore(self) -> Chroma:
        """
        Загружает векторное хранилище.
        
        Returns:
            Chroma: Загруженное векторное хранилище
        """
        print(f"Загрузка векторного хранилища из {self.config.vector_db_path}...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=self.config.vector_db_path
        )
        
        return vectorstore
    
    def _load_llm(self) -> ChatOpenAI:
        """
        Загружает языковую модель.
        
        Returns:
            ChatOpenAI: Загруженная модель
        """
        print(f"Загрузка языковой модели {self.config.llm_model}...")
        
        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Создает шаблон промпта с Few-shot и Chain-of-Thought.
        
        Returns:
            ChatPromptTemplate: Шаблон промпта
        """
        # Системный промпт с Chain-of-Thought
        system_prompt = """Ты помощник QuantumForge Software, который отвечает на вопросы на основе предоставленной базы знаний.

ВАЖНО: Ты должен всегда объяснять свои шаги рассуждения перед тем, как дать ответ.

Правила:
1. Сначала проанализируй найденные документы
2. Объясни свои шаги рассуждения
3. Дай четкий и точный ответ
4. Если информации недостаточно, скажи "Я не знаю"
5. Используй только информацию из предоставленных документов
6. Отвечай на русском языке

Примеры рассуждения:
- "Сначала найду информацию о технологии X..."
- "В документе указано, что Y работает следующим образом..."
- "Следовательно, ответ на вопрос: Z"

Контекст: {context}

Примеры вопросов и ответов:
{few_shot_examples}

Вопрос: {question}

Рассуждение и ответ:"""

        # Few-shot примеры
        few_shot_examples = self._get_few_shot_examples()
        
        return ChatPromptTemplate.from_template(system_prompt)
    
    def _get_few_shot_examples(self) -> str:
        """
        Возвращает Few-shot примеры для промпта.
        
        Returns:
            str: Форматированные примеры
        """
        examples = [
            {
                "question": "Кто такой Kael Vexar?",
                "answer": "Kael Vexar - это Synth Flux-чувствительный Terran мужчина, легендарный Flux Knight Mentor, который сражался в Void Civil Conflict во время правления Galactic Void Empire. Он служил революционером на стороне Alliance вместе со своими спутниками Princess Lyra Organa и Commander Rex Caldor."
            },
            {
                "question": "Что такое Plasma Blade?",
                "answer": "Plasma Blade - это оружие, используемое Flux Knights. Это энергетический меч, который является жизнью воина. Anakin Vexar построил свой Plasma Blade, и Obi-Wan сказал ему, что это оружие - его жизнь. Plasma Blade используется в бою и является символом статуса Flux Knight."
            },
            {
                "question": "Что такое Synth Flux?",
                "answer": "Synth Flux - это энергия, которая пронизывает вселенную. Flux Knights чувствительны к Synth Flux и могут использовать его для различных способностей, включая Mind Control. Сила Synth Flux может быть измерена, и некоторые существа особенно сильны в его использовании."
            }
        ]
        
        formatted_examples = ""
        for example in examples:
            formatted_examples += f"Q: {example['question']}\n"
            formatted_examples += f"A: {example['answer']}\n\n"
        
        return formatted_examples
    
    def search_documents(self, query: str) -> SearchResult:
        """
        Ищет релевантные документы.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            SearchResult: Результат поиска
        """
        start_time = time.time()
        
        # Выполняем поиск
        documents = self.vectorstore.similarity_search(
            query, 
            k=self.config.search_k
        )
        
        search_time = time.time() - start_time
        
        return SearchResult(
            query=query,
            documents=documents,
            search_time=search_time,
            total_results=len(documents)
        )
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Форматирует найденные документы в контекст.
        
        Args:
            documents: Найденные документы
            
        Returns:
            str: Отформатированный контекст
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown")
            category = doc.metadata.get("category", "Unknown")
            
            context_parts.append(f"Документ {i} ({category}): {source}")
            context_parts.append(doc.page_content)
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Извлекает источники из документов.
        
        Args:
            documents: Найденные документы
            
        Returns:
            List[Dict[str, str]]: Список источников
        """
        sources = []
        
        for doc in documents:
            source = {
                "filename": doc.metadata.get("filename", "Unknown"),
                "category": doc.metadata.get("category", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, documents: List[Document], answer: str) -> float:
        """
        Вычисляет уверенность в ответе.
        
        Args:
            documents: Найденные документы
            answer: Сгенерированный ответ
            
        Returns:
            float: Уровень уверенности (0-1)
        """
        if not documents:
            return 0.0
        
        # Простая эвристика: больше документов = больше уверенности
        base_confidence = min(len(documents) / self.config.search_k, 1.0)
        
        # Если ответ содержит "не знаю", снижаем уверенность
        if "не знаю" in answer.lower() or "не могу" in answer.lower():
            base_confidence *= 0.3
        
        return round(base_confidence, 2)
    
    def generate_response(self, query: str, search_result: SearchResult) -> RAGResponse:
        """
        Генерирует ответ на основе найденных документов.
        
        Args:
            query: Пользовательский запрос
            search_result: Результат поиска
            
        Returns:
            RAGResponse: Сгенерированный ответ
        """
        start_time = time.time()
        
        # Форматируем контекст
        context = self._format_context(search_result.documents)
        
        # Формируем промпт
        prompt = self.prompt_template.format(
            context=context,
            few_shot_examples=self._get_few_shot_examples(),
            question=query
        )
        
        # Генерируем ответ
        response = self.llm.invoke(prompt)
        answer = response.content
        
        # Извлекаем рассуждение и финальный ответ
        reasoning, final_answer = self._extract_reasoning_and_answer(answer)
        
        # Вычисляем время и уверенность
        total_time = time.time() - start_time
        confidence = self._calculate_confidence(search_result.documents, final_answer)
        
        # Извлекаем источники
        sources = self._extract_sources(search_result.documents)
        
        return RAGResponse(
            query=query,
            answer=final_answer,
            reasoning=reasoning,
            sources=sources,
            confidence=confidence,
            total_time=total_time
        )
    
    def _extract_reasoning_and_answer(self, response: str) -> tuple[str, str]:
        """
        Извлекает рассуждение и финальный ответ из ответа модели.
        
        Args:
            response: Ответ модели
            
        Returns:
            tuple[str, str]: (рассуждение, финальный ответ)
        """
        # Простая эвристика для разделения рассуждения и ответа
        lines = response.split('\n')
        
        reasoning_lines = []
        answer_lines = []
        found_answer = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Ищем ключевые слова, указывающие на начало ответа
            if any(keyword in line.lower() for keyword in ['следовательно', 'итак', 'ответ:', 'вывод:', 'таким образом']):
                found_answer = True
            
            if found_answer:
                answer_lines.append(line)
            else:
                reasoning_lines.append(line)
        
        reasoning = '\n'.join(reasoning_lines) if reasoning_lines else "Рассуждение не найдено"
        answer = '\n'.join(answer_lines) if answer_lines else response
        
        return reasoning, answer
    
    def ask(self, query: str) -> RAGResponse:
        """
        Основной метод для получения ответа на вопрос.
        
        Args:
            query: Пользовательский вопрос
            
        Returns:
            RAGResponse: Полный ответ RAG-бота
        """
        print(f"🔍 Поиск документов для запроса: '{query}'")
        
        # 1. Ищем релевантные документы
        search_result = self.search_documents(query)
        
        print(f"   Найдено {search_result.total_results} документов за {search_result.search_time:.2f}с")
        
        # 2. Генерируем ответ
        print(f"🤖 Генерация ответа...")
        response = self.generate_response(query, search_result)
        
        print(f"   Ответ сгенерирован за {response.total_time:.2f}с")
        print(f"   Уверенность: {response.confidence * 100:.0f}%")
        
        return response
    
    def format_response(self, response: RAGResponse) -> str:
        """
        Форматирует ответ для вывода.
        
        Args:
            response: Ответ RAG-бота
            
        Returns:
            str: Отформатированный ответ
        """
        output = []
        
        # Заголовок
        output.append("🤖 ОТВЕТ RAG-БОТА")
        output.append("=" * 50)
        
        # Рассуждение
        output.append("🧠 РАССУЖДЕНИЕ:")
        output.append(response.reasoning)
        output.append()
        
        # Финальный ответ
        output.append("💡 ОТВЕТ:")
        output.append(response.answer)
        output.append()
        
        # Источники
        output.append("📚 ИСТОЧНИКИ:")
        for i, source in enumerate(response.sources, 1):
            output.append(f"  {i}. {source['filename']} ({source['category']})")
            output.append(f"     {source['content_preview']}")
            output.append()
        
        # Метаданные
        output.append("📊 МЕТАДАННЫЕ:")
        output.append(f"  Время поиска: {response.total_time:.2f}с")
        output.append(f"  Уверенность: {response.confidence * 100:.0f}%")
        output.append(f"  Источников: {len(response.sources)}")
        
        return "\n".join(output)
