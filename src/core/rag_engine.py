"""
Основной движок RAG для обработки запросов и генерации ответов.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ..models.embedding_model import EmbeddingModel
from ..models.llm_model import LLMModel
from ..vectorstore.vector_store import VectorStore
from ..document_processing.document_processor import DocumentProcessor


class RAGEngine:
    """
    Основной движок RAG для обработки запросов пользователей.
    
    Объединяет компоненты для поиска релевантных документов
    и генерации ответов на основе найденной информации.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_model: LLMModel,
        vector_store: VectorStore,
        document_processor: DocumentProcessor,
    ) -> None:
        """
        Инициализация RAG-движка.
        
        Args:
            embedding_model: Модель для создания эмбеддингов
            llm_model: Языковая модель для генерации ответов
            vector_store: Векторное хранилище для поиска документов
            document_processor: Процессор для обработки документов
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store = vector_store
        self.document_processor = document_processor
        
        # Создаем цепочку RAG
        self.qa_chain = self._create_qa_chain()
        
        logger.info("RAG-движок инициализирован")
    
    def _create_qa_chain(self) -> RetrievalQA:
        """
        Создает цепочку для вопросов и ответов.
        
        Returns:
            RetrievalQA: Цепочка для обработки запросов
        """
        # Промпт-шаблон для контекста QuantumForge Software
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Ты - интеллектуальный помощник компании QuantumForge Software, 
            специализирующейся на SaaS-платформе для моделирования промышленных объектов "Digital Twin".
            
            Используй предоставленный контекст для ответа на вопрос. 
            Если в контексте нет информации для ответа, скажи об этом честно.
            
            Контекст: {context}
            
            Вопрос: {question}
            
            Ответ:"""
        )
        
        # Создаем цепочку
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_model.get_llm(),
            chain_type="stuff",
            retriever=self.vector_store.get_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True,
        )
        
        return qa_chain
    
    def process_query(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя и возвращает ответ.
        
        Args:
            query: Вопрос пользователя
            user_id: Идентификатор пользователя (для логирования)
            top_k: Количество релевантных документов для поиска
            
        Returns:
            Dict[str, Any]: Ответ с информацией о найденных документах
        """
        try:
            logger.info(f"Обработка запроса от пользователя {user_id}: {query}")
            
            # Выполняем поиск и генерацию ответа
            result = self.qa_chain({"query": query})
            
            # Извлекаем информацию из результата
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            
            # Формируем метаданные о найденных документах
            sources = []
            for doc in source_documents:
                if hasattr(doc, 'metadata'):
                    sources.append({
                        "title": doc.metadata.get("title", "Неизвестный документ"),
                        "source": doc.metadata.get("source", "Неизвестный источник"),
                        "page": doc.metadata.get("page", None),
                        "chunk_id": doc.metadata.get("chunk_id", None),
                    })
            
            response = {
                "answer": answer,
                "sources": sources,
                "query": query,
                "user_id": user_id,
                "confidence": self._calculate_confidence(answer, sources),
            }
            
            logger.info(f"Ответ сгенерирован для пользователя {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            return {
                "answer": "Извините, произошла ошибка при обработке вашего запроса. Попробуйте переформулировать вопрос.",
                "sources": [],
                "query": query,
                "user_id": user_id,
                "error": str(e),
            }
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Добавляет новые документы в векторное хранилище.
        
        Args:
            documents: Список документов для добавления
            
        Returns:
            bool: True если документы успешно добавлены
        """
        try:
            logger.info(f"Добавление {len(documents)} документов в хранилище")
            
            # Обрабатываем документы
            processed_docs = self.document_processor.process_documents(documents)
            
            # Добавляем в векторное хранилище
            success = self.vector_store.add_documents(processed_docs)
            
            if success:
                logger.info("Документы успешно добавлены в хранилище")
            else:
                logger.error("Ошибка при добавлении документов")
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            return False
    
    def search_similar_documents(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Document]:
        """
        Ищет похожие документы без генерации ответа.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            List[Document]: Список найденных документов
        """
        try:
            logger.info(f"Поиск похожих документов для запроса: {query}")
            
            # Получаем ретривер и выполняем поиск
            retriever = self.vector_store.get_retriever()
            retriever.search_kwargs["k"] = top_k
            
            documents = retriever.get_relevant_documents(query)
            
            logger.info(f"Найдено {len(documents)} документов")
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            return []
    
    def _calculate_confidence(self, answer: str, sources: List[Dict]) -> float:
        """
        Вычисляет уверенность в ответе на основе источников.
        
        Args:
            answer: Сгенерированный ответ
            sources: Список источников
            
        Returns:
            float: Уровень уверенности (0.0 - 1.0)
        """
        if not sources:
            return 0.0
        
        # Простая эвристика: больше источников = выше уверенность
        base_confidence = min(len(sources) * 0.2, 1.0)
        
        # Дополнительные факторы
        if "не знаю" in answer.lower() or "нет информации" in answer.lower():
            base_confidence *= 0.5
        
        return round(base_confidence, 2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования RAG-движка.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        try:
            stats = {
                "total_documents": self.vector_store.get_document_count(),
                "embedding_model": self.embedding_model.model_name,
                "llm_model": self.llm_model.model_name,
                "vector_store_type": type(self.vector_store).__name__,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            return {}
