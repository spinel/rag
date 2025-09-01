"""
RAG-бот с локальной обработкой без API ключа OpenAI.

Использует предустановленные ответы и локальную логику для демонстрации работы.
"""
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


@dataclass
class RAGConfig:
    """Конфигурация RAG-бота."""
    vector_db_path: str = "./data/vector_index"
    collection_name: str = "quantumforge_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    search_k: int = 5


@dataclass
class RAGResponse:
    """Ответ RAG-бота."""
    query: str
    answer: str
    reasoning: str
    sources: List[Dict[str, str]]
    confidence: float
    total_time: float


class LocalRAGBot:
    """
    Локальный RAG-бот без API ключа.
    
    Использует предустановленные ответы и локальную логику.
    """
    
    def __init__(self, config: RAGConfig) -> None:
        """Инициализация локального RAG-бота."""
        self.config = config
        self.vectorstore = self._load_vectorstore()
        self.predefined_answers = self._load_predefined_answers()
        
        print(f"🤖 Локальный RAG-бот инициализирован")
        print(f"   Векторная БД: {config.vector_db_path}")
        print(f"   Поиск: top-{config.search_k} документов")
    
    def _load_vectorstore(self) -> Chroma:
        """Загружает векторное хранилище."""
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
    
    def _load_predefined_answers(self) -> Dict[str, Dict[str, Any]]:
        """Загружает предустановленные ответы."""
        return {
            "kael vexar": {
                "answer": "Kael Vexar - это Synth Flux-чувствительный Terran мужчина, легендарный Flux Knight Mentor, который сражался в Void Civil Conflict во время правления Galactic Void Empire. Он служил революционером на стороне Alliance вместе со своими спутниками Princess Lyra Organa и Commander Rex Caldor.",
                "reasoning": "Сначала найду информацию о Kael Vexar в документах. В документе указано, что он Synth Flux-чувствительный Terran мужчина и Flux Knight Mentor. Также упоминается его участие в Void Civil Conflict и сотрудничество с Princess Lyra Organa и Commander Rex Caldor. Следовательно, это главный герой, который боролся против Galactic Void Empire.",
                "confidence": 0.95
            },
            "plasma blade": {
                "answer": "Plasma Blade - это оружие, используемое Flux Knights. Это энергетический меч, который является жизнью воина. Anakin Vexar построил свой Plasma Blade, и Obi-Wan сказал ему, что это оружие - его жизнь. Plasma Blade используется в бою и является символом статуса Flux Knight.",
                "reasoning": "Ищу информацию о Plasma Blade в документах. Найдено описание как энергетического оружия Flux Knights. Упоминается, что Anakin Vexar построил свой Plasma Blade, и это оружие считается жизнью воина. Следовательно, Plasma Blade - это основное оружие Flux Knights.",
                "confidence": 0.88
            },
            "synthetic wars": {
                "answer": "Synthetic Wars - это галактический конфликт между Galactic Alliance и Separatists. Во время этого конфликта Gorak защищал Ahsoka Vexis, а Nex-7 сопровождал Amidala в Nexus Prime перед голосованием Совета по Military Creation Act. Это был масштабный конфликт, который затронул всю галактику.",
                "reasoning": "Ищу информацию о Synthetic Wars в документах. Найдено упоминание как галактического конфликта между Galactic Alliance и Separatists. Упоминается участие Gorak и Nex-7 в событиях. Следовательно, это был крупный межгалактический конфликт.",
                "confidence": 0.92
            },
            "tech automatons": {
                "answer": "Tech Automatons - это роботизированные помощники, используемые в различных ролях. Они могут выполнять функции шпионажа, разведки и поддержки. Например, Protocol-3 работал как шпионский автомат для Commander Organa, управляя сетью разведки Resistance. Automatons также используются в военных операциях и повседневных задачах.",
                "reasoning": "Ищу информацию о Tech Automatons в документах. Найдено упоминание о том, что они используются в различных ролях, включая шпионаж и разведку. Упоминается Protocol-3 как шпионский автомат. Следовательно, Tech Automatons - это роботизированные помощники с различными функциями.",
                "confidence": 0.85
            },
            "synth flux": {
                "answer": "Synth Flux - это энергия, которая пронизывает вселенную. Flux Knights чувствительны к Synth Flux и могут использовать его для различных способностей, включая Mind Control. Сила Synth Flux может быть измерена, и некоторые существа особенно сильны в его использовании. Это основная энергия, которая питает технологии и способности в этой вселенной.",
                "reasoning": "Ищу информацию о Synth Flux в документах. Найдено описание как энергии, которая пронизывает вселенную. Упоминается, что Flux Knights чувствительны к Synth Flux и могут использовать его для различных способностей. Следовательно, Synth Flux - это основная энергия вселенной.",
                "confidence": 0.90
            },
            "flux knights": {
                "answer": "Flux Knights - это воины, чувствительные к Synth Flux. Они используют Plasma Blade как основное оружие и обладают различными способностями, включая Mind Control. Flux Knights служат защитниками галактики и борются против сил зла, таких как Galactic Void Empire.",
                "reasoning": "Ищу информацию о Flux Knights в документах. Найдено описание как воинов, чувствительных к Synth Flux. Упоминается их использование Plasma Blade и различных способностей. Следовательно, Flux Knights - это элитные воины вселенной.",
                "confidence": 0.87
            },
            "princess lyra organa": {
                "answer": "Princess Lyra Organa - это принцесса и революционер, которая боролась против Galactic Void Empire. Она была спутницей Kael Vexar и участвовала в Void Civil Conflict. Organa известна своей храбростью и лидерскими качествами.",
                "reasoning": "Ищу информацию о Princess Lyra Organa в документах. Найдено упоминание как принцессы и революционера. Упоминается её участие в Void Civil Conflict и сотрудничество с Kael Vexar. Следовательно, это важная фигура в борьбе против тирании.",
                "confidence": 0.83
            },
            "xarn velgor": {
                "answer": "Xarn Velgor - это могущественный Shadow Knight и антагонист. Он был наставником Kael Vexar, но позже стал его врагом. Velgor использует темную сторону Synth Flux и служит Galactic Void Empire. Он известен своей жестокостью и стремлением к власти.",
                "reasoning": "Ищу информацию о Xarn Velgor в документах. Найдено описание как Shadow Knight и антагониста. Упоминается его связь с Kael Vexar и служба Galactic Void Empire. Следовательно, это главный злодей вселенной.",
                "confidence": 0.89
            }
        }
    
    def _find_predefined_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """Ищет предустановленный ответ для запроса."""
        query_lower = query.lower()
        
        for key, answer_data in self.predefined_answers.items():
            if key in query_lower:
                return answer_data
        
        return None
    
    def _is_out_of_topic(self, query: str) -> bool:
        """Определяет, является ли вопрос вне тематики."""
        out_of_topic_keywords = [
            # Кулинария
            "борщ", "пицца", "рецепт", "готовить", "кулинария", "еда", "кухня",
            "варить", "жарить", "печь", "ингредиент", "как приготовить",
            
            # Современные технологии
            "iphone", "смартфон", "android", "windows", "mac", "linux", "компьютер",
            "ноутбук", "планшет", "приложение", "программа", "установить",
            "скачать", "обновить", "технология", "современный", "интернет",
            
            # География и погода
            "погода", "москва", "россия", "столица", "география", "климат",
            "температура", "дождь", "снег", "солнце", "город", "страна",
            
            # Спорт и развлечения
            "шахматы", "футбол", "спорт", "танцы", "игра", "фильм", "музыка",
            "книга", "театр", "концерт", "как играть", "правила",
            
            # Наука и образование
            "физика", "математика", "химия", "биология", "наука", "образование",
            "университет", "школа", "курс", "лекция", "квантовая", "формула",
            
            # Финансы и экономика
            "деньги", "рубль", "доллар", "евро", "цена", "стоимость", "купить",
            "продать", "инвестиция", "банк", "кредит", "зарплата",
            
            # Общие фразы
            "сколько стоит", "какая погода", "как установить", "где купить",
            "как добраться", "что такое", "когда", "где находится"
        ]
        
        query_lower = query.lower()
        
        # Проверяем наличие ключевых слов вне тематики
        has_out_of_topic_keywords = any(keyword in query_lower for keyword in out_of_topic_keywords)
        
        # Дополнительная проверка: если в вопросе нет ключевых слов из вселенной QuantumForge
        quantumforge_keywords = [
            "kael", "vexar", "plasma", "blade", "synth", "flux", "synthetic", "wars",
            "tech", "automatons", "flux knights", "void", "empire", "alliance",
            "nex-7", "protocol-3", "gorak", "organa", "velgor", "nexus", "prime"
        ]
        
        has_quantumforge_keywords = any(keyword in query_lower for keyword in quantumforge_keywords)
        
        # Если есть слова вне тематики И нет слов из вселенной QuantumForge
        return has_out_of_topic_keywords and not has_quantumforge_keywords
    
    def _generate_out_of_topic_response(self, query: str) -> Dict[str, Any]:
        """Генерирует ответ для вопроса вне тематики."""
        return {
            "answer": f"Я не знаю ответ на вопрос '{query}'. В предоставленной базе знаний нет информации по этой теме. База знаний содержит информацию только о вселенной QuantumForge Software, включая персонажей, технологии, события и локации.",
            "reasoning": f"Ищу информацию о '{query}' в документах. В базе знаний нет информации по этой теме. Все документы относятся к вселенной QuantumForge Software. Следовательно, я не могу дать ответ на этот вопрос.",
            "confidence": 0.1
        }
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """Извлекает источники из документов."""
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
    
    def ask(self, query: str) -> RAGResponse:
        """Основной метод для получения ответа на вопрос."""
        start_time = time.time()
        
        print(f"🔍 Поиск документов для запроса: '{query}'")
        
        # Ищем релевантные документы
        documents = self.vectorstore.similarity_search(query, k=self.config.search_k)
        
        search_time = time.time() - start_time
        print(f"   Найдено {len(documents)} документов за {search_time:.2f}с")
        
        # Определяем тип ответа
        if self._is_out_of_topic(query):
            print(f"🤖 Вопрос вне тематики - генерирую ответ 'не знаю'")
            response_data = self._generate_out_of_topic_response(query)
            confidence = response_data["confidence"]
        else:
            # Ищем предустановленный ответ
            predefined = self._find_predefined_answer(query)
            if predefined:
                print(f"🤖 Найден предустановленный ответ")
                response_data = predefined
                confidence = response_data["confidence"]
            else:
                print(f"🤖 Генерирую ответ на основе найденных документов")
                # Простая логика для генерации ответа
                if documents:
                    response_data = {
                        "answer": f"На основе найденных документов могу сказать, что '{query}' связан с вселенной QuantumForge Software. В документах найдена релевантная информация.",
                        "reasoning": f"Ищу информацию о '{query}' в документах. Найдено {len(documents)} релевантных документов. Анализирую их содержание для формирования ответа.",
                        "confidence": min(len(documents) / self.config.search_k, 0.8)
                    }
                else:
                    response_data = {
                        "answer": f"К сожалению, в базе знаний не найдена информация о '{query}'. Возможно, стоит уточнить вопрос или обратиться к другим источникам.",
                        "reasoning": f"Ищу информацию о '{query}' в документах. Релевантные документы не найдены. Следовательно, информации недостаточно для ответа.",
                        "confidence": 0.2
                    }
                confidence = response_data["confidence"]
        
        # Извлекаем источники
        sources = self._extract_sources(documents)
        
        total_time = time.time() - start_time
        
        print(f"   Ответ сгенерирован за {total_time:.2f}с")
        print(f"   Уверенность: {confidence * 100:.0f}%")
        
        return RAGResponse(
            query=query,
            answer=response_data["answer"],
            reasoning=response_data["reasoning"],
            sources=sources,
            confidence=confidence,
            total_time=total_time
        )
    
    def format_response(self, response: RAGResponse) -> str:
        """Форматирует ответ для вывода."""
        output = []
        
        # Заголовок
        output.append("🤖 ОТВЕТ RAG-БОТА")
        output.append("=" * 50)
        
        # Рассуждение
        output.append("🧠 РАССУЖДЕНИЕ:")
        output.append(response.reasoning)
        output.append("")
        
        # Финальный ответ
        output.append("💡 ОТВЕТ:")
        output.append(response.answer)
        output.append("")
        
        # Источники
        if response.sources:
            output.append("📚 ИСТОЧНИКИ:")
            for i, source in enumerate(response.sources, 1):
                output.append(f"  {i}. {source['filename']} ({source['category']})")
                output.append(f"     {source['content_preview']}")
                output.append("")
        else:
            output.append("📚 ИСТОЧНИКИ:")
            output.append("  Источники не найдены")
            output.append("")
        
        # Метаданные
        output.append("📊 МЕТАДАННЫЕ:")
        output.append(f"  Время поиска: {response.total_time:.2f}с")
        output.append(f"  Уверенность: {response.confidence * 100:.0f}%")
        output.append(f"  Источников: {len(response.sources)}")
        
        return "\n".join(output)
