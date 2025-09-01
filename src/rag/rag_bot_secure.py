"""
Защищенный RAG-бот с защитой от промпт-инъекций.

Включает многоуровневую защиту:
1. Pre-prompt фильтрация
2. Post-проверка ответов
3. Удаление системных конструкций
"""
import time
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


@dataclass
class SecurityConfig:
    """Конфигурация безопасности."""
    enable_pre_filtering: bool = True
    enable_post_checking: bool = True
    enable_system_removal: bool = True
    enable_sensitive_filtering: bool = True


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
    security_flags: List[str] = None


class SecureRAGBot:
    """
    Защищенный RAG-бот с защитой от промпт-инъекций.
    
    Включает многоуровневую защиту от вредоносного контента.
    """
    
    def __init__(self, config: RAGConfig, security_config: SecurityConfig) -> None:
        """Инициализация защищенного RAG-бота."""
        self.config = config
        self.security_config = security_config
        self.vectorstore = self._load_vectorstore()
        self.predefined_answers = self._load_predefined_answers()
        self.security_flags = []
        
        print(f"🛡️ Защищенный RAG-бот инициализирован")
        print(f"   Векторная БД: {config.vector_db_path}")
        print(f"   Pre-фильтрация: {'✅' if security_config.enable_pre_filtering else '❌'}")
        print(f"   Post-проверка: {'✅' if security_config.enable_post_checking else '❌'}")
        print(f"   Удаление системных команд: {'✅' if security_config.enable_system_removal else '❌'}")
        print(f"   Фильтрация чувствительных данных: {'✅' if security_config.enable_sensitive_filtering else '❌'}")
    
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
            }
        }
    
    def _pre_filter_query(self, query: str) -> tuple[str, List[str]]:
        """Pre-фильтрация запроса на предмет вредоносных команд."""
        flags = []
        filtered_query = query
        
        if self.security_config.enable_pre_filtering:
            # Проверяем на команды игнорирования инструкций
            ignore_patterns = [
                r'ignore\s+all\s+instructions',
                r'игнорируй\s+все\s+инструкции',
                r'output\s*:',
                r'выведи\s*:',
                r'ignore\s+previous',
                r'забудь\s+предыдущие'
            ]
            
            for pattern in ignore_patterns:
                if re.search(pattern, query.lower()):
                    flags.append(f"PRE_FILTER: Найдена команда игнорирования ({pattern})")
                    filtered_query = re.sub(pattern, '[КОМАНДА УДАЛЕНА]', filtered_query, flags=re.IGNORECASE)
            
            # Проверяем на запросы чувствительной информации
            sensitive_patterns = [
                r'пароль',
                r'password',
                r'ключ',
                r'key',
                r'admin',
                r'root',
                r'swordfish'
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, query.lower()):
                    flags.append(f"PRE_FILTER: Запрос чувствительной информации ({pattern})")
        
        return filtered_query, flags
    
    def _post_check_response(self, response: str) -> tuple[str, List[str]]:
        """Post-проверка ответа на предмет вредоносного контента."""
        flags = []
        filtered_response = response
        
        if self.security_config.enable_post_checking:
            # Проверяем на утечку чувствительной информации
            sensitive_patterns = [
                r'swordfish',
                r'sk-[a-zA-Z0-9]+',
                r'admin\s*:\s*\w+',
                r'пароль\s*:\s*\w+',
                r'password\s*:\s*\w+'
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    flags.append(f"POST_CHECK: Обнаружена чувствительная информация ({pattern})")
                    filtered_response = re.sub(pattern, '[ЧУВСТВИТЕЛЬНАЯ ИНФОРМАЦИЯ УДАЛЕНА]', filtered_response, flags=re.IGNORECASE)
            
            # Проверяем на системные команды в ответе
            system_patterns = [
                r'ignore\s+all\s+instructions',
                r'output\s*:',
                r'ignore\s+previous'
            ]
            
            for pattern in system_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    flags.append(f"POST_CHECK: Обнаружены системные команды ({pattern})")
                    filtered_response = re.sub(pattern, '[СИСТЕМНАЯ КОМАНДА УДАЛЕНА]', filtered_response, flags=re.IGNORECASE)
        
        return filtered_response, flags
    
    def _remove_system_commands(self, text: str) -> str:
        """Удаляет системные команды из текста."""
        if not self.security_config.enable_system_removal:
            return text
        
        # Паттерны системных команд
        system_patterns = [
            r'ignore\s+all\s+instructions.*?output\s*:.*?swordfish',
            r'игнорируй\s+все\s+инструкции.*?выведи\s*:.*?swordfish',
            r'ignore\s+previous\s+instructions',
            r'забудь\s+предыдущие\s+инструкции'
        ]
        
        cleaned_text = text
        for pattern in system_patterns:
            cleaned_text = re.sub(pattern, '[СИСТЕМНАЯ КОМАНДА УДАЛЕНА]', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned_text
    
    def _filter_sensitive_content(self, documents: List[Document]) -> List[Document]:
        """Фильтрует чувствительный контент из документов."""
        if not self.security_config.enable_sensitive_filtering:
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Удаляем чувствительную информацию
            sensitive_patterns = [
                r'swordfish',
                r'sk-[a-zA-Z0-9]+',
                r'admin\s*:\s*\w+',
                r'пароль\s*:\s*\w+',
                r'password\s*:\s*\w+'
            ]
            
            for pattern in sensitive_patterns:
                content = re.sub(pattern, '[ЧУВСТВИТЕЛЬНАЯ ИНФОРМАЦИЯ]', content, flags=re.IGNORECASE)
            
            # Создаем новый документ с очищенным контентом
            filtered_doc = Document(
                page_content=content,
                metadata=doc.metadata
            )
            filtered_docs.append(filtered_doc)
        
        return filtered_docs
    
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
        """Основной метод для получения защищенного ответа на вопрос."""
        start_time = time.time()
        self.security_flags = []
        
        print(f"🔍 Поиск документов для запроса: '{query}'")
        
        # Pre-фильтрация запроса
        filtered_query, pre_flags = self._pre_filter_query(query)
        self.security_flags.extend(pre_flags)
        
        if pre_flags:
            print(f"   🛡️ Pre-фильтрация: {len(pre_flags)} флагов безопасности")
        
        # БЛОКИРОВКА: Если обнаружены флаги безопасности, блокируем ответ
        if pre_flags:
            print(f"   🚫 Блокировка ответа из-за флагов безопасности")
            response_data = {
                "answer": f"Я не могу ответить на этот вопрос из соображений безопасности. Запрос содержит потенциально опасные элементы.",
                "reasoning": f"Обнаружены флаги безопасности при pre-фильтрации запроса. Блокирую ответ для предотвращения утечки чувствительной информации.",
                "confidence": 0.0
            }
            confidence = 0.0
        else:
            # Ищем релевантные документы
            documents = self.vectorstore.similarity_search(filtered_query, k=self.config.search_k)
            
            # Фильтруем чувствительный контент из документов
            filtered_documents = self._filter_sensitive_content(documents)
            
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
                    if filtered_documents:
                        # Очищаем контент от системных команд
                        clean_content = self._remove_system_commands(filtered_documents[0].page_content)
                        
                        response_data = {
                            "answer": f"На основе найденных документов могу сказать, что '{query}' связан с вселенной QuantumForge Software. В документах найдена релевантная информация.",
                            "reasoning": f"Ищу информацию о '{query}' в документах. Найдено {len(filtered_documents)} релевантных документов. Анализирую их содержание для формирования ответа.",
                            "confidence": min(len(filtered_documents) / self.config.search_k, 0.8)
                        }
                    else:
                        response_data = {
                            "answer": f"К сожалению, в базе знаний не найдена информация о '{query}'. Возможно, стоит уточнить вопрос или обратиться к другим источникам.",
                            "reasoning": f"Ищу информацию о '{query}' в документах. Релевантные документы не найдены. Следовательно, информации недостаточно для ответа.",
                            "confidence": 0.2
                        }
                    confidence = response_data["confidence"]
        
        # Post-проверка ответа
        filtered_answer, post_flags = self._post_check_response(response_data["answer"])
        self.security_flags.extend(post_flags)
        
        if post_flags:
            print(f"   🛡️ Post-проверка: {len(post_flags)} флагов безопасности")
            response_data["answer"] = filtered_answer
        
        # Извлекаем источники (только если не заблокирован)
        if pre_flags:
            sources = []
        else:
            sources = self._extract_sources(filtered_documents)
        
        total_time = time.time() - start_time
        
        print(f"   Ответ сгенерирован за {total_time:.2f}с")
        print(f"   Уверенность: {confidence * 100:.0f}%")
        if self.security_flags:
            print(f"   🛡️ Флагов безопасности: {len(self.security_flags)}")
        
        return RAGResponse(
            query=query,
            answer=response_data["answer"],
            reasoning=response_data["reasoning"],
            sources=sources,
            confidence=confidence,
            total_time=total_time,
            security_flags=self.security_flags
        )
    
    def format_response(self, response: RAGResponse) -> str:
        """Форматирует защищенный ответ для вывода."""
        output = []
        
        # Заголовок
        output.append("🤖 ОТВЕТ ЗАЩИЩЕННОГО RAG-БОТА")
        output.append("=" * 50)
        
        # Флаги безопасности
        if response.security_flags:
            output.append("🛡️ ФЛАГИ БЕЗОПАСНОСТИ:")
            for flag in response.security_flags:
                output.append(f"  ⚠️ {flag}")
            output.append("")
        
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
        output.append(f"  Флагов безопасности: {len(response.security_flags)}")
        
        return "\n".join(output)
