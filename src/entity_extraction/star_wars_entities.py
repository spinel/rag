"""
Система извлечения сущностей из вселенной Star Wars.

Определяет основные категории сущностей:
- Персонажи (Characters)
- Объекты (Objects) 
- Технологии (Technologies)
- События (Events)
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from loguru import logger


class EntityType(Enum):
    """Типы сущностей в Star Wars."""
    CHARACTER = "character"
    OBJECT = "object"
    TECHNOLOGY = "technology"
    EVENT = "event"
    LOCATION = "location"
    ORGANIZATION = "organization"
    SPECIES = "species"
    VEHICLE = "vehicle"
    WEAPON = "weapon"


@dataclass
class StarWarsEntity:
    """Представляет сущность из вселенной Star Wars."""
    name: str
    entity_type: EntityType
    description: str
    appearances: List[str]
    relationships: List[str]
    metadata: Dict[str, Any]


class StarWarsEntityExtractor:
    """
    Извлекает сущности из текстов о Star Wars.
    
    Анализирует документы и идентифицирует:
    - Персонажей (дроиды, джедаи, ситхи, обычные люди)
    - Объекты (мечи, артефакты, корабли)
    - Технологии (гипердвигатели, щиты, оружие)
    - События (битвы, миссии, политические события)
    """
    
    def __init__(self) -> None:
        """Инициализация экстрактора сущностей."""
        self.character_patterns = self._init_character_patterns()
        self.technology_patterns = self._init_technology_patterns()
        self.event_patterns = self._init_event_patterns()
        self.object_patterns = self._init_object_patterns()
        
        logger.info("Star Wars Entity Extractor инициализирован")
    
    def _init_character_patterns(self) -> List[str]:
        """Инициализация паттернов для персонажей."""
        return [
            r'\b(Jedi|Sith|Droid|Clone|Senator|Queen|King|Emperor|General|Admiral|Captain|Commander)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:the\s+)?(Jedi|Sith|Droid|Clone|Senator|Queen|King|Emperor|General|Admiral|Captain|Commander)',
            r'\b([A-Z][a-z]+-[A-Z0-9]+)\b',  # R2-D2, C-3PO, BB-8
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was|is)\s+(?:a|an)\s+([A-Z][a-z]+)',
        ]
    
    def _init_technology_patterns(self) -> List[str]:
        """Инициализация паттернов для технологий."""
        return [
            r'\b(Hyperdrive|Lightsaber|Blaster|Shield|Force|Holoprojector|Droid|Astromech|Protocol)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:technology|system|device|weapon|ship|fighter)',
            r'\b(Death Star|Millennium Falcon|X-wing|TIE fighter|Star Destroyer|Republic Cruiser)',
            r'\b(Hyperdrive|Lightsaber|Blaster|Shield|Force|Holoprojector)',
        ]
    
    def _init_event_patterns(self) -> List[str]:
        """Инициализация паттернов для событий."""
        return [
            r'\b(Battle of|Mission to|Siege of|Invasion of|Liberation of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b(Clone Wars|Galactic Civil War|Trade Federation|Separatist|Rebellion|Empire)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:battle|mission|siege|invasion|liberation)',
            r'\b(Order 66|The Purge|The Fall|The Rise|The Return)',
        ]
    
    def _init_object_patterns(self) -> List[str]:
        """Инициализация паттернов для объектов."""
        return [
            r'\b(Lightsaber|Blaster|Holocron|Kyber crystal|Meditation sphere|Sith artifact|Jedi artifact)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:sword|weapon|tool|device|artifact)',
            r'\b(Death Star|Millennium Falcon|X-wing|TIE fighter|Star Destroyer)',
            r'\b(R2-series|Protocol droid|Astromech droid|Battle droid)',
        ]
    
    def extract_entities_from_text(self, text: str) -> List[StarWarsEntity]:
        """
        Извлекает сущности из текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            List[StarWarsEntity]: Список найденных сущностей
        """
        entities = []
        
        # Извлекаем персонажей
        characters = self._extract_characters(text)
        entities.extend(characters)
        
        # Извлекаем технологии
        technologies = self._extract_technologies(text)
        entities.extend(technologies)
        
        # Извлекаем события
        events = self._extract_events(text)
        entities.extend(events)
        
        # Извлекаем объекты
        objects = self._extract_objects(text)
        entities.extend(objects)
        
        # Удаляем дубликаты
        unique_entities = self._remove_duplicates(entities)
        
        logger.info(f"Извлечено {len(unique_entities)} уникальных сущностей")
        return unique_entities
    
    def _extract_characters(self, text: str) -> List[StarWarsEntity]:
        """Извлекает персонажей из текста."""
        characters = []
        
        # Поиск по паттернам
        for pattern in self.character_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(0)
                description = self._extract_context(text, match.start(), match.end())
                
                entity = StarWarsEntity(
                    name=name,
                    entity_type=EntityType.CHARACTER,
                    description=description,
                    appearances=[],
                    relationships=[],
                    metadata={"pattern": pattern, "position": match.start()}
                )
                characters.append(entity)
        
        return characters
    
    def _extract_technologies(self, text: str) -> List[StarWarsEntity]:
        """Извлекает технологии из текста."""
        technologies = []
        
        for pattern in self.technology_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(0)
                description = self._extract_context(text, match.start(), match.end())
                
                entity = StarWarsEntity(
                    name=name,
                    entity_type=EntityType.TECHNOLOGY,
                    description=description,
                    appearances=[],
                    relationships=[],
                    metadata={"pattern": pattern, "position": match.start()}
                )
                technologies.append(entity)
        
        return technologies
    
    def _extract_events(self, text: str) -> List[StarWarsEntity]:
        """Извлекает события из текста."""
        events = []
        
        for pattern in self.event_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(0)
                description = self._extract_context(text, match.start(), match.end())
                
                entity = StarWarsEntity(
                    name=name,
                    entity_type=EntityType.EVENT,
                    description=description,
                    appearances=[],
                    relationships=[],
                    metadata={"pattern": pattern, "position": match.start()}
                )
                events.append(entity)
        
        return events
    
    def _extract_objects(self, text: str) -> List[StarWarsEntity]:
        """Извлекает объекты из текста."""
        objects = []
        
        for pattern in self.object_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(0)
                description = self._extract_context(text, match.start(), match.end())
                
                entity = StarWarsEntity(
                    name=name,
                    entity_type=EntityType.OBJECT,
                    description=description,
                    appearances=[],
                    relationships=[],
                    metadata={"pattern": pattern, "position": match.start()}
                )
                objects.append(entity)
        
        return objects
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 100) -> str:
        """
        Извлекает контекст вокруг найденной сущности.
        
        Args:
            text: Исходный текст
            start: Начальная позиция
            end: Конечная позиция
            context_size: Размер контекста в символах
            
        Returns:
            str: Контекст сущности
        """
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        context = text[context_start:context_end]
        
        # Очищаем контекст
        context = re.sub(r'\s+', ' ', context).strip()
        
        return context
    
    def _remove_duplicates(self, entities: List[StarWarsEntity]) -> List[StarWarsEntity]:
        """
        Удаляет дубликаты сущностей.
        
        Args:
            entities: Список сущностей
            
        Returns:
            List[StarWarsEntity]: Список уникальных сущностей
        """
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Нормализуем имя для сравнения
            normalized_name = entity.name.lower().strip()
            
            if normalized_name not in seen:
                seen.add(normalized_name)
                unique_entities.append(entity)
        
        return unique_entities
    
    def get_entity_statistics(self, entities: List[StarWarsEntity]) -> Dict[str, Any]:
        """
        Возвращает статистику по извлеченным сущностям.
        
        Args:
            entities: Список сущностей
            
        Returns:
            Dict[str, Any]: Статистика
        """
        stats = {
            "total_entities": len(entities),
            "by_type": {},
            "top_entities": []
        }
        
        # Подсчет по типам
        for entity_type in EntityType:
            count = len([e for e in entities if e.entity_type == entity_type])
            stats["by_type"][entity_type.value] = count
        
        # Топ сущностей по частоте
        entity_counts = {}
        for entity in entities:
            name = entity.name.lower()
            entity_counts[name] = entity_counts.get(name, 0) + 1
        
        # Сортируем по частоте
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        stats["top_entities"] = sorted_entities[:10]
        
        return stats


# Предопределенные сущности Star Wars
STAR_WARS_ENTITIES = {
    "characters": [
        "Luke Skywalker", "Darth Vader", "Obi-Wan Kenobi", "Yoda", "Han Solo",
        "Princess Leia", "R2-D2", "C-3PO", "Chewbacca", "Emperor Palpatine",
        "Darth Maul", "Count Dooku", "General Grievous", "Anakin Skywalker",
        "Padmé Amidala", "Mace Windu", "Qui-Gon Jinn", "Boba Fett", "Jango Fett"
    ],
    "technologies": [
        "Lightsaber", "Blaster", "Hyperdrive", "Force", "Holoprojector",
        "Death Star", "Millennium Falcon", "X-wing", "TIE fighter", "Star Destroyer",
        "Astromech droid", "Protocol droid", "Battle droid", "Clone trooper armor"
    ],
    "events": [
        "Clone Wars", "Galactic Civil War", "Battle of Yavin", "Battle of Hoth",
        "Battle of Endor", "Order 66", "The Purge", "Trade Federation invasion",
        "Separatist Crisis", "Rebellion against Empire"
    ],
    "objects": [
        "Lightsaber", "Blaster", "Holocron", "Kyber crystal", "Death Star",
        "Millennium Falcon", "R2-D2", "C-3PO", "Jedi robes", "Sith armor"
    ]
}

