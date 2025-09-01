"""
Упрощенная система замены терминов Star Wars на вымышленные аналоги.

Создает словарь соответствий и заменяет все упоминания
терминов в текстах, сохраняя логичность и читаемость.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class TermsMapperSimple:
    """
    Упрощенный маппер для замены терминов Star Wars на вымышленные аналоги.
    
    Создает уникальную базу знаний, где все термины Star Wars
    заменены на оригинальные вымышленные названия.
    """
    
    def __init__(self) -> None:
        """Инициализация маппера терминов."""
        self.terms_mapping = self._create_terms_mapping()
        self.reverse_mapping = {v: k for k, v in self.terms_mapping.items()}
        
        print(f"Создан словарь соответствий с {len(self.terms_mapping)} терминами")
    
    def _create_terms_mapping(self) -> Dict[str, str]:
        """
        Создает словарь соответствий терминов.
        
        Returns:
            Dict[str, str]: Словарь {оригинальный_термин: вымышленный_термин}
        """
        return {
            # Персонажи
            "Luke Skywalker": "Kael Vexar",
            "Darth Vader": "Xarn Velgor",
            "Obi-Wan Kenobi": "Toren Vexis",
            "Yoda": "Master Zephyr",
            "Han Solo": "Rex Caldor",
            "Princess Leia": "Princess Lyra",
            "R2-D2": "Nex-7",
            "C-3PO": "Protocol-3",
            "Chewbacca": "Gorak",
            "Emperor Palpatine": "Emperor Malakar",
            "Darth Maul": "Darth Krayt",
            "Count Dooku": "Count Vexus",
            "General Grievous": "General Vorak",
            "Anakin Skywalker": "Anakin Vexar",
            "Padmé Amidala": "Senator Lyra",
            "Mace Windu": "Mace Vindor",
            "Qui-Gon Jinn": "Qui-Gon Vexis",
            "Boba Fett": "Boba Krayt",
            "Jango Fett": "Jango Krayt",
            "Ahsoka Tano": "Ahsoka Vexis",
            "Captain Rex": "Captain Vex",
            "Bail Organa": "Bail Vexar",
            "Darth Sidious": "Darth Malakar",
            "Darth Tyranus": "Darth Vexus",
            
            # Технологии и устройства
            "Lightsaber": "Plasma Blade",
            "Blaster": "Pulse Rifle",
            "Hyperdrive": "Quantum Drive",
            "Force": "Synth Flux",
            "Holoprojector": "Holo Display",
            "Death Star": "Void Core",
            "Millennium Falcon": "Stellar Phoenix",
            "X-wing": "X-fighter",
            "TIE fighter": "Shadow fighter",
            "Star Destroyer": "Void Cruiser",
            "Republic Cruiser": "Alliance Cruiser",
            "Astromech droid": "Tech Droid",
            "Protocol droid": "Protocol Unit",
            "Battle droid": "Combat Unit",
            "Clone trooper": "Elite Guard",
            "Clone trooper armor": "Guard Armor",
            "Droid": "Automaton",
            "R2-series": "Nex-series",
            "Protocol": "Protocol Unit",
            "Astromech": "Tech Unit",
            
            # События и организации
            "Clone Wars": "Synthetic Wars",
            "Galactic Civil War": "Void Civil War",
            "Battle of Yavin": "Battle of Yavin Prime",
            "Battle of Hoth": "Battle of Ice Peak",
            "Battle of Endor": "Battle of Forest Moon",
            "Order 66": "Protocol 66",
            "The Purge": "The Cleansing",
            "Trade Federation": "Trade Alliance",
            "Separatist": "Rebel",
            "Rebellion": "Resistance",
            "Empire": "Void Empire",
            "Republic": "Alliance",
            "Jedi Order": "Flux Order",
            "Sith Order": "Shadow Order",
            "Jedi Council": "Flux Council",
            "Senate": "Council",
            "Rebel Alliance": "Resistance Alliance",
            "Imperial": "Void",
            
            # Локации и планеты
            "Coruscant": "Nexus Prime",
            "Tatooine": "Desert World",
            "Naboo": "Aqua World",
            "Hoth": "Ice World",
            "Endor": "Forest Moon",
            "Yavin": "Yavin Prime",
            "Dagobah": "Swamp World",
            "Bespin": "Cloud City",
            "Alderaan": "Peace World",
            "Kamino": "Clone World",
            "Geonosis": "Bug World",
            "Mustafar": "Lava World",
            "Utapau": "Sinkhole World",
            "Kashyyyk": "Tree World",
            "Ryloth": "Twi'lek World",
            "Mandalore": "Warrior World",
            "Christophsis": "Crystal World",
            "Cato Neimoidia": "Banking World",
            "Scipio": "Financial World",
            "Garel": "Industrial World",
            "Lothal": "Frontier World",
            "Atollon": "Desert Base",
            "D'Qar": "Resistance Base",
            "Starkiller Base": "Void Base",
            "Exegol": "Shadow World",
            "Ahch-To": "Ancient World",
            "Jakku": "Scavenger World",
            "Takodana": "Pirate World",
            "Crait": "Salt World",
            "Kijimi": "Snow World",
            "Pasaana": "Desert Festival",
            "Kef Bir": "Ocean Moon",
            "Ajan Kloss": "Jungle Base",
            "Ruusan": "Ancient Battlefield",
            
            # Объекты и артефакты
            "Holocron": "Memory Crystal",
            "Kyber crystal": "Flux Crystal",
            "Meditation sphere": "Focus Sphere",
            "Sith artifact": "Shadow Artifact",
            "Jedi artifact": "Flux Artifact",
            "Jedi robes": "Flux Robes",
            "Sith armor": "Shadow Armor",
            "Clone armor": "Guard Armor",
            "Stormtrooper armor": "Void Armor",
            "Rebel armor": "Resistance Armor",
            
            # Виды и расы
            "Human": "Terran",
            "Wookiee": "Gorak",
            "Twi'lek": "Lekku",
            "Clone": "Synthetic",
            "Jedi": "Flux Knight",
            "Sith": "Shadow Knight",
            "Padawan": "Apprentice",
            "Master": "Mentor",
            "Knight": "Warrior",
            "Senator": "Councilor",
            "Queen": "Ruler",
            "King": "Monarch",
            "Emperor": "Void Lord",
            "General": "Commander",
            "Admiral": "Fleet Commander",
            "Captain": "Ship Commander",
            "Commander": "Unit Leader",
            
            # Технические термины
            "Hyperspace": "Quantum Space",
            "Lightspeed": "Quantum Speed",
            "Shield": "Deflector",
            "Deflector shield": "Deflector Field",
            "Blaster bolt": "Pulse Bolt",
            "Laser": "Pulse Beam",
            "Turbolaser": "Heavy Pulse",
            "Ion cannon": "Ion Blaster",
            "Proton torpedo": "Proton Missile",
            "Concussion missile": "Concussion Rocket",
            "Thermal detonator": "Thermal Bomb",
            "Stun baton": "Stun Rod",
            "Vibroblade": "Vibration Blade",
            "Electrostaff": "Electric Staff",
            "Force lightning": "Flux Lightning",
            "Force push": "Flux Push",
            "Force pull": "Flux Pull",
            "Force choke": "Flux Choke",
            "Mind trick": "Mind Control",
            "Force sense": "Flux Sense",
            "Force vision": "Flux Vision",
            "Force ghost": "Flux Spirit",
            "Force bond": "Flux Bond",
            "Force sensitive": "Flux Sensitive",
            "Dark side": "Shadow Path",
            "Light side": "Flux Path",
            "Balance": "Harmony",
            "Unlimited power": "Infinite Power",
            "Power of the Force": "Power of the Flux",
            
            # Организации и титулы
            "Jedi Temple": "Flux Temple",
            "Sith Temple": "Shadow Temple",
            "Jedi Academy": "Flux Academy",
            "Sith Academy": "Shadow Academy",
            "Jedi Archives": "Flux Archives",
            "Sith Archives": "Shadow Archives",
            "Jedi Code": "Flux Code",
            "Sith Code": "Shadow Code",
            "Sith Council": "Shadow Council",
            "High Council": "Supreme Council",
            "Council of Reassignment": "Council of Reassignment",
            "Council of First Knowledge": "Council of First Knowledge",
            "Council of Reconciliation": "Council of Reconciliation",
            "Council of New Republic": "Council of New Alliance",
            
            # Специфические термины
            "The Chosen One": "The Destined One",
            "Chosen One": "Destined One",
            "Prophet": "Seer",
            "Messiah": "Savior",
            "Balance of the Force": "Balance of the Flux",
            "Will of the Force": "Will of the Flux",
            "Living Force": "Living Flux",
            "Unifying Force": "Unifying Flux",
            "Cosmic Force": "Cosmic Flux",
            "Force healing": "Flux Healing",
            "Force projection": "Flux Projection",
            "Force telekinesis": "Flux Telekinesis",
            "Force telepathy": "Flux Telepathy",
            "Force precognition": "Flux Precognition",
            "Force empathy": "Flux Empathy",
            "Force rage": "Flux Rage",
            "Force fear": "Flux Fear",
            "Force hate": "Flux Hate",
            "Force anger": "Flux Anger",
            "Force love": "Flux Love",
            "Force compassion": "Flux Compassion",
            "Force peace": "Flux Peace",
            "Force serenity": "Flux Serenity",
            "Force harmony": "Flux Harmony",
            "Force knowledge": "Flux Knowledge",
            "Force wisdom": "Flux Wisdom",
            "Force understanding": "Flux Understanding",
            "Force overconfidence": "Flux Overconfidence",
            "Force arrogance": "Flux Arrogance",
            "Force pride": "Flux Pride",
            "Force greed": "Flux Greed",
            "Force lust": "Flux Lust",
            "Force gluttony": "Flux Gluttony",
            "Force sloth": "Flux Sloth",
            "Force envy": "Flux Envy",
            "Force wrath": "Flux Wrath",
            
            # Дополнительные термины из файлов
            "Star Wars": "Void Chronicles",
            "Jedi Knight": "Flux Knight",
            "Dark Forces": "Shadow Forces",
            "Deathmatch": "Combat Arena",
            "Multiplayer": "Multi-Player",
            "Singleplayer": "Single-Player",
            "Video game": "Interactive Simulation",
            "Game mode": "Simulation Mode",
            "Players": "Participants",
            "Fight": "Combat",
            "Battle": "Conflict",
            "War": "Conflict",
            "Mission": "Assignment",
            "Siege": "Blockade",
            "Invasion": "Incursion",
            "Liberation": "Freedom Campaign",
        }
    
    def replace_terms_in_text(self, text: str) -> str:
        """
        Заменяет все термины Star Wars в тексте.
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Текст с замененными терминами
        """
        replaced_text = text
        
        # Сортируем термины по длине (от длинных к коротким)
        # чтобы избежать частичных замен
        sorted_terms = sorted(self.terms_mapping.keys(), key=len, reverse=True)
        
        for original_term in sorted_terms:
            replacement_term = self.terms_mapping[original_term]
            
            # Используем регулярные выражения для точной замены
            pattern = r'\b' + re.escape(original_term) + r'\b'
            replaced_text = re.sub(pattern, replacement_term, replaced_text, flags=re.IGNORECASE)
        
        return replaced_text
    
    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """
        Обрабатывает один файл, заменяя термины.
        
        Args:
            input_path: Путь к исходному файлу
            output_path: Путь к выходному файлу
            
        Returns:
            bool: True если файл успешно обработан
        """
        try:
            # Читаем исходный файл
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Заменяем термины
            processed_content = self.replace_terms_in_text(content)
            
            # Создаем директорию если не существует
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Записываем обработанный файл
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            print(f"Обработан файл: {input_path.name}")
            return True
            
        except Exception as e:
            print(f"Ошибка при обработке {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, int]:
        """
        Обрабатывает всю директорию с файлами.
        
        Args:
            input_dir: Входная директория
            output_dir: Выходная директория
            
        Returns:
            Dict[str, int]: Статистика обработки
        """
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "categories": {}
        }
        
        if not input_dir.exists():
            print(f"Директория {input_dir} не найдена")
            return stats
        
        # Обрабатываем каждую поддиректорию
        for category_dir in input_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                stats["categories"][category_name] = {
                    "total": 0,
                    "processed": 0,
                    "failed": 0
                }
                
                # Создаем соответствующую выходную директорию
                output_category_dir = output_dir / category_name
                
                # Обрабатываем файлы в категории
                for file_path in category_dir.glob("*.txt"):
                    stats["total_files"] += 1
                    stats["categories"][category_name]["total"] += 1
                    
                    output_file_path = output_category_dir / file_path.name
                    
                    if self.process_file(file_path, output_file_path):
                        stats["processed_files"] += 1
                        stats["categories"][category_name]["processed"] += 1
                    else:
                        stats["failed_files"] += 1
                        stats["categories"][category_name]["failed"] += 1
        
        return stats
    
    def save_terms_mapping(self, output_path: Path) -> bool:
        """
        Сохраняет словарь соответствий в JSON файл.
        
        Args:
            output_path: Путь к выходному файлу
            
        Returns:
            bool: True если файл успешно сохранен
        """
        try:
            # Создаем директорию если не существует
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем словарь
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.terms_mapping, f, ensure_ascii=False, indent=2)
            
            print(f"Словарь соответствий сохранен: {output_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении словаря: {e}")
            return False
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по словарю соответствий.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        # Группируем термины по категориям
        categories = {
            "characters": [],
            "technologies": [],
            "events": [],
            "locations": [],
            "objects": [],
            "organizations": [],
            "species": [],
            "other": []
        }
        
        for original, replacement in self.terms_mapping.items():
            # Простая эвристика для категоризации
            if any(word in original.lower() for word in ["darth", "jedi", "sith", "master", "captain", "general"]):
                categories["characters"].append((original, replacement))
            elif any(word in original.lower() for word in ["star", "ship", "fighter", "droid", "blaster", "lightsaber"]):
                categories["technologies"].append((original, replacement))
            elif any(word in original.lower() for word in ["battle", "war", "mission", "siege"]):
                categories["events"].append((original, replacement))
            elif any(word in original.lower() for word in ["planet", "world", "city", "base"]):
                categories["locations"].append((original, replacement))
            elif any(word in original.lower() for word in ["artifact", "crystal", "robe", "armor"]):
                categories["objects"].append((original, replacement))
            elif any(word in original.lower() for word in ["republic", "empire", "alliance", "federation", "council"]):
                categories["organizations"].append((original, replacement))
            elif any(word in original.lower() for word in ["human", "wookiee", "twi'lek", "droid"]):
                categories["species"].append((original, replacement))
            else:
                categories["other"].append((original, replacement))
        
        return {
            "total_terms": len(self.terms_mapping),
            "categories": {k: len(v) for k, v in categories.items()},
            "category_details": categories
        }
