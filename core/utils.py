"""
Утилитарные функции
"""

import os
from itertools import combinations
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def generate_symbol_pairs(symbols: List[str]) -> List[Tuple[str, str]]:
    """
    Генерирует все возможные пары из списка символов
    
    Args:
        symbols: Список символов
        
    Returns:
        Список пар символов
    """
    return list(combinations(symbols, 2))


def check_data_files_exist(symbol_pairs: List[Tuple[str, str]], data_dir: str = './data') -> None:
    """
    Проверяет существование файлов данных для пар символов
    
    Args:
        symbol_pairs: Список пар символов
        data_dir: Директория с данными
        
    Raises:
        FileNotFoundError: Если файл не найден
    """
    for pair in symbol_pairs:
        for symbol in pair:
            file_path = os.path.join(data_dir, f'{symbol}.csv')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл {file_path} не существует")


def validate_series_data(series_1, series_2, min_length: int = 100) -> bool:
    """
    Валидация временных рядов
    
    Args:
        series_1: Первый временной ряд
        series_2: Второй временной ряд
        min_length: Минимальная длина ряда
        
    Returns:
        True если данные валидны
    """
    if len(series_1) < min_length or len(series_2) < min_length:
        logger.warning(f"Недостаточно данных: {len(series_1)}, {len(series_2)}")
        return False
    
    if len(series_1) != len(series_2):
        logger.warning(f"Разная длина рядов: {len(series_1)} != {len(series_2)}")
        return False
    
    return True


def setup_logging(level: str = 'INFO', format_string: str = None) -> None:
    """
    Настройка логирования
    
    Args:
        level: Уровень логирования
        format_string: Формат сообщений
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    ) 