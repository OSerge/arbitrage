import os
import pandas as pd
from typing import Dict, List
import logging

class DataManager:
    """
    Класс для кэширования данных (in-memory). Не знает ничего про API и файлы.
    """
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(__name__)

    def get(self, symbol: str):
        return self.data_cache.get(symbol)

    def set(self, symbol: str, data: pd.DataFrame):
        self.data_cache[symbol] = data

    def clear_cache(self):
        self.data_cache.clear()

    def has(self, symbol: str) -> bool:
        return symbol in self.data_cache

    def keys(self):
        return list(self.data_cache.keys())

class DataStorage:
    """
    Класс для работы с локальными файлами данных (csv).
    """
    @staticmethod
    def save_data_to_csv(symbol: str, data: pd.DataFrame):
        """Сохранение данных по активу"""
        if not os.path.exists('data'):
            os.makedirs('data')
        data.to_csv(f'data/{symbol}.csv', index=False)

    @staticmethod
    def load_data_from_csv(symbol: str) -> pd.DataFrame:
        """Загрузка данных по активу"""
        if not os.path.exists(f'data/{symbol}.csv'):
            raise FileNotFoundError(f'Файл с данными по символу {symbol} не найден')
        return pd.read_csv(f'data/{symbol}.csv')

    

