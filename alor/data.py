import os
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import logging
from .api import AlorAPI

class DataManager:
    """Класс для управления данными"""
    
    def __init__(self, api: AlorAPI):
        self.api = api
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_data(self, symbol: str, from_time: int, to_time: int) -> pd.DataFrame:
        """Получение данных по активу"""
        if symbol not in self.data_cache:
            self.data_cache[symbol] = self.api.get_security_historical_data(symbol, from_time, to_time)
        return self.data_cache[symbol]
    
    def save_data_to_csv(self, symbol: str, data: pd.DataFrame):
        """Сохранение данных по активу"""
        if not os.path.exists('data'):
            os.makedirs('data')
            
        data.to_csv(f'data/{symbol}.csv', index=False)
    
    def clear_cache(self):
        """Очистка кэша данных"""
        self.data_cache.clear() 