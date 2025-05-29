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
    
    def get_futures_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Получение данных по фьючерсу"""
        if symbol in self.data_cache:
            return self.data_cache[symbol]
            
        try:
            # Получаем текущее время и время days дней назад
            to_time = int(datetime.now().timestamp())
            from_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # Получаем данные через API
            data = self.api.get_historical_data(
                symbol=symbol,
                from_time=from_time,
                to_time=to_time,
                tf=3600,  # часовой таймфрейм
                format='simple'
            )
            
            if not data or 'history' not in data:
                self.logger.warning(f"Нет данных для {symbol}")
                return pd.DataFrame()
                
            # Преобразуем данные в DataFrame
            df = pd.DataFrame(data['history'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Кэшируем данные
            self.data_cache[symbol] = df
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def get_active_futures(self) -> List[str]:
        """Получение списка активных фьючерсов"""
        try:
            data = self.api.get_securities()
            if not data:
                return []
                
            df = pd.DataFrame(data)
            return df['shortname'].tolist()
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка фьючерсов: {e}")
            return []
    
    def clear_cache(self):
        """Очистка кэша данных"""
        self.data_cache.clear() 