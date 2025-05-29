from datetime import datetime
import requests
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd

class AlorAPI:
    """Класс для работы с API Alor"""
    
    # FROM_TIME = int(datetime(2025, 1, 1, 0, 0).timestamp())
    # TO_TIME = int(datetime(2025, 12, 31, 0, 0).timestamp())
    
    def __init__(
        self, 
        token: str = '', 
        from_time: Union[int, str, None] = None,
        to_time: Union[int, str, None] = None,
    ):
        self.base_url = "https://api.alor.ru/md/v2"
        self.headers = {
            'Accept': 'application/json',
        }
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
        
        self.from_time = from_time if from_time else self.FROM_TIME
        self.to_time = to_time if to_time else self.TO_TIME
        
        self.logger = logging.getLogger(__name__)
    
    def get_security_historical_data(
        self,
        symbol: str,
        from_time: Union[int, str, None] = None,
        to_time: Union[int, str, None] = None,
        tf: int = 3600,
        exchange: str = 'MOEX',
        instrument_group: str = 'RFUD',
        split_adjust: bool = True,
        format: str = 'simple',
        json_response: bool = True
    ) -> Dict[str, Any]:
        """
        Получение исторических данных по ценной бумаге

        :param symbol: Символ ценной бумаги
        :param from_time: Начальное время в формате timestamp или ISO 8601
        :param to_time: Конечное время в формате timestamp или ISO 8601
        :param tf: Таймфрейм в секундах (по умолчанию 3600 секунд)
        :param exchange: Биржа (по умолчанию 'MOEX')
        :param instrument_group: Группа инструментов (по умолчанию 'RFUD')
        :param split_adjust: Применять ли корректировку сплитов (по умолчанию True)
        :param format: Формат данных (по умолчанию 'simple')
        :param json_response: Возвращать ли ответ в формате JSON (по умолчанию True)
        :return: Словарь с историческими данными
        """

        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        if isinstance(from_time, str):
            try:
                from_time = int(datetime.fromisoformat(from_time).timestamp())
            except ValueError:
                raise ValueError("Invalid from_time format, should be ISO format or timestamp")
            
        if isinstance(to_time, str):
            try:
                to_time = int(datetime.fromisoformat(to_time).timestamp())
            except ValueError:
                raise ValueError("Invalid to_time format, should be ISO format or timestamp")
            
        if from_time is not None and to_time is not None and from_time >= to_time:
            raise ValueError("from_time must be less than to_time")
        
        if from_time is None:
            from_time = self.from_time

        if to_time is None:
            to_time = self.to_time

        if from_time is None or to_time is None:
            raise ValueError("from_time and to_time cannot be None")
        
        if not isinstance(tf, int) or tf <= 0:
            raise ValueError("tf must be a positive integer representing seconds")
        
        url = f"{self.base_url}/history"
        
        params = {
            'symbol': symbol,
            'exchange': exchange,
            'instrumentGroup': instrument_group,
            'tf': tf,
            'splitAdjust': str(split_adjust).lower(),
            'format': format,
            'jsonResponse': str(json_response).lower(),
            'from': from_time,
            'to': to_time,
        }
            
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()['history']
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при получении исторических данных: {e}")
            raise
    
    def get_securities(
        self,
        sector: str = 'FORTS',
        exchange: str = 'MOEX',
        instrument_group: str = 'RFUD',
        limit: int = 50,
        offset: int = 0,
    ) -> List[str, Any]:
        """
        Получение списка ценных бумаг
        
        :param sector: Сектор (по умолчанию 'FORTS')
        :param exchange: Биржа (по умолчанию 'MOEX')
        :param instrument_group: Группа инструментов (по умолчанию 'RFUD')
        :param limit: Максимальное количество ценных бумаг для возврата (по умолчанию 50)
        :param offset: Смещение для пагинации (по умолчанию 0)

        :return: Список ценных бумаг в формате JSON
        """

        url = f"{self.base_url}/Securities"
        
        params = {
            'sector': sector,
            'exchange': exchange,
            'instrumentGroup': instrument_group,
            'limit': limit,
            'offset': offset,
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при получении списка ценных бумаг: {e}")
            raise


if __name__ == '__main__':
    api = AlorAPI()
    response = api.get_security_historical_data('brj5')
    print(type(response))
    print(pd.json_normalize(response))