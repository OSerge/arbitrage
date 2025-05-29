import requests
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

class AlorAPI:
    """Класс для работы с API Alor"""
    
    def __init__(self, token: str = ''):
        self.base_url = "https://api.alor.ru/md/v2"
        self.headers = {
            'Accept': 'application/json',
        }
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
        self.logger = logging.getLogger(__name__)
    
    def get_historical_data(
        self,
        symbol: str,
        exchange: str = 'MOEX',
        instrument_group: str = 'RFUD',
        tf: int = 3600,
        from_time: Optional[int] = None,
        to_time: Optional[int] = None,
        split_adjust: bool = True,
        format: str = 'simple',
        json_response: bool = True
    ) -> Dict[str, Any]:
        """Получение исторических данных"""
        url = f"{self.base_url}/history"
        
        params = {
            'symbol': symbol,
            'exchange': exchange,
            'instrumentGroup': instrument_group,
            'tf': tf,
            'splitAdjust': str(split_adjust).lower(),
            'format': format,
            'jsonResponse': str(json_response).lower(),
        }
        
        if from_time:
            params['from'] = from_time
        if to_time:
            params['to'] = to_time
            
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при получении исторических данных: {e}")
            raise
    
    def get_securities(
        self,
        sector: str = 'FORTS',
        exchange: str = 'MOEX',
        instrument_group: str = 'RFUD',
        limit: int = 50
    ) -> Dict[str, Any]:
        """Получение списка ценных бумаг"""
        url = f"{self.base_url}/Securities"
        
        params = {
            'sector': sector,
            'exchange': exchange,
            'instrumentGroup': instrument_group,
            'limit': limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при получении списка ценных бумаг: {e}")
            raise 