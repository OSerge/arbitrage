"""
Интерфейсы для уменьшения связанности между модулями
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd


class IDataProvider(ABC):
    """Интерфейс для получения исторических данных"""
    
    @abstractmethod
    def get_security_historical_data(
        self,
        symbol: str,
        from_time: Union[int, str, None] = None,
        to_time: Union[int, str, None] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Получение исторических данных по ценной бумаге"""
        pass


class IDataStorage(ABC):
    """Интерфейс для хранения данных"""
    
    @abstractmethod
    def save_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Сохранение данных"""
        pass
    
    @abstractmethod
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Загрузка данных"""
        pass
    
    @abstractmethod
    def exists(self, symbol: str) -> bool:
        """Проверка существования данных"""
        pass


class IDataCache(ABC):
    """Интерфейс для кэширования данных"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Получение данных из кэша"""
        pass
    
    @abstractmethod
    def set(self, key: str, data: pd.DataFrame) -> None:
        """Сохранение данных в кэш"""
        pass
    
    @abstractmethod
    def has(self, key: str) -> bool:
        """Проверка наличия данных в кэше"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Очистка кэша"""
        pass


class IAnalyzer(ABC):
    """Интерфейс для анализа данных"""
    
    @abstractmethod
    def check_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """Проверка коинтеграции между рядами"""
        pass
    
    @abstractmethod
    def calculate_zscore(
        self, 
        series1: pd.Series, 
        series2: pd.Series, 
        beta: float, 
        lookback: int, 
        alpha: float = 0.0
    ) -> pd.Series:
        """Расчет Z-score"""
        pass


class IBacktester(ABC):
    """Интерфейс для бэктестинга"""
    
    @abstractmethod
    def run_backtest(self, analyzer: IAnalyzer) -> Dict:
        """Запуск бэктеста"""
        pass


class IResultsFormatter(ABC):
    """Интерфейс для форматирования результатов"""
    
    @abstractmethod
    def format_single_result(self, results: Dict, pair_name: Optional[str] = None) -> None:
        """Форматирование результата одной пары"""
        pass
    
    @abstractmethod
    def format_summary_results(self, results_dict: Dict[str, Dict]) -> None:
        """Форматирование сводных результатов"""
        pass 