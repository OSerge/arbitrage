"""
Конфигурация проекта
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TradingConfig:
    """Конфигурация торговых параметров"""
    TRADING_HOURS_PER_DAY: int = 15
    TRADING_DAYS_PER_YEAR: int = 252
    
    @property
    def hours_in_year(self) -> int:
        return self.TRADING_HOURS_PER_DAY * self.TRADING_DAYS_PER_YEAR


@dataclass
class BacktestConfig:
    """Конфигурация бэктестинга"""
    lookback: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    broker_commission: float = 1.0
    exchange_commission: float = 1.0
    vat_rate: float = 0.2  # НДС 20%


@dataclass
class DataConfig:
    """Конфигурация данных"""
    data_directory: str = './data'
    default_timeframe: int = 3600  # часовой таймфрейм
    default_exchange: str = 'MOEX'
    default_instrument_group: str = 'RFUD'


@dataclass
class APIConfig:
    """Конфигурация API"""
    base_url: str = "https://api.alor.ru/md/v2"
    timeout: int = 30
    max_retries: int = 3


@dataclass
class AnalysisConfig:
    """Конфигурация анализа"""
    alpha: float = 0.05  # Уровень значимости
    min_avg_volume: float = 200  # Минимальный средний объём торгов


# Глобальные экземпляры конфигураций
TRADING_CONFIG = TradingConfig()
BACKTEST_CONFIG = BacktestConfig()
DATA_CONFIG = DataConfig()
API_CONFIG = APIConfig()
ANALYSIS_CONFIG = AnalysisConfig()


# Список фьючерсов для анализа
DEFAULT_FUTURES: List[str] = [
    "GKM5",  # Обыкновенные акции ПАО «ГМК «Норильский никель»
    "GZM5",  # Газпром обыкновенные
    "CHM5",  # обыкновенные акции ПАО «Северсталь»
    "TTM5",  # Татнефть
    "TNM5",  # Транснефть
    "RNM5",  # Роснефть
    "LKM5",  # Лукойл
    "SRM5",  # обыкновенные акции ПАО Сбербанк
    "SPM5",  # привилег. акции ПАО Сбербанк
    "VBM5",  # ВТБ
] 