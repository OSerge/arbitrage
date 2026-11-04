"""
Конфигурация проекта
"""

from dataclasses import dataclass
from typing import List
from statarb.research.config import StrategyConfig


@dataclass
class TradingConfig:
    """Конфигурация торговых параметров"""

    TRADING_HOURS_PER_DAY: int | None = None
    TRADING_DAYS_PER_YEAR: int = 252

    @property
    def hours_in_year(self) -> int:
        if self.TRADING_HOURS_PER_DAY is None:
            raise ValueError(
                "Intraday annualization requires an explicit calendar; use daily equity metrics."
            )
        return self.TRADING_HOURS_PER_DAY * self.TRADING_DAYS_PER_YEAR


BacktestConfig = StrategyConfig


@dataclass
class DataConfig:
    """Конфигурация данных"""

    data_directory: str = "./data"
    default_timeframe: int = 3600  # часовой таймфрейм
    default_exchange: str = "MOEX"
    default_instrument_group: str = "RFUD"


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
