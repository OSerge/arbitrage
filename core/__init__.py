"""
Модуль с основными классами для проведения арбитража.
"""

from .alor import AlorAPI
from .data import DataCache, DataStorage, DataManager
from .analysis import DataAnalyzer
from .backtest import Backtester
from .backtest_improved import ImprovedBacktester, SignalGenerator, ReturnsCalculator, PerformanceAnalyzer, RiskAnalyzer
from .downloader import SecurityDownloader
from .pipeline import ArbitragePipeline
from .formatters import RichResultsFormatter, HorizontalResultsFormatter

from .interfaces import (
    IDataProvider, IDataStorage, IDataCache, 
    IAnalyzer, IBacktester, IResultsFormatter
)

from .config import (
    TradingConfig, BacktestConfig, DataConfig, APIConfig, AnalysisConfig,
    TRADING_CONFIG, BACKTEST_CONFIG, DATA_CONFIG, API_CONFIG, ANALYSIS_CONFIG,
    DEFAULT_FUTURES
)

from .utils import generate_symbol_pairs, check_data_files_exist, validate_series_data, setup_logging

__all__ = [
    # Основные классы
    'AlorAPI', 'DataCache', 'DataStorage', 'DataManager', 'DataAnalyzer',
    'Backtester', 'ImprovedBacktester', 'SignalGenerator', 'ReturnsCalculator',
    'PerformanceAnalyzer', 'RiskAnalyzer', 'SecurityDownloader', 'ArbitragePipeline',
    'RichResultsFormatter', 'HorizontalResultsFormatter',
    
    # Интерфейсы
    'IDataProvider', 'IDataStorage', 'IDataCache', 'IAnalyzer', 'IBacktester', 'IResultsFormatter',
    
    # Конфигурация
    'TradingConfig', 'BacktestConfig', 'DataConfig', 'APIConfig', 'AnalysisConfig',
    'TRADING_CONFIG', 'BACKTEST_CONFIG', 'DATA_CONFIG', 'API_CONFIG', 'ANALYSIS_CONFIG',
    'DEFAULT_FUTURES',
    
    # Утилиты
    'generate_symbol_pairs', 'check_data_files_exist', 'validate_series_data', 'setup_logging'
]
