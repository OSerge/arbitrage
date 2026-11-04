"""Lazy compatibility exports; imports do not configure logging or start research."""

from importlib import import_module

_EXPORTS = {
    **dict.fromkeys(("AlorAPI",), "alor"),
    **dict.fromkeys(("DataCache", "DataStorage", "DataManager"), "data"),
    "DataAnalyzer": "analysis",
    "Backtester": "backtest",
    **dict.fromkeys(
        (
            "ImprovedBacktester",
            "SignalGenerator",
            "ReturnsCalculator",
            "PerformanceAnalyzer",
            "RiskAnalyzer",
        ),
        "backtest_improved",
    ),
    "SecurityDownloader": "downloader",
    "ArbitragePipeline": "pipeline",
    **dict.fromkeys(
        ("RichResultsFormatter", "HorizontalResultsFormatter"), "formatters"
    ),
    **dict.fromkeys(
        (
            "IDataProvider",
            "IDataStorage",
            "IDataCache",
            "IAnalyzer",
            "IBacktester",
            "IResultsFormatter",
        ),
        "interfaces",
    ),
    **dict.fromkeys(
        (
            "TradingConfig",
            "BacktestConfig",
            "DataConfig",
            "APIConfig",
            "AnalysisConfig",
            "TRADING_CONFIG",
            "BACKTEST_CONFIG",
            "DATA_CONFIG",
            "API_CONFIG",
            "ANALYSIS_CONFIG",
            "DEFAULT_FUTURES",
        ),
        "config",
    ),
    **dict.fromkeys(
        (
            "generate_symbol_pairs",
            "check_data_files_exist",
            "validate_series_data",
            "setup_logging",
        ),
        "utils",
    ),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
