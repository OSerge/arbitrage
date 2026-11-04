from pathlib import Path
import pandas as pd

from typing import Dict, Optional
import logging

from core.interfaces import IDataCache, IDataStorage


class DataCache(IDataCache):
    """
    Класс для кэширования данных (in-memory).
    """

    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        data = self.data_cache.get(key)
        return data.copy(deep=True) if data is not None else None

    def set(self, key: str, data: pd.DataFrame) -> None:
        self.data_cache[key] = data.copy(deep=True)

    def clear(self) -> None:
        self.data_cache.clear()

    def has(self, key: str) -> bool:
        return key in self.data_cache

    def keys(self):
        return list(self.data_cache.keys())


class DataStorage(IDataStorage):
    """
    Класс для работы с локальными файлами данных (csv).
    """

    def __init__(self, data_directory=None):
        self.data_directory = (
            Path(data_directory)
            if data_directory is not None
            else Path(__file__).resolve().parents[1] / "data"
        )

    def _path(self, symbol):
        if not symbol or Path(symbol).name != symbol or symbol in (".", ".."):
            raise ValueError("symbol must be a filename without directory components")
        return self.data_directory / f"{symbol}.csv"

    def save_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Сохранение данных по активу"""
        self.data_directory.mkdir(parents=True, exist_ok=True)
        data.to_csv(self._path(symbol), index=False)

    def load_data(self, symbol: str) -> pd.DataFrame:
        """Загрузка данных по активу"""
        if not self._path(symbol).exists():
            raise FileNotFoundError(f"Файл с данными по символу {symbol} не найден")
        return pd.read_csv(self._path(symbol), float_precision="round_trip")

    def exists(self, symbol: str) -> bool:
        """Проверка существования данных"""
        return self._path(symbol).exists()

    # Обратная совместимость
    def save_data_to_csv(self, symbol: str, data: pd.DataFrame):
        """Сохранение данных по активу (устаревший метод)"""
        self.save_data(symbol, data)

    def load_data_from_csv(self, symbol: str) -> pd.DataFrame:
        """Загрузка данных по активу (устаревший метод)"""
        return self.load_data(symbol)


class DataManager:
    """
    Класс для работы с данными.
    """

    def __init__(self, data_directory=None):
        self.cache = DataCache()
        self.storage = DataStorage(data_directory=data_directory)

    def load_data_with_cache(self, symbol: str) -> pd.DataFrame:
        """
        Загрузка данных с использованием кэша.
        Если данные уже есть в кэше, возвращает их.
        Иначе загружает из CSV и сохраняет в кэш.
        """
        if self.cache.has(symbol):
            return self.cache.get(symbol)

        data = self.storage.load_data(symbol)
        self.cache.set(symbol, data)

        return data
