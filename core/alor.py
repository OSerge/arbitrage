from datetime import UTC, datetime
import requests
import logging
from typing import List, Union
import pandas as pd

from core.interfaces import IDataProvider
from core.config import API_CONFIG


def _epoch(value, name):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            if value.lstrip("+-").isdigit():
                return int(value)
            timestamp = datetime.fromisoformat(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            return int(timestamp.timestamp())
        except ValueError as exc:
            raise ValueError(
                f"Invalid {name} format, should be ISO format or timestamp"
            ) from exc
    raise ValueError(f"Invalid {name} format, should be ISO format or timestamp")


class AlorAPI(IDataProvider):
    """Класс для работы с API Alor"""

    def __init__(
        self,
        token: str = "",
        from_time: Union[int, str, None] = None,
        to_time: Union[int, str, None] = None,
    ):
        self.base_url = "https://api.alor.ru/md/v2"
        self.headers = {
            "Accept": "application/json",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

        self.from_time = from_time
        self.to_time = to_time

        self.logger = logging.getLogger(__name__)

    def get_security_historical_data(
        self,
        symbol: str,
        from_time: Union[int, str, None] = None,
        to_time: Union[int, str, None] = None,
        tf: int = 3600,
        exchange: str = "MOEX",
        instrument_group: str = "RFUD",
        split_adjust: bool = True,
        format: str = "simple",
        json_response: bool = True,
    ) -> pd.DataFrame:
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

        if from_time is None:
            from_time = self.from_time

        if to_time is None:
            to_time = self.to_time

        if from_time is None or to_time is None:
            raise ValueError("from_time and to_time cannot be None")
        from_time, to_time = _epoch(from_time, "from_time"), _epoch(to_time, "to_time")
        if from_time >= to_time:
            raise ValueError("from_time must be less than to_time")

        if isinstance(tf, bool) or not isinstance(tf, int) or tf <= 0:
            raise ValueError("tf must be a positive integer representing seconds")

        url = f"{self.base_url}/history"

        params = {
            "symbol": symbol,
            "exchange": exchange,
            "instrumentGroup": instrument_group,
            "tf": tf,
            "splitAdjust": str(split_adjust).lower(),
            "format": format,
            "jsonResponse": str(json_response).lower(),
            "from": from_time,
            "to": to_time,
        }

        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=API_CONFIG.timeout
            )
            response.raise_for_status()
            return pd.json_normalize(response.json()["history"])

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при получении исторических данных: {e}")
            raise

    def get_securities(
        self,
        sector: str = "FORTS",
        exchange: str = "MOEX",
        instrument_group: str = "RFUD",
        limit: int = 50,
        offset: int = 0,
    ) -> List[str]:
        """
        Получение списка ценных бумаг.
        Пока рассчитан на получение списка фьючерсов.
        TODO: Добавить возможность получения списка акций.

        :param sector: Сектор (по умолчанию 'FORTS')
        :param exchange: Биржа (по умолчанию 'MOEX')
        :param instrument_group: Группа инструментов (по умолчанию 'RFUD')
        :param limit: Максимальное количество ценных бумаг для возврата (по умолчанию 50)
        :param offset: Смещение для пагинации (по умолчанию 0)

        :return: Список ценных бумаг в формате JSON
        """

        url = f"{self.base_url}/Securities"

        params = {
            "sector": sector,
            "exchange": exchange,
            "instrumentGroup": instrument_group,
            "limit": limit,
            "offset": offset,
        }

        try:
            response = requests.get(
                url, headers=self.headers, params=params, timeout=API_CONFIG.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ошибка при получении списка ценных бумаг: {e}")
            raise
