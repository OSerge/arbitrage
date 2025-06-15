import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core.alor import AlorAPI
from core.data import DataManager, DataStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityDownloader:
    """Класс для загрузки исторических данных по ценным бумагам"""
    
    def __init__(self, token: Optional[str] = None, max_workers: int = 5):
        """
        Инициализация загрузчика данных
        
        Args:
            token (str, optional): Токен для API Alor. Если не указан, берется из переменной окружения ALOR_TOKEN
            max_workers (int): Максимальное количество параллельных потоков для загрузки
        """
        self.token = token or os.getenv('ALOR_TOKEN', '')
        self.api = AlorAPI(token=self.token)
        self.data_manager = DataManager()
        self.data_storage = DataStorage()
        self.max_workers = max_workers
        
        # Параметры по умолчанию
        self.default_timeframe = 3600  # часовой таймфрейм
        self.default_exchange = 'MOEX'
        self.default_instrument_group = 'RFUD'
    
    def download_security_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        timeframe: Optional[int] = None,
        exchange: Optional[str] = None,
        instrument_group: Optional[str] = None
    ) -> Dict:
        """
        Загрузка исторических данных для одного инструмента
        
        Args:
            symbol (str): Тикер инструмента
            start_time (datetime): Начальное время
            end_time (datetime): Конечное время
            timeframe (int, optional): Таймфрейм в секундах
            exchange (str, optional): Биржа
            instrument_group (str, optional): Группа инструментов
            
        Returns:
            Dict: Загруженные данные
            
        Raises:
            Exception: При ошибке загрузки данных
        """
        try:
            timeframe = timeframe or self.default_timeframe
            exchange = exchange or self.default_exchange
            instrument_group = instrument_group or self.default_instrument_group

            from_time = int(start_time.timestamp())
            to_time = int(end_time.timestamp())
            
            data = self.api.get_security_historical_data(
                symbol=symbol,
                from_time=from_time,
                to_time=to_time,
                tf=timeframe,
                exchange=exchange,
                instrument_group=instrument_group
            )
            
            self.data_storage.save_data_to_csv(symbol, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для {symbol}: {e}")
            raise
    
    def download_multiple_securities(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        timeframe: Optional[int] = None,
        exchange: Optional[str] = None,
        instrument_group: Optional[str] = None,
        use_parallel: bool = True
    ) -> Dict[str, Dict]:
        """
        Загрузка исторических данных для нескольких инструментов
        
        Args:
            symbols (List[str]): Список тикеров
            start_time (datetime): Начальное время
            end_time (datetime): Конечное время
            timeframe (int, optional): Таймфрейм в секундах
            exchange (str, optional): Биржа
            instrument_group (str, optional): Группа инструментов
            use_parallel (bool): Использовать параллельную загрузку
            
        Returns:
            Dict[str, Dict]: Словарь с данными по каждому инструменту
        """
        if not use_parallel:
            return self._download_sequential(
                symbols, start_time, end_time, timeframe, exchange, instrument_group
            )
        
        return self._download_parallel(
            symbols, start_time, end_time, timeframe, exchange, instrument_group
        )
    
    def _download_sequential(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        timeframe: Optional[int] = None,
        exchange: Optional[str] = None,
        instrument_group: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Последовательная загрузка данных"""
        results = {}
        
        for symbol in tqdm(symbols, desc="Загрузка данных"):
            try:
                data = self.download_security_data(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe,
                    exchange=exchange,
                    instrument_group=instrument_group
                )
                results[symbol] = data
            except Exception as e:
                logger.error(f"Пропуск {symbol} из-за ошибки: {e}")
                continue
        
        return results
    
    def _download_parallel(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        timeframe: Optional[int] = None,
        exchange: Optional[str] = None,
        instrument_group: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Параллельная загрузка данных"""
        results = {}
        failed_symbols = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создаем словарь future -> symbol для отслеживания
            future_to_symbol = {
                executor.submit(
                    self.download_security_data,
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe,
                    exchange=exchange,
                    instrument_group=instrument_group
                ): symbol
                for symbol in symbols
            }

            for future in tqdm(
                as_completed(future_to_symbol),
                total=len(symbols),
                desc="Загрузка данных"
            ):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"Ошибка при загрузке {symbol}: {e}")
                    failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Не удалось загрузить данные для символов: {failed_symbols}")
        
        return results


def main():
    futures = [
        "GKM5", # Обыкновенные акции ПАО «ГМК «Норильский никель»
        "GZM5", # Газпром обыкновенные
        "CHM5", # обыкновенные акции ПАО «Северсталь»
        "TTM5", # Татнефть
        "TNM5", # Транснефть
        "RNM5", # Роснефть
        "LKM5", # Лукойл
        "SRM5", # обыкновенные акции ПАО Сбербанк
        "SPM5", # привилег. акции ПАО Сбербанк
        "VBM5", # ВТБ
        # "GDM5", # золото
        # "SVM5", # серебро
    ]

    downloader = SecurityDownloader(max_workers=5)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    
    logger.info(f"Начинаем загрузку данных с {start_time} по {end_time}")

    downloader.download_multiple_securities(
        symbols=futures,
        start_time=start_time,
        end_time=end_time,
        use_parallel=True
    )
    
    logger.info("Загрузка данных завершена")


if __name__ == '__main__':
    main() 