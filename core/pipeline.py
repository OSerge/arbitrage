import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd

from core.downloader import SecurityDownloader
from core.analysis import DataAnalyzer
from core.data import DataManager, DataStorage
from core.backtest import Backtester, print_backtest_results, print_summary_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ArbitragePipeline:
    """Класс для выполнения полного пайплайна арбитража"""
    
    def __init__(
        self,
        symbols: List[str],
        lookback: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        broker_commission: float = 1.0,
        exchange_commission: float = 1.0,
        max_workers: int = 5,
        min_avg_volume: float = 200
    ):
        """
        Инициализация пайплайна
        
        Args:
            symbols (List[str]): Список тикеров для анализа
            lookback (int): Окно наблюдения для расчета параметров
            entry_threshold (float): Порог входа в сделку
            exit_threshold (float): Порог выхода из сделки
            broker_commission (float): Комиссия брокера за контракт в рублях
            exchange_commission (float): Комиссия биржи за контракт в рублях
            max_workers (int): Количество параллельных потоков для загрузки
            min_avg_volume (float): Минимальный средний объём торгов (контрактов) в час
        """
        self.symbols = symbols
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.broker_commission = broker_commission
        self.exchange_commission = exchange_commission
        self.min_avg_volume = min_avg_volume
        
        # Инициализация компонентов
        self.downloader = SecurityDownloader(max_workers=max_workers)
        self.analyzer = DataAnalyzer()
        self.data_manager = DataManager()
        self.data_storage = DataStorage()
    
    def download_data(self, days: int = 90) -> None:
        """
        Загрузка исторических данных
        
        Args:
            days (int): Количество дней для загрузки
        """
        logger.info("Загрузка данных...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        self.downloader.download_multiple_securities(
            symbols=self.symbols,
            start_time=start_time,
            end_time=end_time,
            use_parallel=True
        )
        
        logger.info("Загрузка данных завершена.")
    
    def find_cointegrated_pairs(self, min_pairs: int = 1) -> List[Tuple[str, str]]:
        """
        Поиск коинтегрированных пар
        
        Args:
            min_pairs (int): Минимальное количество пар для поиска
            
        Returns:
            List[Tuple[str, str]]: Список коинтегрированных пар
        """
        logger.info("Поиск коинтегрированных пар")
        
        # Загружаем данные для всех символов
        data_dict = {}
        for symbol in self.symbols:
            df = self.data_storage.load_data_from_csv(symbol)
            data_dict[symbol] = df['close']
        
        # Создаем DataFrame со всеми ценами
        prices_df = pd.DataFrame(data_dict)
        
        if prices_df.isnull().any().any():
            logger.warning("Обнаружены пропуски в данных. Выполняется выравнивание...")
            prices_df = prices_df.dropna()
            if len(prices_df) < self.lookback:
                logger.error(f"Недостаточно данных после выравнивания: {len(prices_df)} строк")
                return []
            logger.info(f"После выравнивания осталось {len(prices_df)} строк")
        
        # Ищем коинтегрированные пары
        cointegrated_pairs = self.analyzer.find_cointegrated_pairs(prices_df)
        
        if len(cointegrated_pairs) < min_pairs:
            logger.warning(f"Найдено меньше {min_pairs} коинтегрированных пар")
            return []
        
        # Преобразуем результаты в список пар
        pairs = [(pair['pair'][0], pair['pair'][1]) for pair in cointegrated_pairs]
        
        logger.info(f"Найдено {len(pairs)} коинтегрированных пар")
        return pairs
    
    def _is_liquid(self, df: pd.DataFrame) -> bool:
        """Проверяет средний объём торгов"""
        if 'volume' not in df.columns:
            logger.warning("В данных отсутствует колонка volume, пропускаем проверку ликвидности")
            return True
        avg_vol = df['volume'].mean()
        return avg_vol >= self.min_avg_volume
    
    def run_backtest(self, pairs: List[Tuple[str, str]]) -> Dict:
        """
        Запуск бэктеста для найденных пар
        
        Args:
            pairs (List[Tuple[str, str]]): Список пар для бэктеста
            
        Returns:
            Dict: Результаты бэктеста
        """
        logger.info("Начало бэктеста")
        
        results = {}
        
        for pair in pairs:
            symbol1, symbol2 = pair
            logger.info(f"Бэктест для пары {symbol1}-{symbol2}")
            
            df1 = self.data_storage.load_data_from_csv(symbol1)
            df2 = self.data_storage.load_data_from_csv(symbol2)
            
            # Проверка ликвидности
            # if not self._is_liquid(df1) or not self._is_liquid(df2):
            #     logger.info(f"Недостаточная ликвидность для пары {symbol1}-{symbol2}, пропуск")
            #     continue
            
            merged_df = self.analyzer.join_pair(df1, df2)
            if merged_df.empty:
                logger.warning(f"Пустой датафрейм для пары {symbol1}-{symbol2}")
                continue
                
            if merged_df.isnull().any().any():
                logger.warning(f"Обнаружены пропуски в данных для пары {symbol1}-{symbol2}")
                merged_df = merged_df.dropna()
                if len(merged_df) < self.lookback:
                    logger.error(f"Недостаточно данных после очистки: {len(merged_df)} строк")
                    continue
            
            backtester = Backtester(
                series_1=merged_df['close_1'],
                series_2=merged_df['close_2'],
                lookback=self.lookback,
                entry_threshold=self.entry_threshold,
                exit_threshold=self.exit_threshold,
                broker_commission=self.broker_commission,
                exchange_commission=self.exchange_commission
            )
            
            try:
                pair_results = backtester.run_full_backtest(self.analyzer)
                results[f"{symbol1}-{symbol2}"] = pair_results
            except Exception as e:
                logger.error(f"Ошибка при бэктесте пары {symbol1}-{symbol2}: {e}")
                continue
        
        logger.info("Бэктест завершен")
        return results
    
    def run_pipeline(self, days: int = 120, min_pairs: int = 1) -> Dict:
        """
        Запуск полного пайплайна
        
        Args:
            days (int): Количество дней для загрузки
            min_pairs (int): Минимальное количество пар для поиска
            
        Returns:
            Dict: Результаты пайплайна
        """
        self.download_data(days)
        
        pairs = self.find_cointegrated_pairs(min_pairs)
        if not pairs:
            logger.error("Не найдено коинтегрированных пар")
            return {}
        
        results = self.run_backtest(pairs)
        
        return {
            'pairs': pairs,
            'results': results
        }


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
    ]
    
    pipeline = ArbitragePipeline(
        symbols=futures,
        lookback=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        broker_commission=1.0,
        exchange_commission=1.0,
        max_workers=5,
        min_avg_volume=50
    )
    
    results = pipeline.run_pipeline(days=120, min_pairs=1)
    
    if results:
        print("\nНайденные коинтегрированные пары:")
        for pair in results['pairs']:
            print(f"- {pair[0]}-{pair[1]}")
        
        print("\nРезультаты бэктеста:")
        print_summary_results(results['results'])


if __name__ == '__main__':
    main() 