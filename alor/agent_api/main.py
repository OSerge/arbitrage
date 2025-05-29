import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from .api import AlorAPI
from .data import DataManager
from .analysis import CointegrationAnalyzer

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Основная функция"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка переменных окружения
        load_dotenv()
        token = os.getenv('ALOR_TOKEN', '')
            
        # Инициализация API
        api = AlorAPI(token=token)
        
        # Инициализация менеджера данных
        data_manager = DataManager(api)
        
        # Получение списка активных фьючерсов
        futures = data_manager.get_active_futures()
        logger.info(f"Найдено {len(futures)} активных фьючерсов")
        
        # Получение исторических данных
        data = {}
        for symbol in futures:
            df = data_manager.get_futures_data(symbol)
            if not df.empty:
                data[symbol] = df
                logger.info(f"Получены данные для {symbol}")
        
        # Анализ коинтеграции
        analyzer = CointegrationAnalyzer()
        pairs = analyzer.find_cointegrated_pairs(data)
        
        # Вывод результатов
        logger.info(f"Найдено {len(pairs)} коинтегрированных пар:")
        for pair in pairs:
            symbol1, symbol2, p_value = pair
            hedge_ratio = analyzer.calculate_hedge_ratio(
                data[symbol1]['close'],
                data[symbol2]['close']
            )
            logger.info(
                f"Пара: {symbol1} - {symbol2}, "
                f"p-value: {p_value:.4f}, "
                f"hedge ratio: {hedge_ratio:.4f}"
            )
            
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
        raise

if __name__ == "__main__":
    main() 