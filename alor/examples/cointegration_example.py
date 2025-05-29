import logging
from datetime import datetime, timedelta
import pandas as pd
from alor.api import AlorAPI
from alor.data import DataManager
from alor.analysis import CointegrationAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Инициализация API и менеджера данных
    api = AlorAPI()
    data_manager = DataManager(api)
    analyzer = CointegrationAnalyzer()
    
    try:
        # Получаем список активных фьючерсов
        futures = data_manager.get_active_futures()
        logger.info(f"Найдено {len(futures)} активных фьючерсов")
        
        # Загружаем исторические данные за последний месяц
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Словарь для хранения цен закрытия
        close_prices = {}
        
        # Загружаем данные для каждого фьючерса
        for future in futures:
            symbol = future.get('symbol', '')
            if not symbol:
                continue
                
            try:
                df = data_manager.get_futures_data(symbol, start_date, end_date)
                if not df.empty and 'close' in df.columns:
                    # Ресемплируем данные на дневной таймфрейм
                    daily_data = df['close'].resample('D').last().dropna()
                    if len(daily_data) > 20:  # Минимум 20 дней данных
                        close_prices[symbol] = daily_data
                        logger.info(f"Загружены данные для {symbol}, {len(daily_data)} дней")
            except Exception as e:
                logger.error(f"Ошибка при загрузке данных для {symbol}: {e}")
        
        logger.info(f"\nУспешно загружены данные для {len(close_prices)} фьючерсов")
        
        if len(close_prices) < 2:
            logger.error("Недостаточно данных для анализа коинтеграции")
            return
                
        # Находим коинтегрированные пары
        pairs = analyzer.find_cointegrated_pairs(close_prices)
        
        # Выводим результаты
        logger.info("\nНайдены коинтегрированные пары:")
        for symbol1, symbol2, pvalue in pairs:
            logger.info(f"{symbol1} - {symbol2}: p-value = {pvalue:.4f}")
            
            # Рассчитываем коэффициент хеджирования и спред
            try:
                hedge_ratio = analyzer.calculate_hedge_ratio(
                    close_prices[symbol1],
                    close_prices[symbol2]
                )
                spread = analyzer.calculate_spread(
                    close_prices[symbol1],
                    close_prices[symbol2],
                    hedge_ratio
                )
                
                logger.info(f"Коэффициент хеджирования: {hedge_ratio:.4f}")
                logger.info(f"Средний спред: {spread.mean():.2f}")
                logger.info(f"Стандартное отклонение спреда: {spread.std():.2f}\n")
            except Exception as e:
                logger.error(f"Ошибка при расчете параметров для пары {symbol1}-{symbol2}: {e}")
                
    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа: {e}")

if __name__ == "__main__":
    main() 