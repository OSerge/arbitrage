import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
import logging
from alor.api import AlorAPI
from alor.data import DataManager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Список фьючерсов для загрузки
    futures = [
        "GKH5",
        "GZH5",
        "CHH5",
        "TTH5",
        "TNH5",
    ]
    
    # Создаем экземпляр API
    api = AlorAPI(token=os.getenv('ALOR_TOKEN', ''))
    
    # Создаем менеджер данных
    data_manager = DataManager(api)
    
    # Устанавливаем временной диапазон (последние 30 дней)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    
    # Конвертируем в timestamp
    from_time = int(start_time.timestamp())
    to_time = int(end_time.timestamp())
    
    logger.info(f"Начинаем загрузку данных с {start_time} по {end_time}")
    
    # Загружаем данные по каждому фьючерсу
    for symbol in futures:
        try:
            logger.info(f"Загрузка данных для {symbol}")
            
            # Получаем данные
            data = data_manager.get_data(symbol, from_time, to_time)
            
            # Сохраняем в CSV
            data_manager.save_data_to_csv(symbol, data)
            
            logger.info(f"Данные для {symbol} успешно сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для {symbol}: {e}")
    
    # Очищаем кэш
    data_manager.clear_cache()
    logger.info("Загрузка данных завершена")

if __name__ == '__main__':
    main() 