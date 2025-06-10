import os

from datetime import datetime, timedelta
import logging

from alor import AlorAPI
from data import DataManager, DataStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        "GDM5", # золото
        "SVM5", # серебро
    ]
    
    # Создаем экземпляр API
    api = AlorAPI(token=os.getenv('ALOR_TOKEN', ''))
    
    # Создаем менеджер данных
    data_manager = DataManager()
    data_storage = DataStorage()
    
    # Устанавливаем временной диапазон (последние 90 дней)
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
            
            # Получаем данные через API
            data = api.get_security_historical_data(
                symbol=symbol,
                from_time=from_time,
                to_time=to_time,
                tf=3600,  # часовой таймфрейм
                exchange='MOEX',
                instrument_group='RFUD'
            )
            
            # Сохраняем в CSV
            data_storage.save_data_to_csv(symbol, data)
            
            logger.info(f"Данные для {symbol} успешно сохранены")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для {symbol}: {e}")
    
    logger.info("Загрузка данных завершена")

if __name__ == '__main__':
    main() 