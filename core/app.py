import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from alor.api import AlorAPI
from alor.data import DataManager
from alor.analysis import CointAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(force_download: bool = False):
    """Основная функция для запуска анализа
    
    Args:
        force_download (bool): Если True, то перезагружает все данные, даже если они уже есть
    """
    # Инициализация
    api = AlorAPI(token=os.getenv('ALOR_TOKEN', ''))
    data_manager = DataManager(api)
    analyzer = CointAnalyzer(alpha=0.05)
    
    # Список фьючерсов для анализа
    futures = [
        "RIH5",     # Индекс РТС
        "SiH5",     # USD/RUB
        "BRH5",     # Нефть Brent
        "SBRFH5",   # Сбербанк
        "GAZPH5",   # Газпром
        "GOLDH5",   # Золото
        "LKOHH5",   # Лукойл
        "RTKMH5",   # Ростелеком
        "GMKNH5",   # Норильский никель
        "ROSNH5",   # Роснефть
        "VTBRH5",   # ВТБ
        "TATNH5",   # Татнефть
        "MXH5",     # Индекс IMOEX
        "HYDRH5",   # РусГидро
        "PLZLH5",   # Полюс Золото
        "SNGSH5",   # Сургутнефтегаз
        "ALRSH5",   # Алроса
        "MAGNH5",   # ММК
        "NVTKH5",   # Новатэк
        "MTLRH5",   # Мечел
    ]
    
    # Устанавливаем временной диапазон (последние 30 дней)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    # Конвертируем в timestamp
    from_time = int(start_time.timestamp())
    to_time = int(end_time.timestamp())
    
    logger.info(f"Начинаем загрузку данных с {start_time} по {end_time}")
    
    # Загрузка данных
    all_data = pd.DataFrame()
    for symbol in futures:
        try:
            logger.info(f"Загрузка данных для {symbol}")
            
            # Получаем данные
            data = data_manager.get_data(symbol, from_time, to_time)
            
            # Добавляем в общий датафрейм
            if 'close' in data.columns:
                all_data[symbol] = data['close']
                logger.info(f"Данные для {symbol} успешно загружены")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для {symbol}: {e}")
    
    if all_data.empty:
        logger.error("Нет доступных данных для анализа")
        return
    
    # Поиск коинтегрированных пар
    logger.info("Поиск коинтегрированных пар...")
    cointegrated_pairs = analyzer.find_cointegrated_pairs(all_data)
    logger.info(f"Найдено {len(cointegrated_pairs)} коинтегрированных пар")
    
    # Анализ и бэктестинг найденных пар
    for pair in cointegrated_pairs:
        col1, col2 = pair['pair']
        series1, series2 = all_data[col1], all_data[col2]
        
        # Визуализация
        analyzer.plot_cointegrated_pair(series1, series2, col1, col2, pair['results'])
        
        # Бэктестинг
        backtest_results = analyzer.backtest_pair(series1, series2)
        print(f"\nРезультаты бэктестинга для пары {col1}/{col2}:")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ коинтеграции фьючерсов')
    parser.add_argument('--force-download', action='store_true',
                      help='Принудительно перезагрузить все данные')
    
    args = parser.parse_args()
    main(force_download=args.force_download) 