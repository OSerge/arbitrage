import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from typing import List
import pandas as pd
from alor.data import DataStorage
from alor.analysis import DataAnalyzer
from alor.backtest import BackTest

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Указываем два конкретных символа
    symbol1 = 'SRM5'
    symbol2 = 'VBM5'
    symbols = [symbol1, symbol2]

    # Проверяем наличие файлов
    for sym in symbols:
        if not os.path.exists(f'data/{sym}.csv'):
            logger.error(f"Нет файла data/{sym}.csv для анализа")
            return

    # Склеиваем данные по close
    all_data = DataAnalyzer.join_pairs(symbols)
    if all_data.empty:
        logger.error("Не удалось склеить данные для анализа")
        return
    logger.info(f"Склеено {len(all_data.columns)} инструментов, {len(all_data)} строк")

    # Проверяем коинтеграцию пары
    analyzer = DataAnalyzer()
    series1, series2 = all_data[symbol1], all_data[symbol2]
    coint_result = analyzer.check_cointegration(series1, series2)
    pval = coint_result['p_value']
    is_coint = coint_result['is_cointegrated']
    print(f"Пара: {symbol1} / {symbol2}")
    print(f"p-value: {pval:.5f}")
    print(f"Коинтегрирована: {'ДА' if is_coint else 'НЕТ'} (alpha={analyzer.alpha})")

    # Бэктестинг только для выбранной пары
    results = BackTest.backtest_pair(series1, series2)
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")

if __name__ == "__main__":
    main() 