import logging
import os
from itertools import combinations
from typing import List

from data import DataManager
from analysis import DataAnalyzer
from backtest import BackTest, ArbitrageBacktest


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def generate_symbol_pairs(symbols: list[str]) -> list[tuple]:
    return list(combinations(symbols, 2))

def check_files(symbol_pairs: list[tuple]):
    for pair in symbol_pairs:
        if not os.path.exists(f'./data/{pair[0]}.csv'):
            raise FileNotFoundError(f"Файл ./data/{pair[0]}.csv не существует")
        if not os.path.exists(f'./data/{pair[1]}.csv'):
            raise FileNotFoundError(f"Файл ./data/{pair[1]}.csv не существует")

def join_all_securities():
    ...

def main():

    data_manager = DataManager()
    analyzer = DataAnalyzer()

    symbol_pairs = generate_symbol_pairs(futures)
    check_files(symbol_pairs)

    for pair in symbol_pairs:
        df1 = data_manager.storage.load_data_from_csv(pair[0])
        df2 = data_manager.storage.load_data_from_csv(pair[1])

        merged_df = analyzer.join_pair(df1, df2)
        if merged_df.empty:
            logger.error("Не удалось склеить данные для анализа")
            return
        
        # logger.info(f"Склеено {len(merged_df.columns)} инструментов, {len(merged_df)} строк")

        # Проверяем коинтеграцию пары
        series1, series2 = merged_df['close_1'], merged_df['close_2']
        coint_result = analyzer.check_cointegration(series1, series2)
        pval = coint_result['p_value']
        is_coint = coint_result['is_cointegrated']

        if is_coint:
            print(f"\nПара: {pair}")
            print(f"p-value: {pval:.5f}")
            print(f"Коинтегрирована: {'ДА' if is_coint else 'НЕТ'} (alpha={analyzer.alpha})")
            # print(f"Стационар. спреда: {'ДА' if coint_result['is_spread_stationary'] else 'НЕТ'}")

            # Бэктестинг только для выбранной пары
            # results = BackTest.backtest_pair(series1, series2)
            # print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            # print(f"Max Drawdown: {results['max_drawdown']:.2%}")

            bt = ArbitrageBacktest(
                series1, 
                series2, 
                entry_threshold=2.0, 
                exit_threshold=0.5,
                capital=100000, 
                commission=0.01
                )
            df = bt.run()
            metrics = bt.metrics()
            print("=== Результаты бэктеста ===")
            print(f"Total Return:     {metrics['total_return']:.2%}")
            print(f"Annual Return:    {metrics['annual_return']:.2%}")
            print(f"Sharpe Ratio:     {metrics['sharpe']:.2f}")
            print(f"Max Drawdown:     {metrics['max_drawdown']:.2%}")

if __name__ == "__main__":
    main() 