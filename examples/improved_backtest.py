"""
Пример использования улучшенной архитектуры для бэктестинга
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    DataManager, DataAnalyzer, ImprovedBacktester,
    RichResultsFormatter, generate_symbol_pairs,
    check_data_files_exist, setup_logging, DEFAULT_FUTURES
)

def main():
    setup_logging(level='INFO')
    
    test_symbols = ["SRM5", "SPM5"]
    
    symbol_pairs = generate_symbol_pairs(test_symbols)
    
    try:
        check_data_files_exist(symbol_pairs)
        
        data_manager = DataManager()
        analyzer = DataAnalyzer()
        formatter = RichResultsFormatter()
        
        for pair in symbol_pairs:
            symbol1, symbol2 = pair
            print(f"\n{'='*60}")
            print(f"Анализ пары: {symbol1} - {symbol2}")
            print(f"{'='*60}")
            
            df1 = data_manager.load_data_with_cache(symbol1)
            df2 = data_manager.load_data_with_cache(symbol2)
            
            merged_df = analyzer.join_pair(df1, df2)
            
            if merged_df.empty:
                print(f"Пустой датафрейм для пары {symbol1}-{symbol2}")
                continue
            
            backtester = ImprovedBacktester(
                series_1=merged_df['close_1'],
                series_2=merged_df['close_2']
            )
            
            try:
                results = backtester.run_backtest(analyzer)
                
                pair_name = f"{symbol1}-{symbol2}"
                formatter.format_single_result(results, pair_name)
                
            except ValueError as e:
                print(f"Ошибка при бэктесте пары {symbol1}-{symbol2}: {e}")
                continue
                
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что данные загружены в директорию ./data/")
        return
    
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return

if __name__ == "__main__":
    main() 