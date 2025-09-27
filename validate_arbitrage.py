#!/usr/bin/env python3
"""
Скрипт для валидации математической корректности арбитражных операций
"""

import sys
import logging
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from core.data import DataManager
from core.analysis import DataAnalyzer
from core.backtest import Backtester
from core.validation import ArbitrageValidator, print_validation_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_arbitrage_system():
    """Основная функция валидации системы арбитража"""
    
    print("🔍 Запуск валидации системы статистического арбитража")
    print("=" * 60)
    
    # Инициализация компонентов
    data_manager = DataManager()
    analyzer = DataAnalyzer()
    validator = ArbitrageValidator()
    
    try:
        # Загрузка тестовых данных (Сбербанк обыкновенные и привилегированные)
        print("\n📊 Загрузка тестовых данных...")
        df1 = data_manager.storage.load_data('SRM5')  # Сбербанк обыкновенные
        df2 = data_manager.storage.load_data('SPM5')  # Сбербанк привилегированные
        
        print(f"✅ Загружено: SRM5 - {len(df1)} записей, SPM5 - {len(df2)} записей")
        
        # Объединение данных
        print("\n🔗 Объединение временных рядов...")
        merged_df = analyzer.join_pair(df1, df2)
        print(f"✅ Объединено: {len(merged_df)} записей")
        
        if len(merged_df) < 100:
            print("❌ Недостаточно данных для валидации")
            return
        
        # Извлечение рядов цен
        series1 = merged_df['close_1']
        series2 = merged_df['close_2']
        
        # Тест коинтеграции
        print("\n📈 Тестирование коинтеграции...")
        coint_result = analyzer.check_cointegration(series1, series2)
        print(f"✅ P-value: {coint_result['p_value']:.6f}")
        print(f"✅ Коинтегрированы: {coint_result['is_cointegrated']}")
        print(f"✅ Beta: {coint_result['beta']:.6f}, Alpha: {coint_result['alpha']:.6f}")
        
        if not coint_result['is_cointegrated']:
            print("⚠️  Активы не коинтегрированы, но продолжаем валидацию...")
        
        # Создание бэктестера
        print("\n🎯 Создание бэктестера...")
        backtester = Backtester(
            series_1=series1,
            series_2=series2,
            lookback=60,
            entry_threshold=2.0,
            exit_threshold=0.5,
            broker_commission=1.0,
            exchange_commission_rate=0.02,  # 2 базисных пункта
            min_estimation_window=200
        )
        
        # Запуск бэктеста
        print("\n⚡ Запуск бэктеста...")
        try:
            results = backtester.run_full_backtest(analyzer)
            print("✅ Бэктест завершен успешно")
        except ValueError as e:
            if "не коинтегрированы" in str(e):
                print("⚠️  Активы не коинтегрированы на всех интервалах")
                print("   Используем простой бэктест для валидации...")
                
                # Простой бэктест для валидации
                z_score = analyzer.calculate_zscore(
                    series1, series2, coint_result['beta'], 60, coint_result['alpha']
                )
                signals = backtester.generate_signals(z_score)
                returns = backtester.calculate_returns(coint_result['beta'])
                performance = backtester.performance_metrics()
                risk_analysis = backtester.monte_carlo_analysis()
                
                results = {
                    'cointegration': coint_result,
                    'performance': performance,
                    'risk_analysis': risk_analysis,
                    'returns': returns
                }
            else:
                raise
        
        # Валидация результатов
        print("\n🔍 Валидация математической корректности...")
        validation_results = validator.full_validation(results, series1, series2)
        
        print_validation_results(validation_results)
        
        # Дополнительные проверки
        print("\n📊 Дополнительные проверки:")
        
        # Проверка диапазонов значений
        if abs(results['performance']['sharpe_ratio']) > 10:
            print("⚠️  Коэффициент Шарпа подозрительно высокий")
        else:
            print("✅ Коэффициент Шарпа в разумных пределах")
        
        if abs(results['performance']['annual_return']) > 5:  # 500%
            print("⚠️  Годовая доходность подозрительно высокая")
        else:
            print("✅ Годовая доходность в разумных пределах")
        
        if results['performance']['max_drawdown'] < -0.5:  # -50%
            print("⚠️  Максимальная просадка очень большая")
        else:
            print("✅ Максимальная просадка приемлемая")
        
        # Проверка консистентности данных
        returns_array = results['returns']
        if len(returns_array) > 0:
            if np.any(np.isinf(returns_array)) or np.any(np.isnan(returns_array)):
                print("❌ Обнаружены некорректные значения в доходности")
            else:
                print("✅ Доходность не содержит некорректных значений")
        
        print("\n" + "=" * 60)
        print("🎉 Валидация завершена!")
        
    except FileNotFoundError as e:
        print(f"❌ Файл данных не найден: {e}")
        print("   Убедитесь, что данные загружены в папку ./data/")
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
        logger.exception("Подробности ошибки:")


if __name__ == "__main__":
    import numpy as np  # Импортируем numpy для проверок
    validate_arbitrage_system()