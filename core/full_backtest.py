import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel

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
    # "GKM5", # Обыкновенные акции ПАО «ГМК «Норильский никель»
    "GZM5", # Газпром обыкновенные
    # "CHM5", # обыкновенные акции ПАО «Северсталь»
    "TTM5", # Татнефть
    # "TNM5", # Транснефть
    # "RNM5", # Роснефть
    # "LKM5", # Лукойл
    # "SRM5", # обыкновенные акции ПАО Сбербанк
    # "SPM5", # привилег. акции ПАО Сбербанк
    # "VBM5", # ВТБ
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
        

class Backtester:
    """Класс для комплексного бэктестинга парного статистического арбитража"""
    
    def __init__(self, series_1, series_2, lookback=60, entry_threshold=2.0,
                 exit_threshold=0.5, transaction_cost=0.001):
        """
        Инициализация параметров бэктестера
        
        :param series_1: Временной ряд цен первого актива
        :param series_2: Временной ряд цен второго актива
        :param lookback: Окно наблюдения для расчета параметров
        :param entry_threshold: Порог входа в сделку (в стандартных отклонениях)
        :param exit_threshold: Порог выхода из сделки
        :param transaction_cost: Транзакционные издержки
        """
        self.series_1 = np.array(series_1)
        self.series_2 = np.array(series_2)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        
        # Результаты тестов
        self.stationarity = None
        self.cointegration = None
        self.signals = None
        self.z_score = None
        self.returns = None
        self.performance = None
        self.risk_analysis = None
    
    def adf_test(self, series):
        """Расширенный тест Дики-Фуллера для проверки стационарности"""
        result = adfuller(series)
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def engle_granger_test(self):
        """Тест коинтеграции Энгла-Грейнджера с использованием statsmodels"""
        # Выполняем тест коинтеграции
        score, pvalue, _ = coint(self.series_1, self.series_2)
        
        # Оцениваем параметры коинтеграции (МНК)
        X = np.column_stack([np.ones(len(self.series_1)), self.series_2])
        beta = np.linalg.lstsq(X, self.series_1, rcond=None)[0]
        
        return {
            'alpha': beta[0],
            'beta': beta[1],
            'adf_statistic': score,
            'p_value': pvalue,
            'is_cointegrated': pvalue < 0.05,
        }
    
    def calculate_zscore(self):
        """Расчет Z-score для генерации торговых сигналов"""
        spread = self.series_1 - self.cointegration['beta'] * self.series_2
        spread_series = pd.Series(spread)
        
        spread_mean = spread_series.rolling(self.lookback).mean()
        spread_std = spread_series.rolling(self.lookback).std()
        
        self.z_score = (spread - spread_mean) / spread_std
        return self.z_score
    
    def generate_signals(self):
        """Генерация торговых сигналов"""
        signals = np.zeros(len(self.z_score))
        position = 0
        
        for i in range(self.lookback, len(self.z_score)):
            if np.isnan(self.z_score[i]):
                continue
                
            if position == 0:
                if self.z_score[i] > self.entry_threshold:
                    signals[i] = -1
                    position = -1
                elif self.z_score[i] < -self.entry_threshold:
                    signals[i] = 1
                    position = 1
            elif position != 0 and abs(self.z_score[i]) < self.exit_threshold:
                signals[i] = 0
                position = 0
            else:
                signals[i] = position
        
        self.signals = signals
        return self.signals
    
    def calculate_returns(self):
        """Расчет доходности стратегии"""
        if self.signals is None:
            raise ValueError("Сначала сгенерируйте торговые сигналы")
            
        returns_1 = np.diff(np.log(self.series_1))
        returns_2 = np.diff(np.log(self.series_2))
        strategy_returns = np.zeros(len(returns_1))
        beta = self.cointegration['beta']
        
        for i in range(1, len(self.signals)):
            if self.signals[i-1] == 1:
                strategy_returns[i-1] = returns_1[i-1] - beta * returns_2[i-1]
            elif self.signals[i-1] == -1:
                strategy_returns[i-1] = -returns_1[i-1] + beta * returns_2[i-1]
            
            if i > 1 and self.signals[i-1] != self.signals[i-2]:
                strategy_returns[i-1] -= 2 * self.transaction_cost
        
        self.returns = strategy_returns
        return self.returns
    
    def performance_metrics(self):
        """Расчет метрик производительности для часовых данных"""
        if self.returns is None:
            raise ValueError("Сначала рассчитайте доходность")
        
        # Количество часов в торговом году (252 дня * 7 часов)
        HOURS_IN_YEAR = 252 * 7
        
        # Получаем реальное количество часовых интервалов
        n_hours = len(self.returns)
        
        # Расчет годового коэффициента
        annual_factor = HOURS_IN_YEAR / n_hours
        
        total_return = np.sum(self.returns)
        annual_return = total_return * annual_factor
        volatility = np.std(self.returns) * np.sqrt(annual_factor)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Просадка
        cumulative = np.cumsum(self.returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        max_drawdown = np.min(drawdown)
        
        # Дополнительная информация о временных интервалах
        time_info = {
            'total_hours': n_hours,
            'trading_days': n_hours / 7,  # примерное количество торговых дней
            'annual_factor': annual_factor,
            'hours_in_year': HOURS_IN_YEAR
        }
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'time_info': time_info
        }
    
    def monte_carlo_analysis(self, n_simulations=1000):
        """Анализ рисков методом Монте-Карло"""
        returns = self.returns
        mean = np.mean(returns)
        std = np.std(returns)
        
        simulated_returns = np.random.normal(mean, std, (n_simulations, len(returns)))
        cumulative_returns = np.cumsum(simulated_returns, axis=1)
        
        max_drawdowns = np.array([
            np.min(cumulative_returns[i] - np.maximum.accumulate(cumulative_returns[i])) 
            for i in range(n_simulations)
        ])
        
        return {
            'var_95': np.percentile(cumulative_returns[:, -1], 5),
            'cvar_95': np.mean(cumulative_returns[:, -1][cumulative_returns[:, -1] <= np.percentile(cumulative_returns[:, -1], 5)]),
            'max_drawdown_dist': max_drawdowns
        }
    
    def run_full_backtest(self):
        """Запуск полного цикла бэктестинга"""
        # Шаг 1: Проверка стационарности
        self.stationarity = {
            'asset_1': self.adf_test(self.series_1),
            'asset_2': self.adf_test(self.series_2)
        }
        
        # Шаг 2: Проверка коинтеграции
        self.cointegration = self.engle_granger_test()
        
        if not self.cointegration['is_cointegrated']:
            raise ValueError("Активы не коинтегрированы")
        
        # Шаг 3: Генерация сигналов
        self.calculate_zscore()
        self.generate_signals()
        
        # Шаг 4: Расчет доходности
        self.calculate_returns()
        
        # Шаг 5: Расчет метрик
        self.performance = self.performance_metrics()
        
        # Шаг 6: Анализ рисков
        self.risk_analysis = self.monte_carlo_analysis()
        
        return {
            'stationarity': self.stationarity,
            'cointegration': self.cointegration,
            'performance': self.performance,
            'risk_analysis': self.risk_analysis,
            'returns': self.returns
        }

def print_backtest_results(results, layout="single"):
    """Вывод результатов бэктеста с использованием rich
    
    Args:
        results: Результаты бэктеста
        layout: "single" для одной таблицы, "horizontal" для таблиц в строку
    """
    console = Console()
    
    if layout == "single":
        # Создаем одну общую таблицу
        results_table = Table(title="Результаты бэктеста")
        results_table.add_column("Метрика", style="cyan")
        results_table.add_column("Значение", style="green")
        
        # Добавляем разделитель для коинтеграции
        results_table.add_row("[bold blue]Тест коинтеграции[/bold blue]", "")
        for param, value in results['cointegration'].items():
            if param != 'critical_values':
                results_table.add_row(
                    param,
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
        
        # Добавляем разделитель для временных интервалов
        results_table.add_row("[bold blue]Временные интервалы[/bold blue]", "")
        time_info = results['performance']['time_info']
        results_table.add_row("Всего часов", f"{time_info['total_hours']:.0f}")
        results_table.add_row("Торговых дней", f"{time_info['trading_days']:.1f}")
        results_table.add_row("Годовой коэффициент", f"{time_info['annual_factor']:.2f}")
        
        # Добавляем разделитель для метрик производительности
        results_table.add_row("[bold blue]Метрики производительности[/bold blue]", "")
        for metric, value in results['performance'].items():
            if metric != 'time_info':
                results_table.add_row(
                    metric,
                    f"{value:.2%}" if metric != 'sharpe_ratio' else f"{value:.2f}"
                )
        
        # Добавляем разделитель для анализа рисков
        results_table.add_row("[bold blue]Анализ рисков[/bold blue]", "")
        results_table.add_row("VaR 95%", f"{results['risk_analysis']['var_95']:.2%}")
        results_table.add_row("CVaR 95%", f"{results['risk_analysis']['cvar_95']:.2%}")
        
        console.print(results_table)
        
    elif layout == "horizontal":
        # Создаем таблицы для каждого раздела
        tables = []
        
        # Таблица коинтеграции
        cointegration_table = Table(title="Тест коинтеграции")
        cointegration_table.add_column("Параметр", style="cyan")
        cointegration_table.add_column("Значение", style="green")
        for param, value in results['cointegration'].items():
            if param != 'critical_values':
                cointegration_table.add_row(
                    param,
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
        tables.append(cointegration_table)
        
        # Таблица временных интервалов
        time_table = Table(title="Временные интервалы")
        time_table.add_column("Метрика", style="cyan")
        time_table.add_column("Значение", style="green")
        time_info = results['performance']['time_info']
        time_table.add_row("Всего часов", f"{time_info['total_hours']:.0f}")
        time_table.add_row("Торговых дней", f"{time_info['trading_days']:.1f}")
        time_table.add_row("Годовой коэффициент", f"{time_info['annual_factor']:.2f}")
        tables.append(time_table)
        
        # Таблица метрик производительности
        performance_table = Table(title="Метрики производительности")
        performance_table.add_column("Метрика", style="cyan")
        performance_table.add_column("Значение", style="green")
        for metric, value in results['performance'].items():
            if metric != 'time_info':
                performance_table.add_row(
                    metric,
                    f"{value:.2%}" if metric != 'sharpe_ratio' else f"{value:.2f}"
                )
        tables.append(performance_table)
        
        # Таблица анализа рисков
        risk_table = Table(title="Анализ рисков")
        risk_table.add_column("Метрика", style="cyan")
        risk_table.add_column("Значение", style="green")
        risk_table.add_row("VaR 95%", f"{results['risk_analysis']['var_95']:.2%}")
        risk_table.add_row("CVaR 95%", f"{results['risk_analysis']['cvar_95']:.2%}")
        tables.append(risk_table)
        
        # Выводим все таблицы в одну строку
        console.print(Panel.fit(
            Group(*tables),
            title="Результаты бэктеста",
            border_style="blue"
        ))

if __name__ == "__main__":

    series_1, series_2 = pd.Series, pd.Series

    data_manager = DataManager()
    analyzer = DataAnalyzer()

    symbol_pairs = generate_symbol_pairs(futures)
    check_files(symbol_pairs)

    for pair in symbol_pairs:
        df1 = data_manager.storage.load_data_from_csv(pair[0])
        df2 = data_manager.storage.load_data_from_csv(pair[1])

        merged_df = analyzer.join_pair(df1, df2)
        if merged_df.empty:
            raise ValueError("Пустой датафрейм для анализа")
        
        series_1, series_2 = merged_df['close_1'], merged_df['close_2']
    
    # Инициализация и запуск бэктеста
    backtester = Backtester(series_1, series_2, lookback=60)
    results = backtester.run_full_backtest()
    
    # Вывод результатов
    print_backtest_results(results)
