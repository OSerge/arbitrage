import numpy as np
import pandas as pd
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
    # "GZM5", # Газпром обыкновенные
    "CHM5", # обыкновенные акции ПАО «Северсталь»
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
    """Класс для комплексного бэктестинга"""
    
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
        self.signals = None
        self.returns = None
        self.performance = None
        self.risk_analysis = None
    
    def generate_signals(self, z_score):
        """Генерация торговых сигналов"""
        signals = np.zeros(len(z_score))
        position = 0
        
        for i in range(self.lookback, len(z_score)):
            if np.isnan(z_score[i]):
                continue
                
            if position == 0:
                if z_score[i] > self.entry_threshold:
                    signals[i] = -1
                    position = -1
                elif z_score[i] < -self.entry_threshold:
                    signals[i] = 1
                    position = 1
            elif position != 0 and abs(z_score[i]) < self.exit_threshold:
                signals[i] = 0
                position = 0
            else:
                signals[i] = position
        
        self.signals = signals
        return self.signals
    
    def calculate_returns(self, beta):
        """Расчёт процентной доходности стратегии с учётом капитала."""
        if self.signals is None:
            raise ValueError("Сначала сгенерируйте торговые сигналы")

        s1 = self.series_1
        s2 = self.series_2
        n = len(s1)
        returns = np.zeros(n - 1)

        for i in range(1, n):
            pos_prev = self.signals[i-1]
            if pos_prev == 0:
                continue

            # Номинал портфеля (долларовая нейтральность)
            notional = abs(s1[i-1]) + abs(beta * s2[i-1])
            if notional == 0:
                continue

            pnl = pos_prev * ((s1[i] - s1[i-1]) - beta * (s2[i] - s2[i-1]))
            returns[i-1] = pnl / notional

            # Комиссии при смене позиции (две ножки)
            if i > 1 and self.signals[i-1] != self.signals[i-2]:
                returns[i-1] -= 2 * self.transaction_cost

        self.returns = returns
        return self.returns
    
    def performance_metrics(self):
        """Расчет метрик производительности для часовых данных"""
        if self.returns is None:
            raise ValueError("Сначала рассчитайте доходность")
        
        # Количество торговых часов в году (примерно)
        HOURS_IN_YEAR = 252 * 7

        n_hours = len(self.returns)

        if n_hours == 0:
            raise ValueError("Пустой массив доходностей")

        # Кумулятивная доходность (compound)
        equity_curve = np.cumprod(1 + self.returns)
        total_return = equity_curve[-1] - 1

        annual_factor = HOURS_IN_YEAR / n_hours
        annual_return = (1 + total_return) ** annual_factor - 1

        volatility = np.std(self.returns) * np.sqrt(HOURS_IN_YEAR)
        
        # Коэффициент Шарпа
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve / peak - 1
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
    
    def run_full_backtest(self, analyzer: DataAnalyzer):
        """Запуск полного цикла бэктестинга"""
        # Шаг 1: Проверка коинтеграции
        cointegration = analyzer.engle_granger_test(self.series_1, self.series_2)
        
        if not cointegration['is_cointegrated']:
            raise ValueError("Активы не коинтегрированы")
        
        # Шаг 2: Генерация сигналов
        z_score = analyzer.calculate_zscore(
            self.series_1,
            self.series_2,
            cointegration['beta'],
            self.lookback,
            cointegration['alpha']
        )
        self.generate_signals(z_score)
        
        # Шаг 3: Расчет доходности
        self.calculate_returns(cointegration['beta'])
        
        # Шаг 4: Расчет метрик
        self.performance = self.performance_metrics()
        
        # Шаг 5: Анализ рисков
        self.risk_analysis = self.monte_carlo_analysis()
        
        return {
            'cointegration': cointegration,
            'performance': self.performance,
            'risk_analysis': self.risk_analysis,
            'returns': self.returns
        }

def print_backtest_results(results, layout="single", pair_name=None):
    """Вывод результатов бэктеста с использованием rich
    
    Args:
        results: Результаты бэктеста
        layout: "single" для одной таблицы, "horizontal" для таблиц в строку
        pair_name: Название пары для отображения в заголовке
    """
    console = Console()
    
    if layout == "single":
        # Создаем одну общую таблицу
        title = "Результаты бэктеста"
        if pair_name:
            title += f" для пары {pair_name}"
        results_table = Table(title=title)
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
    results = backtester.run_full_backtest(analyzer=analyzer)
    
    pair_name = f"{pair[0]}-{pair[1]}"
    print_backtest_results(results, pair_name=pair_name)
