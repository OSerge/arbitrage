import numpy as np
import pandas as pd
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel

import logging
import os
from itertools import combinations

from core.data import DataManager
from core.analysis import DataAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRADING_HOURS_PER_DAY = 16   # Торгуем 16 часов в день
TRADING_DAYS_PER_YEAR = 252  # Среднее количество торговых дней в году
HOURS_IN_YEAR = TRADING_HOURS_PER_DAY * TRADING_DAYS_PER_YEAR

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
                 exit_threshold=0.5, broker_commission=1.0, exchange_commission_rate=0.02,
                 min_estimation_window=512):
        """
        Инициализация параметров бэктестера
        
        :param series_1: Временной ряд цен первого актива
        :param series_2: Временной ряд цен второго актива
        :param lookback: Окно наблюдения для расчета параметров
        :param entry_threshold: Порог входа в сделку (в стандартных отклонениях)
        :param exit_threshold: Порог выхода из сделки
        :param broker_commission: Комиссия брокера за сделку в рублях (фиксированная)
        :param exchange_commission_rate: Комиссия биржи в процентах от суммы сделки
        :param min_estimation_window: Минимальное окно для оценки коинтеграции (часов)
        """
        self.series_1 = np.array(series_1)
        self.series_2 = np.array(series_2)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.broker_commission = broker_commission
        self.exchange_commission_rate = exchange_commission_rate
        self.min_estimation_window = min_estimation_window
        
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

            notional = abs(s1[i-1]) + abs(beta * s2[i-1])
            if notional == 0:
                continue

            # PnL от изменения спреда
            pnl = pos_prev * ((s1[i] - s1[i-1]) - beta * (s2[i] - s2[i-1]))
            returns[i-1] = pnl / notional

            # Комиссии применяются только при изменении позиции
            if i > 1 and self.signals[i-1] != self.signals[i-2]:
                # Фиксированная комиссия брокера за сделку
                broker_cost = self.broker_commission 
                # Добавляем НДС к комиссии брокера (20%)
                broker_cost_with_vat = broker_cost * 1.2
                
                # комиссия биржи от стоимости сделки 
                exchange_cost = notional * self.exchange_commission_rate / 100 
                
                total_commission = broker_cost_with_vat + exchange_cost
                returns[i-1] -= total_commission / notional

        self.returns = returns
        return self.returns
    
    def performance_metrics(self):
        """Расчет метрик производительности для часовых данных"""
        if self.returns is None:
            raise ValueError("Сначала рассчитайте доходность")
        
        n_hours = len(self.returns)
        logger.info(f"Количество часов: {n_hours}")

        if n_hours == 0:
            raise ValueError("Пустой массив доходностей")

        # Кумулятивная доходность (compound)
        equity_curve = np.cumprod(1 + self.returns)
        total_return = equity_curve[-1] - 1

        # Годовой коэффициент с учётом 15 торговых часов в день
        annual_factor = HOURS_IN_YEAR / n_hours
        annual_return = (1 + total_return) ** annual_factor - 1

        # Волатильность (часовая) годовая
        volatility = np.std(self.returns) * np.sqrt(HOURS_IN_YEAR)
        
        # Коэффициент Шарпа
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve / peak - 1
        max_drawdown = np.min(drawdown)
        
        time_info = {
            'total_hours': n_hours,
            'trading_days': n_hours / TRADING_HOURS_PER_DAY,
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
        """
        Запуск walk-forward бэктестинга без look-ahead bias
        
        Коэффициенты коинтеграции пересчитываются на каждом шаге, 
        используя только прошлые данные.
        """
        n = len(self.series_1)
        signals = np.zeros(n)
        
        current_position = 0
        current_coint_result = None
        recompute_frequency = 168  # Пересчитываем коинтеграцию раз в неделю (168 часов)
        
        logger.info(f"Запуск бэктеста на {n} точках")
        
        current_coint_result = self._estimate_cointegration_at_point(analyzer, self.min_estimation_window)
        
        for i in range(self.min_estimation_window, n):
            # Периодически пересчитываем коинтеграцию используя только данные до момента i
            if i % recompute_frequency == 0:
                new_coint_result = self._estimate_cointegration_at_point(analyzer, i)
                if new_coint_result is not None:
                    current_coint_result = new_coint_result
            
            if (current_coint_result is None or 
                not current_coint_result.get('is_cointegrated', False)):
                signals[i] = 0
                current_position = 0
                continue
            
            z_score = self._calculate_current_zscore(i, current_coint_result)
            
            new_signal = self._generate_signal_at_point(z_score, current_position)
            signals[i] = new_signal
            current_position = new_signal
        
        self.signals = signals
        
        if current_coint_result is None or not current_coint_result.get('is_cointegrated', False):
            raise ValueError("Активы не коинтегрированы ни на одном интервале")
        
        self.calculate_returns(current_coint_result['beta'])
        self.performance = self.performance_metrics()
        self.risk_analysis = self.monte_carlo_analysis()
        
        return {
            'cointegration': current_coint_result,
            'performance': self.performance,
            'risk_analysis': self.risk_analysis,
            'returns': self.returns
        }
    
    def _estimate_cointegration_at_point(self, analyzer, end_idx):
        """Оценка коинтеграции на данных до определенной точки"""
        try:
            # Используем только данные до текущего момента
            estimation_data_1 = pd.Series(self.series_1[:end_idx])
            estimation_data_2 = pd.Series(self.series_2[:end_idx])
            
            return analyzer.engle_granger_test(estimation_data_1, estimation_data_2)
        except Exception as e:
            logger.warning(f"Ошибка при тесте коинтеграции на индексе {end_idx}: {e}")
            return None
    
    def _calculate_current_zscore(self, current_idx, coint_result):
        """Расчет Z-score для текущего момента без look-ahead bias"""
        if current_idx < self.lookback:
            return np.nan
        
        # Используем только данные до текущего момента для расчета статистик
        lookback_start = current_idx - self.lookback
        s1_window = self.series_1[lookback_start:current_idx]
        s2_window = self.series_2[lookback_start:current_idx]
        
        # Спред на историческом окне
        spread_window = s1_window - coint_result['beta'] * s2_window - coint_result['alpha']
        spread_mean = np.mean(spread_window)
        spread_std = np.std(spread_window)
        
        if spread_std == 0:
            return np.nan
        
        current_spread = (self.series_1[current_idx] - 
                         coint_result['beta'] * self.series_2[current_idx] - 
                         coint_result['alpha'])
        
        return (current_spread - spread_mean) / spread_std
    
    def _generate_signal_at_point(self, z_score, current_position):
        """Генерация торгового сигнала для текущей точки"""
        if np.isnan(z_score):
            return current_position
        
        if current_position == 0:
            if z_score > self.entry_threshold:
                return -1  # Короткая позиция
            elif z_score < -self.entry_threshold:
                return 1   # Длинная позиция
        else:
            if abs(z_score) < self.exit_threshold:
                return 0   # Закрытие позиции
        
        return current_position

def print_backtest_results(results, layout="single", pair_name=None):
    """Вывод результатов бэктеста с использованием rich
    
    Args:
        results: Результаты бэктеста
        layout: "single" для одной таблицы, "horizontal" для таблиц в строку
        pair_name: Название пары для отображения в заголовке
    """
    console = Console()
    
    if layout == "single":
        title = "Результаты бэктеста"
        if pair_name:
            title += f" для пары {pair_name}"
        results_table = Table(title=title)
        results_table.add_column("Метрика", style="cyan")
        results_table.add_column("Значение", style="green")
        
        results_table.add_row("[bold blue]Тест коинтеграции[/bold blue]", "")
        for param, value in results['cointegration'].items():
            if param != 'critical_values':
                results_table.add_row(
                    param,
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
        
        results_table.add_row("[bold blue]Временные интервалы[/bold blue]", "")
        time_info = results['performance']['time_info']
        results_table.add_row("Всего часов", f"{time_info['total_hours']:.0f}")
        results_table.add_row("Торговых дней", f"{time_info['trading_days']:.1f}")
        results_table.add_row("Годовой коэффициент", f"{time_info['annual_factor']:.2f}")
        
        results_table.add_row("[bold blue]Метрики производительности[/bold blue]", "")
        for metric, value in results['performance'].items():
            if metric != 'time_info':
                results_table.add_row(
                    metric,
                    f"{value:.2%}" if metric != 'sharpe_ratio' else f"{value:.2f}"
                )
        
        results_table.add_row("[bold blue]Анализ рисков[/bold blue]", "")
        results_table.add_row("VaR 95%", f"{results['risk_analysis']['var_95']:.2%}")
        results_table.add_row("CVaR 95%", f"{results['risk_analysis']['cvar_95']:.2%}")
        
        console.print(results_table)
        
    elif layout == "horizontal":
        tables = []
        
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
        
        time_table = Table(title="Временные интервалы")
        time_table.add_column("Метрика", style="cyan")
        time_table.add_column("Значение", style="green")
        time_info = results['performance']['time_info']
        time_table.add_row("Всего часов", f"{time_info['total_hours']:.0f}")
        time_table.add_row("Торговых дней", f"{time_info['trading_days']:.1f}")
        time_table.add_row("Годовой коэффициент", f"{time_info['annual_factor']:.2f}")
        tables.append(time_table)
        
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
        
        risk_table = Table(title="Анализ рисков")
        risk_table.add_column("Метрика", style="cyan")
        risk_table.add_column("Значение", style="green")
        risk_table.add_row("VaR 95%", f"{results['risk_analysis']['var_95']:.2%}")
        risk_table.add_row("CVaR 95%", f"{results['risk_analysis']['cvar_95']:.2%}")
        tables.append(risk_table)
        
        console.print(Panel.fit(
            Group(*tables),
            title="Результаты бэктеста",
            border_style="blue"
        ))

def print_summary_results(results_dict):
    """Вывод сводной таблицы результатов бэктеста для нескольких пар
    
    Args:
        results_dict: Словарь с результатами бэктеста для каждой пары
    """
    console = Console()
    
    summary_table = Table(title="Сводные результаты бэктеста")
    summary_table.add_column("Пара", style="cyan")
    summary_table.add_column("Коэффициент Шарпа", style="green")
    summary_table.add_column("Годовая доходность", style="green")
    summary_table.add_column("Максимальная просадка", style="green")
    summary_table.add_column("VaR 95%", style="green")
    summary_table.add_column("CVaR 95%", style="green")
    
    for pair, results in results_dict.items():
        summary_table.add_row(
            pair,
            f"{results['performance']['sharpe_ratio']:.2f}",
            f"{results['performance']['annual_return']:.2%}",
            f"{results['performance']['max_drawdown']:.2%}",
            f"{results['risk_analysis']['var_95']:.2%}",
            f"{results['risk_analysis']['cvar_95']:.2%}"
        )
    
    console.print(summary_table)

if __name__ == "__main__":
    data_manager = DataManager()
    analyzer = DataAnalyzer()

    symbol_pairs = generate_symbol_pairs(futures)
    check_files(symbol_pairs)

    all_results = {}
    
    for pair in symbol_pairs:
        try:
            df1 = data_manager.load_data_with_cache(pair[0])
            df2 = data_manager.load_data_with_cache(pair[1])

            merged_df = analyzer.join_pair(df1, df2)
            if merged_df.empty:
                logger.warning(f"Пустой датафрейм для пары {pair[0]}-{pair[1]}, пропускаем")
                continue
            
            series_1, series_2 = merged_df['close_1'], merged_df['close_2']
            
            # Предварительная проверка коинтеграции
            preliminary_coint = analyzer.engle_granger_test(series_1, series_2)
            if not preliminary_coint['is_cointegrated']:
                logger.warning(f"Пара {pair[0]}-{pair[1]} не коинтегрирована, пропускаем")
                continue
            
            backtester = Backtester(series_1, series_2, lookback=60, 
                                   entry_threshold=2.0, exit_threshold=0.5)
            results = backtester.run_full_backtest(analyzer=analyzer)
            
            pair_name = f"{pair[0]}-{pair[1]}"
            all_results[pair_name] = results
            
            print_backtest_results(results, pair_name=pair_name)
            print()
            
        except Exception as e:
            logger.error(f"Ошибка при обработке пары {pair[0]}-{pair[1]}: {e}")
            continue
    
    if all_results:
        print("\n" + "="*80)
        print_summary_results(all_results)
