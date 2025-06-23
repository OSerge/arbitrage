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
                 min_estimation_window=512, enable_slippage=False, enable_bid_ask_spread=False,
                 slippage_rate=0.02, bid_ask_spread_rate=0.05, 
                 ohlcv_data_1=None, ohlcv_data_2=None, recompute_frequency=16):
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
        :param enable_slippage: Включить учет проскальзывания
        :param enable_bid_ask_spread: Включить учет bid-ask спреда
        :param slippage_rate: Проскальзывание в процентах (по умолчанию 0.02%)
        :param bid_ask_spread_rate: Bid-ask спред в процентах (по умолчанию 0.05%)
        :param ohlcv_data_1: OHLCV данные первого актива (DataFrame с колонками open, high, low, close, volume)
        :param ohlcv_data_2: OHLCV данные второго актива (DataFrame с колонками open, high, low, close, volume)
        :param recompute_frequency: Частота пересчета коинтеграции в часах (по умолчанию 16)
        """
        self.series_1 = np.array(series_1)
        self.series_2 = np.array(series_2)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.broker_commission = broker_commission
        self.exchange_commission_rate = exchange_commission_rate
        self.min_estimation_window = min_estimation_window
        
        # Параметры для учета реалистичных торговых условий
        self.enable_slippage = enable_slippage
        self.enable_bid_ask_spread = enable_bid_ask_spread
        self.slippage_rate = slippage_rate / 100  # Перевод в доли
        self.bid_ask_spread_rate = bid_ask_spread_rate / 100  
        self.ohlcv_data_1 = ohlcv_data_1
        self.ohlcv_data_2 = ohlcv_data_2
        self.recompute_frequency = recompute_frequency
        
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

            # Цены для расчета PnL
            if self.enable_slippage or self.enable_bid_ask_spread:
                if i > 1 and self.signals[i-1] != self.signals[i-2]:
                    if (self.ohlcv_data_1 is not None and 
                        self.ohlcv_data_2 is not None and 
                        i-1 < len(self.ohlcv_data_1) and 
                        i-1 < len(self.ohlcv_data_2)):
                        
                        # Волатильность для адаптивного slippage
                        vol_factor = self._calculate_volatility_factor(i-1)
                        
                        trade_dir_1 = 1 if pos_prev > 0 else -1  # Длинная позиция по первому активу
                        trade_dir_2 = -1 if pos_prev > 0 else 1  # Короткая позиция по второму активу
                        
                        # Реалистичные цены исполнения
                        exec_price_1 = self._calculate_execution_price(
                            s1[i-1], 
                            self.ohlcv_data_1.iloc[i-1]['high'],
                            self.ohlcv_data_1.iloc[i-1]['low'],
                            trade_dir_1, vol_factor
                        )
                        exec_price_2 = self._calculate_execution_price(
                            s2[i-1],
                            self.ohlcv_data_2.iloc[i-1]['high'], 
                            self.ohlcv_data_2.iloc[i-1]['low'],
                            trade_dir_2, vol_factor
                        )
                        
                        # PnL с учетом реалистичных цен входа и текущих цен
                        pnl = pos_prev * ((s1[i] - exec_price_1) - beta * (s2[i] - exec_price_2))
                    else:
                        # Fallback к стандартному расчету
                        pnl = pos_prev * ((s1[i] - s1[i-1]) - beta * (s2[i] - s2[i-1]))
                else:
                    # Обычный расчет для уже открытой позиции
                    pnl = pos_prev * ((s1[i] - s1[i-1]) - beta * (s2[i] - s2[i-1]))
            else:
                # Стандартный расчет без учета slippage
                pnl = pos_prev * ((s1[i] - s1[i-1]) - beta * (s2[i] - s2[i-1]))

            notional = abs(s1[i-1]) + abs(beta * s2[i-1])
            if notional == 0:
                continue
                
            returns[i-1] = pnl / notional

            # Комиссии 
            if i > 1 and self.signals[i-1] != self.signals[i-2]:
                broker_cost = self.broker_commission 
                broker_cost_with_vat = broker_cost * 1.2
                
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

        # Годовой коэффициент с учётом 16 торговых часов в день
        annual_factor = HOURS_IN_YEAR / n_hours
        annual_return = (1 + total_return) ** annual_factor - 1

        volatility = np.std(self.returns) * np.sqrt(HOURS_IN_YEAR)
        
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
        
        logger.info(f"Запуск бэктеста на {n} точках")
        
        current_coint_result = self._estimate_cointegration_at_point(analyzer, self.min_estimation_window)
        
        for i in range(self.min_estimation_window, n):
            # Периодически пересчитываем коинтеграцию используя только данные до момента i
            if i % self.recompute_frequency == 0:
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
        
        # Проверяем есть ли торговые сигналы
        total_signals = np.sum(np.abs(signals))
        if total_signals == 0:
            logger.warning("Стратегия не сгенерировала торговых сигналов")
            # Возвращаем нулевые результаты
            return {
                'cointegration': current_coint_result,
                'performance': {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'time_info': {
                        'total_hours': n-1,
                        'trading_days': (n-1) / TRADING_HOURS_PER_DAY,
                        'annual_factor': HOURS_IN_YEAR / (n-1),
                        'hours_in_year': HOURS_IN_YEAR
                    }
                },
                'risk_analysis': {
                    'var_95': 0.0,
                    'cvar_95': 0.0,
                    'max_drawdown_dist': np.array([0.0])
                },
                'returns': np.zeros(n-1)
            }
        
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
        
        if z_score > self.entry_threshold:
            return -1
        elif z_score < -self.entry_threshold:
            return 1
        elif current_position != 0 and abs(z_score) < self.exit_threshold:
            return 0
        else:
            return current_position
    
    def _calculate_execution_price(self, close_price, high_price, low_price, 
                                   trade_direction, volatility_factor=1.0):
        """
        Расчет реалистичной цены исполнения с учетом slippage и bid-ask spread
        
        :param close_price: Цена закрытия
        :param high_price: Максимальная цена за период
        :param low_price: Минимальная цена за период  
        :param trade_direction: Направление сделки (1 = покупка, -1 = продажа, 0 = нет сделки)
        :param volatility_factor: Коэффициент волатильности для адаптивного slippage
        :return: Скорректированная цена исполнения
        """
        if trade_direction == 0:
            return close_price
            
        execution_price = close_price
        
        # Bid-ask спред
        if self.enable_bid_ask_spread:
            spread_half = close_price * self.bid_ask_spread_rate / 2
            if trade_direction > 0:  # Покупка - платим ask
                execution_price += spread_half
            else:  # Продажа - получаем bid
                execution_price -= spread_half
        
        # Проскальзывание на основе волатильности внутри бара
        if self.enable_slippage:
            # Волатильность внутри бара как доля от high-low
            intrabar_volatility = (high_price - low_price) / close_price if close_price > 0 else 0
            adaptive_slippage = self.slippage_rate * volatility_factor * (1 + intrabar_volatility)
            
            slippage_amount = close_price * adaptive_slippage
            if trade_direction > 0:  # Покупка - цена хуже
                execution_price += slippage_amount
            else:  # Продажа - цена хуже
                execution_price -= slippage_amount
        
        return execution_price
    
    def _calculate_volatility_factor(self, index):
        """
        Расчет коэффициента волатильности для адаптивного slippage
        
        :param index: Текущий индекс в данных
        :return: Коэффициент волатильности (1.0 = средний, >1.0 = высокая волатильность)
        """
        if (self.ohlcv_data_1 is None or self.ohlcv_data_2 is None or 
            index < self.lookback):
            return 1.0
        
        # Считаем волатильность по последним lookback периодам
        try:
            # Волатильность первого актива
            lookback_data_1 = self.ohlcv_data_1.iloc[index-self.lookback+1:index+1]
            vol_1 = ((lookback_data_1['high'] - lookback_data_1['low']) / lookback_data_1['close']).mean()
            
            # Волатильность второго актива  
            lookback_data_2 = self.ohlcv_data_2.iloc[index-self.lookback+1:index+1]
            vol_2 = ((lookback_data_2['high'] - lookback_data_2['low']) / lookback_data_2['close']).mean()
            
            # Средняя волатильность по паре
            avg_volatility = (vol_1 + vol_2) / 2
            
            # Нормируем относительно "типичной" волатильности (2%)
            typical_volatility = 0.02
            volatility_factor = min(3.0, max(0.5, avg_volatility / typical_volatility))
            
            return volatility_factor
            
        except (IndexError, KeyError):
            return 1.0

def print_backtest_results(results, layout="single", pair_name=None, backtester=None):
    """Вывод результатов бэктеста с использованием rich
    
    Args:
        results: Результаты бэктеста
        layout: "single" для одной таблицы, "horizontal" для таблиц в строку
        pair_name: Название пары для отображения в заголовке
        backtester: экземпляр Backtester для отображения настроек торговли
    """
    console = Console()
    
    if layout == "single":
        title = "Результаты бэктеста"
        if pair_name:
            title += f" для пары {pair_name}"
        results_table = Table(title=title)
        results_table.add_column("Метрика", style="cyan")
        results_table.add_column("Значение", style="green")
        
        # Настройки торговли (если передан backtester)
        if backtester:
            results_table.add_row("[bold blue]Настройки торговли[/bold blue]", "")
            slippage_status = "✓ Включен" if backtester.enable_slippage else "✗ Выключен"
            spread_status = "✓ Включен" if backtester.enable_bid_ask_spread else "✗ Выключен"
            results_table.add_row("Учет проскальзывания", slippage_status)
            if backtester.enable_slippage:
                results_table.add_row("  └─ Размер slippage", f"{backtester.slippage_rate*100:.3f}%")
            results_table.add_row("Учет bid-ask спреда", spread_status)
            if backtester.enable_bid_ask_spread:
                results_table.add_row("  └─ Размер спреда", f"{backtester.bid_ask_spread_rate*100:.3f}%")
            results_table.add_row("", "")  # Разделитель
        
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

def create_enhanced_backtester(symbol1, symbol2, data_manager, **kwargs):
    """
    Создание расширенного бэктестера с OHLCV данными для учета slippage и bid-ask spread
    
    Args:
        symbol1: Символ первого актива
        symbol2: Символ второго актива
        data_manager: Менеджер данных для загрузки OHLCV
        **kwargs: Дополнительные параметры для Backtester
        
    Returns:
        Backtester: Настроенный экземпляр бэктестера
    """
    # Загружаем данные
    df1 = data_manager.storage.load_data(symbol1)
    df2 = data_manager.storage.load_data(symbol2)
    
    # Проверяем наличие необходимых колонок
    required_columns = ['close', 'open', 'high', 'low', 'volume']
    for col in required_columns:
        if col not in df1.columns or col not in df2.columns:
            raise ValueError(f"Отсутствует колонка '{col}' в данных")
    
    # Синхронизируем данные по времени (если есть колонка time)
    if 'time' in df1.columns and 'time' in df2.columns:
        df1_indexed = df1.set_index('time')
        df2_indexed = df2.set_index('time')
        
        # Берем пересечение временных меток
        common_times = df1_indexed.index.intersection(df2_indexed.index)
        df1_sync = df1_indexed.loc[common_times].reset_index()
        df2_sync = df2_indexed.loc[common_times].reset_index()
    else:
        # Если нет временных меток, предполагаем что данные уже синхронизированы
        min_len = min(len(df1), len(df2))
        df1_sync = df1.iloc[:min_len].copy()
        df2_sync = df2.iloc[:min_len].copy()
    
    # Создаем бэктестер
    backtester = Backtester(
        series_1=df1_sync['close'].values,
        series_2=df2_sync['close'].values,
        ohlcv_data_1=df1_sync,
        ohlcv_data_2=df2_sync,
        **kwargs
    )
    
    return backtester

def run_comparison_backtest(symbol1, symbol2, data_manager, **base_kwargs):
    """
    Запуск сравнительного бэктеста: наивный подход vs. реалистичный
    
    Args:
        symbol1: Символ первого актива
        symbol2: Символ второго актива  
        data_manager: Менеджер данных
        **base_kwargs: Базовые параметры бэктестера
        
    Returns:
        dict: Результаты сравнения
    """
    from core.analysis import DataAnalyzer
    
    analyzer = DataAnalyzer()
    
    # Наивный бэктест (без учета slippage/spread)
    naive_backtester = create_enhanced_backtester(
        symbol1, symbol2, data_manager,
        enable_slippage=False,
        enable_bid_ask_spread=False,
        **base_kwargs
    )
    
    # Реалистичный бэктест (с учетом slippage/spread)
    realistic_backtester = create_enhanced_backtester(
        symbol1, symbol2, data_manager,
        enable_slippage=True,
        enable_bid_ask_spread=True,
        **base_kwargs
    )
    
    try:
        naive_results = naive_backtester.run_full_backtest(analyzer)
        realistic_results = realistic_backtester.run_full_backtest(analyzer)
        
        return {
            'naive': {
                'results': naive_results,
                'backtester': naive_backtester
            },
            'realistic': {
                'results': realistic_results,
                'backtester': realistic_backtester
            }
        }
    except Exception as e:
        print(f"Ошибка при сравнительном бэктесте {symbol1}-{symbol2}: {e}")
        return None

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
            
            backtester = Backtester(
                series_1, series_2, 
                lookback=60, 
                entry_threshold=2.0, 
                exit_threshold=0.5,
                enable_slippage=True,
                enable_bid_ask_spread=True,
                slippage_rate=0.08,
                bid_ask_spread_rate=0.15,
                ohlcv_data_1=df1,
                ohlcv_data_2=df2,
                recompute_frequency=16  # Пересчет коинтеграции каждые 16 часов
            )
            
            try:
                results = backtester.run_full_backtest(analyzer=analyzer)
            except ValueError as e:
                if "не коинтегрированы" in str(e):
                    logger.warning(f"Пара {pair[0]}-{pair[1]} не коинтегрирована, пропуск.")
                    continue
                else:
                    raise
            
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
