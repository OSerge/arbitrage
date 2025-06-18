"""
Улучшенный модуль бэктестинга с разделением ответственностей
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

from .interfaces import IBacktester, IAnalyzer
from .config import TRADING_CONFIG, BACKTEST_CONFIG

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Класс для генерации торговых сигналов"""
    
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def generate_signals(self, z_score: pd.Series) -> np.ndarray:
        """Генерация торговых сигналов на основе Z-score"""
        signals = np.zeros(len(z_score))
        position = 0
        
        for i in range(len(z_score)):
            if np.isnan(z_score.iloc[i]):
                continue
                
            if position == 0:
                if z_score.iloc[i] > self.entry_threshold:
                    signals[i] = -1
                    position = -1
                elif z_score.iloc[i] < -self.entry_threshold:
                    signals[i] = 1
                    position = 1
            elif position != 0 and abs(z_score.iloc[i]) < self.exit_threshold:
                signals[i] = 0
                position = 0
            else:
                signals[i] = position
        
        return signals


class ReturnsCalculator:
    """Класс для расчета доходности"""
    
    def __init__(self, broker_commission: float = 1.0, exchange_commission: float = 1.0, vat_rate: float = 0.2):
        self.broker_commission = broker_commission
        self.exchange_commission = exchange_commission
        self.vat_rate = vat_rate
    
    def calculate_returns(self, series_1: np.ndarray, series_2: np.ndarray, 
                         signals: np.ndarray, beta: float) -> np.ndarray:
        """Расчёт процентной доходности стратегии с учётом капитала"""
        n = len(series_1)
        returns = np.zeros(n - 1)

        for i in range(1, n):
            pos_prev = signals[i-1]
            if pos_prev == 0:
                continue

            notional = abs(series_1[i-1]) + abs(beta * series_2[i-1])
            if notional == 0:
                continue

            pnl = pos_prev * ((series_1[i] - series_1[i-1]) - beta * (series_2[i] - series_2[i-1]))
            returns[i-1] = pnl / notional

            # Учет комиссий при смене позиции
            if i > 1 and signals[i-1] != signals[i-2]:
                total_commission = self._calculate_commission()
                returns[i-1] -= total_commission / notional

        return returns
    
    def _calculate_commission(self) -> float:
        """Расчет общей комиссии за сделку"""
        total_commission = 2 * (self.broker_commission + self.exchange_commission)
        total_commission += 2 * (self.broker_commission * self.vat_rate)
        return total_commission


class PerformanceAnalyzer:
    """Класс для анализа производительности"""
    
    def __init__(self, trading_config=None):
        self.trading_config = trading_config or TRADING_CONFIG
    
    def calculate_metrics(self, returns: np.ndarray) -> Dict:
        """Расчет метрик производительности для часовых данных"""
        n_hours = len(returns)
        
        if n_hours == 0:
            raise ValueError("Пустой массив доходностей")

        # Кумулятивная доходность
        equity_curve = np.cumprod(1 + returns)
        total_return = equity_curve[-1] - 1

        # Годовой коэффициент
        annual_factor = self.trading_config.hours_in_year / n_hours
        annual_return = (1 + total_return) ** annual_factor - 1

        volatility = np.std(returns) * np.sqrt(self.trading_config.hours_in_year)
        
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = equity_curve / peak - 1
        max_drawdown = np.min(drawdown)
        
        time_info = {
            'total_hours': n_hours,
            'trading_days': n_hours / self.trading_config.TRADING_HOURS_PER_DAY,
            'annual_factor': annual_factor,
            'hours_in_year': self.trading_config.hours_in_year
        }
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'time_info': time_info
        }


class RiskAnalyzer:
    """Класс для анализа рисков"""
    
    def monte_carlo_analysis(self, returns: np.ndarray, n_simulations: int = 1000) -> Dict:
        """Анализ рисков методом Монте-Карло"""
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


class ImprovedBacktester(IBacktester):
    """Улучшенный класс для комплексного бэктестинга"""
    
    def __init__(self, series_1, series_2, config: Optional[Dict] = None):
        """
        Инициализация бэктестера
        
        :param series_1: Временной ряд цен первого актива
        :param series_2: Временной ряд цен второго актива
        :param config: Конфигурация бэктестера
        """
        self.series_1 = np.array(series_1)
        self.series_2 = np.array(series_2)
        
        backtest_config = config or BACKTEST_CONFIG.__dict__
        
        self.signal_generator = SignalGenerator(
            entry_threshold=backtest_config.get('entry_threshold', 2.0),
            exit_threshold=backtest_config.get('exit_threshold', 0.5)
        )
        
        self.returns_calculator = ReturnsCalculator(
            broker_commission=backtest_config.get('broker_commission', 1.0),
            exchange_commission=backtest_config.get('exchange_commission', 1.0),
            vat_rate=backtest_config.get('vat_rate', 0.2)
        )
        
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        
        self.lookback = backtest_config.get('lookback', 60)
        
        self.signals = None
        self.returns = None
        self.performance = None
        self.risk_analysis = None
    
    def run_backtest(self, analyzer: IAnalyzer) -> Dict:
        """Запуск полного цикла бэктестинга"""
        cointegration = analyzer.check_cointegration(self.series_1, self.series_2)
        
        if not cointegration['is_cointegrated']:
            raise ValueError("Активы не коинтегрированы")
        
        z_score = analyzer.calculate_zscore(
            pd.Series(self.series_1),
            pd.Series(self.series_2),
            cointegration['beta'],
            self.lookback,
            cointegration['alpha']
        )
        
        self.signals = self.signal_generator.generate_signals(z_score)
        
        self.returns = self.returns_calculator.calculate_returns(
            self.series_1, self.series_2, self.signals, cointegration['beta']
        )
        
        self.performance = self.performance_analyzer.calculate_metrics(self.returns)
        
        self.risk_analysis = self.risk_analyzer.monte_carlo_analysis(self.returns)
        
        return {
            'cointegration': cointegration,
            'performance': self.performance,
            'risk_analysis': self.risk_analysis,
            'returns': self.returns
        } 