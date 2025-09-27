"""
Модуль для валидации математической корректности арбитражных операций
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ArbitrageValidator:
    """Класс для валидации математической корректности арбитражных стратегий"""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Инициализация валидатора
        
        Args:
            tolerance (float): Допустимая погрешность для численных сравнений
        """
        self.tolerance = tolerance
        
    def validate_cointegration_test(self, series1: pd.Series, series2: pd.Series, 
                                   coint_result: Dict) -> List[str]:
        """
        Валидация результатов теста коинтеграции
        
        Args:
            series1, series2: Временные ряды
            coint_result: Результат теста коинтеграции
            
        Returns:
            List[str]: Список найденных проблем
        """
        issues = []
        
        # Проверка корректности коэффициентов регрессии
        X = np.column_stack([np.ones(len(series1)), series2])
        try:
            beta_manual = np.linalg.lstsq(X, series1, rcond=None)[0]
            
            if abs(coint_result['alpha'] - beta_manual[0]) > self.tolerance:
                issues.append(f"Некорректный расчет alpha: {coint_result['alpha']:.6f} vs {beta_manual[0]:.6f}")
                
            if abs(coint_result['beta'] - beta_manual[1]) > self.tolerance:
                issues.append(f"Некорректный расчет beta: {coint_result['beta']:.6f} vs {beta_manual[1]:.6f}")
                
        except np.linalg.LinAlgError:
            issues.append("Ошибка в расчете коэффициентов регрессии")
        
        # Проверка диапазона p-value
        if not (0 <= coint_result['p_value'] <= 1):
            issues.append(f"P-value вне допустимого диапазона: {coint_result['p_value']}")
        
        return issues
    
    def validate_zscore_calculation(self, series1: pd.Series, series2: pd.Series, 
                                  beta: float, alpha: float, lookback: int, 
                                  zscore_result: pd.Series) -> List[str]:
        """
        Валидация расчета Z-score
        
        Args:
            series1, series2: Временные ряды
            beta, alpha: Коэффициенты коинтеграции
            lookback: Размер окна
            zscore_result: Результат расчета Z-score
            
        Returns:
            List[str]: Список найденных проблем
        """
        issues = []
        
        # Пересчет Z-score для проверки
        spread = series1 - beta * series2 - alpha
        
        for i in range(lookback, len(series1)):
            window_spread = spread.iloc[i-lookback:i]
            expected_zscore = (spread.iloc[i] - window_spread.mean()) / window_spread.std()
            
            if not np.isnan(expected_zscore) and not np.isnan(zscore_result.iloc[i]):
                if abs(expected_zscore - zscore_result.iloc[i]) > self.tolerance:
                    issues.append(f"Некорректный Z-score на позиции {i}: {zscore_result.iloc[i]:.6f} vs {expected_zscore:.6f}")
                    break  # Достаточно одной ошибки для диагностики
        
        return issues
    
    def validate_returns_calculation(self, series1: np.ndarray, series2: np.ndarray,
                                   signals: np.ndarray, beta: float,
                                   calculated_returns: np.ndarray) -> List[str]:
        """
        Валидация расчета доходности
        
        Args:
            series1, series2: Ценовые ряды
            signals: Торговые сигналы
            beta: Коэффициент хеджирования
            calculated_returns: Рассчитанная доходность
            
        Returns:
            List[str]: Список найденных проблем
        """
        issues = []
        
        for i in range(1, len(series1)):
            if signals[i-1] == 0:
                continue
                
            # Пересчет PnL
            pnl = signals[i-1] * ((series1[i] - series1[i-1]) - beta * (series2[i] - series2[i-1]))
            notional = max(abs(series1[i-1]), abs(beta * series2[i-1]))
            
            if notional > 0:
                expected_return = pnl / notional
                
                # Допускаем небольшую разницу из-за комиссий
                if abs(calculated_returns[i-1] - expected_return) > 0.01:  # 1% tolerance for commissions
                    issues.append(f"Некорректный расчет доходности на позиции {i-1}")
                    break
        
        return issues
    
    def validate_performance_metrics(self, returns: np.ndarray, 
                                   performance: Dict, hours_in_year: int) -> List[str]:
        """
        Валидация метрик производительности
        
        Args:
            returns: Массив доходностей
            performance: Рассчитанные метрики
            hours_in_year: Количество торговых часов в году
            
        Returns:
            List[str]: Список найденных проблем
        """
        issues = []
        
        # Проверка кумулятивной доходности
        equity_curve = np.cumprod(1 + returns)
        expected_total_return = equity_curve[-1] - 1
        
        if abs(performance['total_return'] - expected_total_return) > self.tolerance:
            issues.append(f"Некорректный расчет общей доходности: {performance['total_return']:.6f} vs {expected_total_return:.6f}")
        
        # Проверка волатильности
        expected_volatility = np.std(returns) * np.sqrt(hours_in_year)
        if abs(performance['volatility'] - expected_volatility) > self.tolerance:
            issues.append(f"Некорректный расчет волатильности: {performance['volatility']:.6f} vs {expected_volatility:.6f}")
        
        # Проверка коэффициента Шарпа
        if performance['volatility'] > 0:
            expected_sharpe = performance['annual_return'] / performance['volatility']
            if abs(performance['sharpe_ratio'] - expected_sharpe) > self.tolerance:
                issues.append(f"Некорректный расчет коэффициента Шарпа: {performance['sharpe_ratio']:.6f} vs {expected_sharpe:.6f}")
        
        return issues
    
    def validate_risk_metrics(self, returns: np.ndarray, 
                            risk_analysis: Dict, n_simulations: int = 1000) -> List[str]:
        """
        Валидация метрик риска
        
        Args:
            returns: Массив доходностей
            risk_analysis: Результаты анализа рисков
            n_simulations: Количество симуляций Монте-Карло
            
        Returns:
            List[str]: Список найденных проблем
        """
        issues = []
        
        # Проверка диапазонов VaR и CVaR
        if not (-1 <= risk_analysis['var_95'] <= 1):
            issues.append(f"VaR 95% вне разумного диапазона: {risk_analysis['var_95']:.4f}")
        
        if not (-1 <= risk_analysis['cvar_95'] <= 1):
            issues.append(f"CVaR 95% вне разумного диапазона: {risk_analysis['cvar_95']:.4f}")
        
        # CVaR должен быть меньше или равен VaR (по абсолютному значению)
        if abs(risk_analysis['cvar_95']) < abs(risk_analysis['var_95']):
            issues.append(f"CVaR должен быть больше VaR по абсолютному значению: CVaR={risk_analysis['cvar_95']:.4f}, VaR={risk_analysis['var_95']:.4f}")
        
        return issues
    
    def full_validation(self, backtest_results: Dict, 
                       series1: pd.Series, series2: pd.Series) -> Dict[str, List[str]]:
        """
        Полная валидация результатов бэктеста
        
        Args:
            backtest_results: Результаты бэктеста
            series1, series2: Исходные временные ряды
            
        Returns:
            Dict[str, List[str]]: Словарь с проблемами по категориям
        """
        validation_results = {
            'cointegration': [],
            'zscore': [],
            'returns': [],
            'performance': [],
            'risk': [],
            'general': []
        }
        
        try:
            # Валидация коинтеграции
            coint_issues = self.validate_cointegration_test(
                series1, series2, backtest_results['cointegration']
            )
            validation_results['cointegration'].extend(coint_issues)
            
            # Валидация метрик производительности
            perf_issues = self.validate_performance_metrics(
                backtest_results['returns'], 
                backtest_results['performance'],
                backtest_results['performance']['time_info']['hours_in_year']
            )
            validation_results['performance'].extend(perf_issues)
            
            # Валидация метрик риска
            risk_issues = self.validate_risk_metrics(
                backtest_results['returns'],
                backtest_results['risk_analysis']
            )
            validation_results['risk'].extend(risk_issues)
            
        except Exception as e:
            validation_results['general'].append(f"Ошибка валидации: {str(e)}")
        
        return validation_results


def print_validation_results(validation_results: Dict[str, List[str]]) -> None:
    """Вывод результатов валидации"""
    total_issues = sum(len(issues) for issues in validation_results.values())
    
    if total_issues == 0:
        print("✅ Валидация пройдена успешно - критических ошибок не найдено")
        return
    
    print(f"⚠️  Найдено {total_issues} проблем:")
    
    for category, issues in validation_results.items():
        if issues:
            print(f"\n📊 {category.upper()}:")
            for issue in issues:
                print(f"  • {issue}")