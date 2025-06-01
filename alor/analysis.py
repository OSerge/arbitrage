import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class CointAnalyzer:
    """Класс для анализа коинтеграции"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Инициализация анализатора коинтеграции
        
        Args:
            alpha (float): Уровень значимости для тестов (по умолчанию 0.05)
        """
        self.alpha = alpha
    
    def check_stationarity(self, series: pd.Series) -> Tuple[float, bool]:
        """
        Проверяет стационарность временного ряда
        
        Args:
            series (pd.Series): Временной ряд для проверки
            
        Returns:
            Tuple[float, bool]: (p-value, является ли ряд стационарным)
        """
        result = adfuller(series)
        p_value = result[1]
        return p_value, p_value < self.alpha
    
    def calculate_spread(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Вычисляет спред между двумя рядами
        
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            
        Returns:
            pd.Series: Нормализованный спред
        """
        # Нормализация данных
        norm1 = (series1 - series1.mean()) / series1.std()
        norm2 = (series2 - series2.mean()) / series2.std()
        return norm1 - norm2
    
    def check_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Проверяет коинтеграцию между двумя рядами
        
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            
        Returns:
            Dict: Результаты анализа коинтеграции
        """
        # Проверка стационарности исходных рядов
        p_value1, is_stationary1 = self.check_stationarity(series1)
        p_value2, is_stationary2 = self.check_stationarity(series2)
        
        # Тест Engle-Granger
        score, p_value, _ = coint(series1, series2)
        
        # Расчет коэффициентов регрессии
        model = OLS(series1, series2).fit()
        beta = model.params[0]
        
        # Расчет спреда
        spread = self.calculate_spread(series1, series2)
        spread_p_value, is_spread_stationary = self.check_stationarity(spread)
        
        return {
            'p_value': p_value,
            'is_cointegrated': p_value < self.alpha,
            'beta': beta,
            'spread_p_value': spread_p_value,
            'is_spread_stationary': is_spread_stationary,
            'series1_stationary': is_stationary1,
            'series2_stationary': is_stationary2,
            'score': score,
            'spread': spread
        }
    
    def find_cointegrated_pairs(self, df: pd.DataFrame) -> List[Dict]:
        """
        Находит все коинтегрированные пары в датафрейме
        
        Args:
            df (pd.DataFrame): Датафрейм с временными рядами
            
        Returns:
            List[Dict]: Список коинтегрированных пар с результатами анализа
        """
        cointegrated_pairs = []
        columns = df.columns
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                result = self.check_cointegration(df[col1], df[col2])
                
                if result['is_cointegrated']:
                    cointegrated_pairs.append({
                        'pair': (col1, col2),
                        'results': result
                    })
        
        return cointegrated_pairs
    
    def plot_cointegrated_pair(self, series1: pd.Series, series2: pd.Series, 
                             name1: str, name2: str, results: Dict):
        """
        Визуализирует коинтегрированную пару
        
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            name1 (str): Название первого ряда
            name2 (str): Название второго ряда
            results (Dict): Результаты анализа коинтеграции
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # График цен
        ax1.plot(series1, label=name1)
        ax1.plot(series2, label=name2)
        ax1.set_title(f'Цены {name1} и {name2}')
        ax1.legend()
        
        # График спреда
        spread = results['spread']
        ax2.plot(spread, label='Спред')
        ax2.axhline(y=0, color='r', linestyle='-')
        ax2.set_title('Спред')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Вывод статистики
        print(f"\nСтатистика для пары {name1}/{name2}:")
        print(f"p-value коинтеграции: {results['p_value']:.4f}")
        print(f"Коэффициент бета: {results['beta']:.4f}")
        print(f"p-value спреда: {results['spread_p_value']:.4f}")
    
    def backtest_pair(self, series1: pd.Series, series2: pd.Series, 
                     entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> Dict:
        """
        Проводит бэктестинг для коинтегрированной пары
        
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            entry_threshold (float): Порог входа в позицию (в стандартных отклонениях)
            exit_threshold (float): Порог выхода из позиции (в стандартных отклонениях)
            
        Returns:
            Dict: Результаты бэктестинга
        """
        # Расчет спреда
        spread = self.calculate_spread(series1, series2)
        
        # Расчет z-score
        z_score = (spread - spread.mean()) / spread.std()
        
        # Сигналы
        positions = pd.Series(0, index=spread.index)
        positions[z_score > entry_threshold] = -1  # Короткая позиция
        positions[z_score < -entry_threshold] = 1  # Длинная позиция
        positions[abs(z_score) < exit_threshold] = 0  # Выход
        
        # Расчет доходности
        returns = positions.shift(1) * spread.diff()
        cumulative_returns = returns.cumsum()
        
        # Расчет метрик
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
        
        return {
            'returns': returns,
            'cumulative_returns': cumulative_returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'positions': positions,
            'z_score': z_score
        }
    