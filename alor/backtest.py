import numpy as np
import pandas as pd

class BackTest:
    """
    Класс для бэктестинга стратегий на коинтегрированных парах.
    """
    @staticmethod
    def backtest_pair(series1: pd.Series, series2: pd.Series, 
                      entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> dict:
        """
        Проводит бэктестинг для коинтегрированной пары
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            entry_threshold (float): Порог входа в позицию (в стандартных отклонениях)
            exit_threshold (float): Порог выхода из позиции (в стандартных отклонениях)
        Returns:
            dict: Результаты бэктестинга
        """
        # Нормализация данных
        norm1 = (series1 - series1.mean()) / series1.std()
        norm2 = (series2 - series2.mean()) / series2.std()
        spread = norm1 - norm2
        z_score = (spread - spread.mean()) / spread.std()
        positions = pd.Series(0, index=spread.index)
        positions[z_score > entry_threshold] = -1  # Короткая позиция
        positions[z_score < -entry_threshold] = 1  # Длинная позиция
        positions[abs(z_score) < exit_threshold] = 0  # Выход
        returns = positions.shift(1) * spread.diff()
        cumulative_returns = returns.cumsum()
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
