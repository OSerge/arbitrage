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


class ArbitrageBacktest:
    """
    Класс для векторизованного бэктестинга статистического арбитража по паре активов.
    Поддерживает расчет спреда, z-score, генерацию сигналов, учет комиссий и расчёт метрик.
    """
    def __init__(
        self,
        series1: pd.Series,
        series2: pd.Series,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        capital: float = 100000,
        commission: float = 0.01,
        freq: int = 252,
    ):
        # Исходные данные
        self.s1 = series1
        self.s2 = series2
        # Параметры стратегии
        self.entry = entry_threshold
        self.exit = exit_threshold
        self.capital = capital
        self.commission = commission
        self.freq = freq
        # Фреймы для результатов
        self.data = pd.DataFrame(index=series1.index)

    def prepare_data(self):
        # Расчет нормализованных рядов и спреда
        norm1 = (self.s1 - self.s1.mean()) / self.s1.std()
        norm2 = (self.s2 - self.s2.mean()) / self.s2.std()
        spread = norm1 - norm2
        # z-score
        z = (spread - spread.mean()) / spread.std()
        self.data['spread'] = spread
        self.data['z'] = z

    def generate_signals(self):
        # Создаем сигналы на вход и выход
        conds = [
            self.data['z'] >  self.entry,
            self.data['z'] < -self.entry,
            self.data['z'].abs() < self.exit,
        ]
        choices = [-1, 1, 0]
        sig = np.select(conds, choices, default=np.nan)
        signals = pd.Series(sig, index=self.data.index)
        # Приведение к позициям: удерживаем позицию до сигнала выхода
        self.data['position'] = signals.replace(0, np.nan).ffill().fillna(0)

    def backtest(self):
        # Подготовка
        self.prepare_data()
        self.generate_signals()
        # Рассчет PnL по спреду
        pnl = self.data['position'].shift(1) * self.data['spread'].diff()
        # Учёт комиссии за изменение позиции
        trades = self.data['position'].diff().abs()
        cost = trades * self.commission
        # Экьютити-кривая
        returns = pnl - cost
        equity = (1 + returns.fillna(0)).cumprod() * self.capital

        self.data['pnl'] = pnl
        self.data['cost'] = cost
        self.data['returns'] = returns
        self.data['equity'] = equity

    def metrics(self) -> dict:
        # Расчет ключевых метрик
        ret = self.data['returns'].dropna()
        total_return = self.data['equity'].iloc[-1] / self.capital - 1
        annual_return = (1 + total_return) ** (self.freq / len(ret)) - 1
        sharpe = np.sqrt(self.freq) * ret.mean() / ret.std()
        drawdown = self.data['equity'] / self.data['equity'].cummax() - 1
        max_dd = drawdown.min()
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        }

    def run(self) -> pd.DataFrame:
        """Запуск бэктеста и возврат расширенного DataFrame с результатами."""
        self.backtest()
        return self.data
