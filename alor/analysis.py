import logging
from typing import Dict, List, Tuple
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from .data import DataStorage

logger = logging.getLogger(__name__)


class DataAnalyzer:
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
        # Тест Энгла-Грейнджера
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
    
    @staticmethod
    def join_pairs(symbols: List[str]) -> pd.DataFrame:
        """
        Склеивает загруженные датафреймы фьючерсов по полю 'close'.
        Возвращает DataFrame, где столбцы — это символы, а строки — общие индексы (timestamp), без NaN.
        """
        close_data = {}
        for symbol in symbols:
            df = DataStorage.load_data_from_csv(symbol)
            # Определяем индекс: если есть 'time', используем его, иначе первый столбец
            if 'time' in df.columns:
                df = df.set_index('time')
            else:
                df = df.set_index(df.columns[0])
            if 'close' in df.columns:
                close_data[symbol] = df['close']
            else:
                raise ValueError(f"В данных для {symbol} нет столбца 'close'")
        # Объединяем по индексу (outer join, чтобы не терять данные)
        return pd.DataFrame(close_data).dropna()
    