import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Класс для анализа коинтеграции
    
    Формат данных в исходных .csv:
    ```
    time,close,open,high,low,volume
    1742191200,139800.0,139800.0,139800.0,139800.0,10
    1742194800,139587.0,139351.0,139715.0,139220.0,31
    1742198400,139975.0,139827.0,140080.0,139500.0,16
    ...
    ```

    Формат данных в .csv после join_pair:
    ```
    time,close_1,close_2
    1742191200,139800.0,139800.0
    1742194800,139587.0,139351.0
    ...
    ```
    """
    
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
    
    def engle_granger_test(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Тест коинтеграции Энгла-Грейнджера
        
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            
        Returns:
            Dict: Результаты теста коинтеграции
        """
        score, pvalue, _ = coint(series1, series2)
        X = np.column_stack([np.ones(len(series1)), series2])
        beta = np.linalg.lstsq(X, series1, rcond=None)[0]
        
        return {
            'alpha': beta[0],
            'beta': beta[1],
            'adf_statistic': score,
            'p_value': pvalue,
            'is_cointegrated': pvalue < self.alpha
        }
    
    def calculate_zscore(self, series1: pd.Series, series2: pd.Series, 
                        beta: float, lookback: int, alpha: float = 0.0) -> pd.Series:
        """
        Расчет Z-score для пары инструментов
        
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            beta (float): Коэффициент коинтеграции
            lookback (int): Размер окна для расчета
            alpha (float): Константа сдвига
            
        Returns:
            pd.Series: Z-score
        """
        # Спред c учётом константы alpha (сдвига), получаемой из регрессии
        spread = series1 - beta * series2 - alpha
        spread_series = pd.Series(spread)
        
        spread_mean = spread_series.rolling(lookback).mean()
        spread_std = spread_series.rolling(lookback).std()
        
        return (spread - spread_mean) / spread_std
    
    def check_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Проверяет коинтеграцию между двумя рядами
        
        Args:
            series1 (pd.Series): Первый временной ряд
            series2 (pd.Series): Второй временной ряд
            
        Returns:
            Dict: Результаты анализа коинтеграции
        """
        return self.engle_granger_test(series1, series2)
    
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
    def join_pair(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Склеивает загруженные датафреймы фьючерсов по индексу 'time'.

        Возвращает DataFrame, где столбцы — это 'close'.
        """
        return pd.merge(
            df1['close'], 
            df2['close'], 
            left_index=True, 
            right_index=True, 
            suffixes=("_1", "_2")
            )
    