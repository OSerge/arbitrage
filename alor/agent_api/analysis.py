import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

logger = logging.getLogger(__name__)

class CointegrationAnalyzer:
    """Класс для анализа коинтеграции"""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        
    def find_cointegrated_pairs(self, data: Dict[str, pd.Series]) -> List[Tuple[str, str, float]]:
        """Находит коинтегрированные пары фьючерсов
        
        Args:
            data: словарь {символ: временной ряд цен закрытия}
        Returns:
            список кортежей (символ1, символ2, p-value)
        """
        pairs = []
        symbols = list(data.keys())
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Получаем общий индекс для обоих рядов
                series1 = data[symbol1]
                series2 = data[symbol2]
                common_index = series1.index.intersection(series2.index)
                
                if len(common_index) < 20:  # Минимум 20 точек данных
                    continue
                    
                # Выравниваем ряды по общему индексу
                aligned1 = series1[common_index]
                aligned2 = series2[common_index]
                
                try:
                    # Проверяем коинтеграцию
                    score, pvalue, _ = coint(aligned1, aligned2)
                    if pvalue < self.threshold:
                        pairs.append((symbol1, symbol2, pvalue))
                except Exception as e:
                    logger.error(f"Ошибка при анализе пары {symbol1}-{symbol2}: {e}")
                    
        return sorted(pairs, key=lambda x: x[2])
        
    def calculate_hedge_ratio(self, x: pd.Series, y: pd.Series) -> float:
        """Рассчитывает коэффициент хеджирования"""
        try:
            # Получаем общий индекс
            common_index = x.index.intersection(y.index)
            if len(common_index) < 2:
                raise ValueError("Недостаточно данных для расчета коэффициента хеджирования")
                
            # Выравниваем ряды
            x_aligned = x[common_index]
            y_aligned = y[common_index]
            
            # Добавляем константу для регрессии
            X = pd.DataFrame({'x': x_aligned})
            X = sm.add_constant(X)
            
            # Рассчитываем коэффициенты регрессии
            model = OLS(y_aligned, X)
            results = model.fit()
            return results.params['x']
        except Exception as e:
            logger.error(f"Ошибка при расчете коэффициента хеджирования: {e}")
            raise
            
    def calculate_spread(self, x: pd.Series, y: pd.Series, hedge_ratio: float) -> pd.Series:
        """Рассчитывает спред между активами"""
        try:
            # Получаем общий индекс
            common_index = x.index.intersection(y.index)
            if len(common_index) < 2:
                raise ValueError("Недостаточно данных для расчета спреда")
                
            # Выравниваем ряды
            x_aligned = x[common_index]
            y_aligned = y[common_index]
            
            # Рассчитываем спред
            spread = y_aligned - hedge_ratio * x_aligned
            return spread
        except Exception as e:
            logger.error(f"Ошибка при расчете спреда: {e}")
            raise 