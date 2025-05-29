import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlorAPI:
    """Класс для работы с API Alor"""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.alor.ru/md/v2"
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Выполняет запрос к API"""
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            raise

class FuturesData:
    """Класс для работы с данными фьючерсов"""
    
    def __init__(self, api: AlorAPI):
        self.api = api
        self.data_dir = Path("data/futures")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_futures(self) -> pd.DataFrame:
        """Получает список всех доступных фьючерсов"""
        params = {
            'sector': 'FORTS',
            'exchange': 'MOEX',
            'instrumentGroup': 'RFUD',
            'limit': 100
        }
        data = self.api._make_request("Securities", params)
        return pd.json_normalize(data)
    
    def fetch_futures_data(self, fut_code: str, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Скачивает исторические данные по фьючерсу"""
        params = {
            'symbol': fut_code,
            'exchange': 'MOEX',
            'instrumentGroup': 'RFUD',
            'tf': 3600,  # часовой таймфрейм
            'from': int(from_date.timestamp()),
            'to': int(to_date.timestamp()),
            'splitAdjust': 'true',
            'format': 'Heavy',
            'jsonResponse': 'true',
        }
        
        data = self.api._make_request("history", params)
        return pd.json_normalize(data['history'])
    
    def save_futures_data(self, fut_code: str, data: pd.DataFrame):
        """Сохраняет данные фьючерса в файл"""
        file_path = self.data_dir / f"{fut_code}.csv"
        data.to_csv(file_path, index=False)
    
    def load_futures_data(self, fut_code: str) -> pd.DataFrame:
        """Загружает данные фьючерса из файла"""
        file_path = self.data_dir / f"{fut_code}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Данные для {fut_code} не найдены")
        return pd.read_csv(file_path)
    
    def get_available_futures(self) -> List[str]:
        """Возвращает список фьючерсов, для которых есть скачанные данные"""
        return [f.stem for f in self.data_dir.glob("*.csv")]
    
    def is_data_available(self, fut_code: str) -> bool:
        """Проверяет наличие данных для конкретного фьючерса"""
        return (self.data_dir / f"{fut_code}.csv").exists()

class CointegrationAnalyzer:
    """Класс для анализа коинтеграции"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def check_stationarity(self, series: pd.Series) -> Tuple[float, bool]:
        """Проверяет стационарность временного ряда"""
        result = adfuller(series)
        p_value = result[1]
        return p_value, p_value < self.alpha
    
    def calculate_spread(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Вычисляет спред между двумя рядами"""
        # Нормализация данных
        norm1 = (series1 - series1.mean()) / series1.std()
        norm2 = (series2 - series2.mean()) / series2.std()
        return norm1 - norm2
    
    def check_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """Проверяет коинтеграцию между двумя рядами"""
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
            'series2_stationary': is_stationary2
        }
    
    def find_cointegrated_pairs(self, df: pd.DataFrame) -> List[Dict]:
        """Находит все коинтегрированные пары в датафрейме"""
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
        """Визуализирует коинтегрированную пару"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # График цен
        ax1.plot(series1, label=name1)
        ax1.plot(series2, label=name2)
        ax1.set_title(f'Цены {name1} и {name2}')
        ax1.legend()
        
        # График спреда
        spread = self.calculate_spread(series1, series2)
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

class Backtester:
    """Класс для бэктестинга стратегий"""
    
    def __init__(self, analyzer: CointegrationAnalyzer):
        self.analyzer = analyzer
    
    def backtest_pair(self, series1: pd.Series, series2: pd.Series, 
                     entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> Dict:
        """Проводит бэктестинг для коинтегрированной пары"""
        # Расчет спреда
        spread = self.analyzer.calculate_spread(series1, series2)
        
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
            'positions': positions
        }

def main(force_download: bool = False):
    """Основная функция для запуска анализа
    
    Args:
        force_download (bool): Если True, то перезагружает все данные, даже если они уже есть
    """
    # Инициализация
    api = AlorAPI(token="your_token_here")
    futures_data = FuturesData(api)
    analyzer = CointegrationAnalyzer()
    backtester = Backtester(analyzer)
    
    # Получение списка фьючерсов
    futures = futures_data.get_all_futures()
    
    # Скачивание данных (если нужно)
    if force_download:
        logger.info("Принудительная загрузка всех данных")
        from_date = datetime(2024, 1, 1)
        to_date = datetime(2024, 3, 1)
        
        for _, fut in futures.iterrows():
            try:
                data = futures_data.fetch_futures_data(fut['shortname'], from_date, to_date)
                futures_data.save_futures_data(fut['shortname'], data)
                logger.info(f"Данные для {fut['shortname']} успешно сохранены")
            except Exception as e:
                logger.error(f"Ошибка при скачивании данных для {fut['shortname']}: {e}")
    else:
        logger.info("Проверка наличия данных...")
        available_futures = futures_data.get_available_futures()
        logger.info(f"Найдено {len(available_futures)} файлов с данными")
    
    # Загрузка и объединение данных
    all_data = pd.DataFrame()
    for fut_code in futures['shortname']:
        try:
            if futures_data.is_data_available(fut_code):
                data = futures_data.load_futures_data(fut_code)
                if 'close' in data.columns:
                    all_data[fut_code] = data['close']
                    logger.info(f"Данные для {fut_code} успешно загружены")
            else:
                logger.warning(f"Данные для {fut_code} отсутствуют")
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для {fut_code}: {e}")
    
    if all_data.empty:
        logger.error("Нет доступных данных для анализа")
        return
    
    # Поиск коинтегрированных пар
    logger.info("Поиск коинтегрированных пар...")
    cointegrated_pairs = analyzer.find_cointegrated_pairs(all_data)
    logger.info(f"Найдено {len(cointegrated_pairs)} коинтегрированных пар")
    
    # Бэктестинг найденных пар
    for pair in cointegrated_pairs:
        col1, col2 = pair['pair']
        series1, series2 = all_data[col1], all_data[col2]
        
        # Визуализация
        analyzer.plot_cointegrated_pair(series1, series2, col1, col2, pair['results'])
        
        # Бэктестинг
        backtest_results = backtester.backtest_pair(series1, series2)
        print(f"\nРезультаты бэктестинга для пары {col1}/{col2}:")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ коинтеграции фьючерсов')
    parser.add_argument('--force-download', action='store_true',
                      help='Принудительно перезагрузить все данные')
    
    args = parser.parse_args()
    main(force_download=args.force_download) 