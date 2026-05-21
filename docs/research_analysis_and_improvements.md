# Анализ актуальных исследований по статистическому арбитражу и предложения по улучшению проекта

## Обзор проанализированных статей

### 1. "Copula-Based Trading of Cointegrated Cryptocurrency Pairs" (2023)
**Авторы:** Masood Tadi, Jiří Witzany

**Ключевые инновации:**
- Использование копул для моделирования зависимости между активами
- Применение как линейных (Engle-Granger), так и нелинейных (KSS) тестов коинтеграции
- Новый подход к генерации торговых сигналов через условные вероятности копул
- Использование стационарных спредов вместо логарифмических доходностей

**Результаты:** Годовая доходность 76.2% с коэффициентом Шарпа 0.97 при α₁ = 0.10

### 2. "End-to-End Policy Learning of a Statistical Arbitrage Autoencoder Architecture" (2024)
**Авторы:** Fabian Krause, Jan-Peter Calliess

**Ключевые инновации:**
- Замена PCA на автоэнкодеры для извлечения факторов
- End-to-end обучение всего пайплайна статистического арбитража
- Интеграция модели ценообразования активов с торговой стратегией
- Оптимизация по коэффициенту Шарпа напрямую

**Результаты:** Коэффициент Шарпа до 1.81 с превосходством над традиционными методами

## Анализ текущего проекта

### Сильные стороны
1. **Модульная архитектура** - четкое разделение ответственностей между компонентами
2. **Walk-forward бэктестинг** - избежание look-ahead bias
3. **Реалистичное моделирование торговых условий** - учет комиссий, проскальзывания, bid-ask спредов
4. **Комплексный анализ рисков** - методы Монте-Карло, VaR, CVaR
5. **Современный стек технологий** - Python 3.13, uv для управления зависимостями

### Области для улучшения
1. **Ограниченность методов коинтеграции** - только Engle-Granger тест
2. **Простая модель генерации сигналов** - только Z-score пороги
3. **Отсутствие нелинейных методов** - нет машинного обучения
4. **Фиксированные параметры** - нет адаптивной оптимизации
5. **Ограниченный набор активов** - только российские фьючерсы

## Предложения по улучшению

### 1. Расширение методов коинтеграции

#### Добавление нелинейных тестов коинтеграции
```python
class NonlinearCointegrationAnalyzer(DataAnalyzer):
    """Анализатор с поддержкой нелинейных тестов коинтеграции"""
    
    def kapetanios_shin_snell_test(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Тест коинтеграции Капетаниоса-Шина-Снелла для нелинейных отношений
        """
        # Реализация KSS теста
        pass
    
    def threshold_cointegration_test(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Тест пороговой коинтеграции для режимных изменений
        """
        # Реализация TAR/SETAR моделей
        pass
```

### 2. Внедрение копула-подхода

#### Модуль для работы с копулами
```python
from scipy import stats
from copulas.multivariate import GaussianMultivariate
from copulas.bivariate import Clayton, Gumbel, Frank

class CopulaSignalGenerator:
    """Генератор сигналов на основе копул"""
    
    def __init__(self):
        self.copula_families = {
            'gaussian': GaussianMultivariate,
            'clayton': Clayton,
            'gumbel': Gumbel,
            'frank': Frank
        }
    
    def fit_best_copula(self, spread1: pd.Series, spread2: pd.Series) -> str:
        """Выбор наилучшей копулы по AIC критерию"""
        best_aic = np.inf
        best_copula = None
        
        for name, copula_class in self.copula_families.items():
            try:
                copula = copula_class()
                data = np.column_stack([spread1, spread2])
                copula.fit(data)
                aic = copula.aic(data)
                
                if aic < best_aic:
                    best_aic = aic
                    best_copula = name
            except:
                continue
                
        return best_copula
    
    def generate_copula_signals(self, spread1: pd.Series, spread2: pd.Series, 
                              alpha1: float = 0.1, alpha2: float = 0.1) -> np.ndarray:
        """Генерация торговых сигналов через условные вероятности копул"""
        # Преобразование в униформные переменные
        u1 = stats.rankdata(spread1) / (len(spread1) + 1)
        u2 = stats.rankdata(spread2) / (len(spread2) + 1)
        
        # Расчет условных вероятностей h1|2 и h2|1
        # Генерация сигналов по правилам из статьи
        pass
```

### 3. Автоэнкодер для извлечения факторов

#### Модуль автоэнкодера для статистического арбитража
```python
import torch
import torch.nn as nn
from torch.optim import Adam

class StatArbAutoencoder(nn.Module):
    """Автоэнкодер для статистического арбитража с end-to-end обучением"""
    
    def __init__(self, n_assets: int, n_factors: int = 10, lambda_mse: float = 0.5):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.lambda_mse = lambda_mse
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(n_assets, n_factors),
            nn.ReLU()
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(n_factors, n_assets),
            nn.Tanh()
        )
        
        # Слой для генерации весов портфеля
        self.portfolio_layer = nn.Sequential(
            nn.Linear(n_assets, n_assets),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Кодирование
        factors = self.encoder(x)
        
        # Декодирование
        reconstructed = self.decoder(factors)
        
        # Расчет остатков
        residuals = x - reconstructed
        
        # Генерация весов портфеля
        weights = self.portfolio_layer(residuals)
        
        # Нормализация весов
        weights = weights / torch.sum(torch.abs(weights), dim=1, keepdim=True)
        
        return reconstructed, residuals, weights
    
    def loss_function(self, x, reconstructed, weights, returns):
        """Комбинированная функция потерь: MSE + Sharpe ratio"""
        mse_loss = nn.MSELoss()(x, reconstructed)
        
        # Расчет доходности портфеля
        portfolio_returns = torch.sum(weights * returns, dim=1)
        
        # Sharpe ratio loss (минимизируем отрицательный Sharpe)
        sharpe_loss = -torch.mean(portfolio_returns) / torch.std(portfolio_returns)
        
        return self.lambda_mse * mse_loss + (1 - self.lambda_mse) * sharpe_loss
```

### 4. Адаптивная оптимизация параметров

#### Модуль для динамической оптимизации
```python
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit

class AdaptiveParameterOptimizer:
    """Адаптивная оптимизация параметров стратегии"""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.parameter_history = []
    
    def optimize_parameters(self, series1: pd.Series, series2: pd.Series, 
                          objective: str = 'sharpe') -> Dict:
        """
        Оптимизация параметров стратегии на скользящем окне
        """
        def objective_function(params):
            entry_threshold, exit_threshold = params
            
            # Запуск бэктеста с данными параметрами
            backtester = ImprovedBacktester(
                series1[-self.lookback_window:],
                series2[-self.lookback_window:],
                config={'entry_threshold': entry_threshold, 'exit_threshold': exit_threshold}
            )
            
            results = backtester.run_backtest(analyzer)
            
            if objective == 'sharpe':
                return -results['performance']['sharpe_ratio']  # Минимизируем отрицательный Sharpe
            elif objective == 'return':
                return -results['performance']['annual_return']
            elif objective == 'calmar':
                return -results['performance']['annual_return'] / abs(results['performance']['max_drawdown'])
        
        # Оптимизация с ограничениями
        bounds = [(0.5, 4.0), (0.1, 1.0)]  # entry_threshold, exit_threshold
        result = minimize(objective_function, x0=[2.0, 0.5], bounds=bounds, method='L-BFGS-B')
        
        optimal_params = {
            'entry_threshold': result.x[0],
            'exit_threshold': result.x[1],
            'objective_value': -result.fun
        }
        
        self.parameter_history.append(optimal_params)
        return optimal_params
```

### 5. Расширение вселенной активов

#### Модуль для работы с международными активами
```python
class InternationalDataManager(DataManager):
    """Менеджер данных для международных активов"""
    
    def __init__(self):
        super().__init__()
        self.data_sources = {
            'crypto': 'binance',
            'us_stocks': 'alpha_vantage',
            'forex': 'oanda',
            'commodities': 'quandl'
        }
    
    def load_crypto_pairs(self, symbols: List[str], timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """Загрузка данных криптовалютных пар"""
        # Интеграция с Binance API
        pass
    
    def load_us_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Загрузка данных американских акций"""
        # Интеграция с Alpha Vantage или другими источниками
        pass
    
    def find_cross_asset_pairs(self, asset_classes: List[str]) -> List[Tuple[str, str]]:
        """Поиск коинтегрированных пар между классами активов"""
        # Кросс-активный анализ коинтеграции
        pass
```

### 6. Улучшенная система управления рисками

#### Модуль продвинутого риск-менеджмента
```python
class AdvancedRiskManager:
    """Продвинутая система управления рисками"""
    
    def __init__(self, max_portfolio_risk: float = 0.02):
        self.max_portfolio_risk = max_portfolio_risk
        self.position_limits = {}
        self.correlation_matrix = None
    
    def calculate_position_sizes(self, signals: Dict[str, float], 
                               volatilities: Dict[str, float],
                               correlations: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет размеров позиций с учетом корреляций и волатильностей
        """
        # Kelly criterion для оптимального размера позиций
        kelly_fractions = {}
        for pair, signal in signals.items():
            if pair in volatilities:
                # Упрощенная формула Келли: f = (bp - q) / b
                # где b = odds, p = вероятность выигрыша, q = вероятность проигрыша
                win_prob = 0.55  # Историческая вероятность успешных сделок
                avg_win = 0.02   # Средний выигрыш
                avg_loss = 0.015 # Средний проигрыш
                
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                kelly_fractions[pair] = min(kelly_fraction * abs(signal), 0.1)  # Ограничиваем 10%
        
        return kelly_fractions
    
    def apply_risk_overlay(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Применение риск-оверлея к весам портфеля"""
        # Проверка лимитов концентрации
        # Проверка общего левериджа
        # Динамическое хеджирование
        pass
```

### 7. Система мониторинга и алертов

#### Модуль мониторинга в реальном времени
```python
import asyncio
import websockets
from datetime import datetime

class RealTimeMonitor:
    """Система мониторинга стратегий в реальном времени"""
    
    def __init__(self, strategies: List[str]):
        self.strategies = strategies
        self.alerts = []
        self.performance_metrics = {}
    
    async def monitor_strategies(self):
        """Мониторинг стратегий в реальном времени"""
        while True:
            for strategy in self.strategies:
                # Проверка текущих позиций
                current_positions = self.get_current_positions(strategy)
                
                # Проверка лимитов риска
                risk_metrics = self.calculate_real_time_risk(current_positions)
                
                # Генерация алертов при превышении лимитов
                if risk_metrics['var_95'] > 0.05:  # 5% VaR лимит
                    self.send_alert(f"VaR превышен для стратегии {strategy}")
                
                # Проверка коинтеграции в реальном времени
                cointegration_status = self.check_cointegration_stability(strategy)
                if not cointegration_status['is_stable']:
                    self.send_alert(f"Коинтеграция нарушена для {strategy}")
            
            await asyncio.sleep(60)  # Проверка каждую минуту
    
    def send_alert(self, message: str):
        """Отправка уведомлений"""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': 'HIGH'
        }
        self.alerts.append(alert)
        # Интеграция с Telegram/Slack/Email
        print(f"ALERT: {message}")
```

## План внедрения улучшений

### Фаза 1: Основные улучшения (1-2 месяца)
1. **Добавление KSS теста коинтеграции**
2. **Реализация базового копула-подхода**
3. **Адаптивная оптимизация параметров**

### Фаза 2: Продвинутые методы (2-3 месяца)
1. **Внедрение автоэнкодера для извлечения факторов**
2. **Расширение вселенной активов (криптовалюты)**
3. **Улучшенная система риск-менеджмента**

### Фаза 3: Производственная готовность (1-2 месяца)
1. **Система мониторинга в реальном времени**
2. **Интеграция с брокерскими API**
3. **Комплексное тестирование и валидация**

## Ожидаемые результаты

### Количественные улучшения
- **Коэффициент Шарпа:** увеличение с 0.5-1.0 до 1.5-2.0
- **Максимальная просадка:** снижение на 20-30%
- **Стабильность доходности:** повышение за счет адаптивности
- **Количество торговых возможностей:** увеличение в 2-3 раза

### Качественные улучшения
- **Робастность:** лучшая адаптация к изменениям рынка
- **Масштабируемость:** возможность работы с большим количеством активов
- **Автоматизация:** снижение необходимости ручного вмешательства
- **Прозрачность:** лучшее понимание источников доходности

## Заключение

Анализ современных исследований показывает значительный потенциал для улучшения существующей системы статистического арбитража. Ключевые направления развития включают:

1. **Нелинейные методы** - копулы и автоэнкодеры для лучшего моделирования зависимостей
2. **Адаптивность** - динамическая оптимизация параметров
3. **Расширение вселенной** - работа с различными классами активов
4. **Продвинутый риск-менеджмент** - учет корреляций и динамическое хеджирование

Поэтапное внедрение предложенных улучшений позволит создать современную, конкурентоспособную систему статистического арбитража, соответствующую лучшим мировым практикам.
