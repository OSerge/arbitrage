# Статистический арбитраж на MOEX

Система для бэктестинга арбитражных стратегий на фьючерсах Московской биржи.

## Что делает проект

Проект анализирует пары фьючерсов на коинтеграцию и проводит бэктестинг арбитражных стратегий.

## Установка

```bash
git clone https://github.com/OSerge/arbitrage.git
cd arbitrage
uv sync
source .venv/bin/activate
```

## Как использовать

### 1. Загрузка данных

Если нужны новые данные или до этого момента данные не загружались вовсе - используйте `core/downloader.py` для загрузки. Можно запустить его как есть.

В репозитории уже есть загруженные данные в `data/`, поэтому для быстрого старта шаг загрузки можно пропустить.


### 2. Бэктестинг

Используйте `core/backtest.py`.

По умолчанию в файле перечислены следующие фьючерсы, которые будут использоваться для попарного анализа (каждый с каждым):

```python
futures = [
    "GKM5", # Обыкновенные акции ПАО «ГМК «Норильский никель»
    "GZM5", # Газпром обыкновенные
    "CHM5", # обыкновенные акции ПАО «Северсталь»
    "TTM5", # Татнефть
    "TNM5", # Транснефть
    "RNM5", # Роснефть
    "LKM5", # Лукойл
    "SRM5", # обыкновенные акции ПАО Сбербанк
    "SPM5", # привилег. акции ПАО Сбербанк
    "VBM5", # ВТБ
]
```

- Раскомментируйте/закомментируйте нужные активы в списке `futures`.
- Запустите `python core/backtest.py`.


### 3. Jupyter блокноты

В папке `notebooks/` есть готовый блокнот `backtest.ipynb` для анализа пары фьючерсов на простые и привилегированные акции Сбера, в нем приведен код, который наглядно показывает работу бэктеста с простой стратегией через `z-score`.


## Структура проекта

- `core/` - основной код
- `data/` - CSV файлы с данными фьючерсов
- `notebooks/` - Jupyter блокноты для анализа
- `tests/` - тесты

## Основные модули

- `downloader.py` - загрузка данных через API Alor
- `backtest.py` - запуск бэктестинга пар активов
- `analysis.py` - анализ коинтеграции и др.
- `data.py` - работа с файлами и данными
- `config.py` - настройки стратегии (для продвинутого бэктеста, в разработке)

## Требования

- Python >= 3.12
- uv для управления зависимостями
