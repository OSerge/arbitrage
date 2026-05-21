# ADR 0004: Python-first runtime для foundation и MVP

- Статус: accepted
- Дата: 2026-05-21

## Контекст

Ранние архитектурные формулировки слишком рано тянули платформу к обязательному `Rust`-first runtime на live-контуре.

По факту текущий этап проекта требует другого приоритета:

- быстрее превратить существующий research-код в package-based platform;
- получить reproducible backtest, replay, signal generation и portfolio construction;
- довести систему до paper/live MVP через брокерские адаптеры и независимый risk layer;
- удержать архитектурную дисциплину без преждевременного multi-language overhead.

Текущая кодовая база, инструменты и скорость разработки уже сосредоточены вокруг Python. На этой стадии проекта это важнее, чем преждевременная оптимизация под потенциальный low-latency future state.

## Решение

Принять `Python-first` как дефолтную стратегию реализации для:

- foundation-фазы;
- research platform;
- execution/risk MVP;
- control-plane API и internal operator tooling.

Практически это означает:

1. `Python` является основным языком для research, backtest, replay, signal generation, portfolio construction, paper/live orchestration, risk services и брокерских MVP-адаптеров.
2. Разделение research, execution/risk и UI остается обязательным, но на MVP эта граница реализуется через отдельные процессы, сервисы и контракты, а не через обязательный переход на другой язык.
3. `Rust` или `C++` рассматриваются только для подтвержденных hotspot-участков: сетевой I/O, order routing, replay throughput, microstructure-heavy pipelines или latency-sensitive guards.
4. До появления измеримого bottleneck не делать multi-language rewrite как baseline-решение.

## Почему так

- это кратчайший путь к research/live parity без архитектурного отката;
- снижается когнитивная и операционная стоимость платформы на раннем этапе;
- проще поддерживать testing, packaging, CI и инженерные стандарты в одном toolchain;
- Python уже покрывает домены, которые сейчас важнее всего: исследования, симуляция, портфель, risk logic и paper/live MVP;
- системные языки остаются доступной эволюцией, а не преждевременной догмой.

## Последствия

Положительные:

- быстрее time-to-MVP;
- меньше platform overhead на ранней стадии;
- проще AI-assisted разработка и сопровождение кода;
- ниже риск сделать дорогостоящий low-level runtime до подтверждения реальной потребности.

Отрицательные:

- часть latency/jitter проблем придется отслеживать раньше и строже;
- нужны профилирование и performance budgets, чтобы не пропустить момент для selective rewrite;
- некоторые ultra-low-latency use cases будут отложены до следующей стадии зрелости.
