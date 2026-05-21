# ADR 0001: разделить research и live-runtime

- Статус: accepted
- Дата: 2026-05-21

## Контекст

Текущий репозиторий успешно решает research-задачи в Python, но live execution platform предъявляет другие требования:

- latency predictability;
- устойчивость сетевых соединений;
- независимые risk gates;
- контролируемый state management;
- воспроизводимый audit trail.

Если оставить research, execution, risk и operator workflows в одном runtime-контуре, то система быстро станет хрупкой и плохо масштабируемой организационно.

## Решение

Разделить платформу на независимые платформенные контуры:

1. `Research contour` — исследования, batch jobs, feature generation, backtest, replay analysis.
2. `Execution/Risk contour` — headless сервисы исполнения, позиций, risk checks и broker connectivity.
3. `Control plane / UI contour` — operator-facing API и internal UI для research, ops и risk.

Контуры взаимодействуют через versioned event schemas, control-plane API и общие data contracts.

## Почему так

- Такая граница уменьшает соблазн тащить notebook-код в продакшен.
- UI перестает быть точкой отказа для execution и risk.
- Появляется clean-room интерфейс между моделями, операционным управлением и торговым исполнением.
- Языковая реализация может меняться по фазам, но организационная и процессная граница должна быть постоянной.
- Языковая стратегия для MVP отдельно уточняется в `ADR 0004`.

## Последствия

Положительные:

- лучшее разделение ответственности;
- меньше operational risk на critical path;
- проще тестировать parity между replay и live.

Отрицательные:

- выше стартовая архитектурная сложность;
- нужен дисциплинированный schema governance;
- нужно поддерживать отдельные API/contracts между контурами.
