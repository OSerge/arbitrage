# ADR 0002: писать core execution, risk и simulation самим

- Статус: accepted
- Дата: 2026-05-21

## Контекст

На рынке много библиотек для retail backtesting и prototyping, но они плохо подходят как фундамент institutional-style statarb platform:

- у них нет нужного audit trail;
- они редко обеспечивают research/live parity;
- они не знают про конкретику MOEX, брокерские события и внутренний risk governance;
- их execution models часто слишком наивны.

## Решение

Собственным core framework считаются:

- broker/direct adapters;
- canonical event schema;
- OMS/EMS;
- position and ledger services;
- risk guards;
- event-driven simulator/replay;
- portfolio construction layer;
- reconciliation and audit trail.

Open-source стек используется только для:

- econometrics и ML;
- storage и observability;
- orchestration;
- notebooks и developer tooling.

## Почему так

- именно эти компоненты определяют переносимость research в live;
- именно здесь лежит основная институциональная дифференциация;
- именно эти части критичны для надежности и контроля риска.

## Последствия

Положительные:

- больше контроля над моделью исполнения;
- проще поддерживать одинаковые contracts между backtest, replay и live;
- ниже зависимость от retail-framework решений.

Отрицательные:

- выше стоимость разработки;
- дольше time-to-first-live;
- потребуется сильная инженерная дисциплина на schema и testing уровне.
