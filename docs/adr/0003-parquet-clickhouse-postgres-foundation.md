# ADR 0003: foundation storage = MinIO/Parquet + ClickHouse + Postgres

- Статус: accepted
- Дата: 2026-05-21

## Контекст

Платформе нужен storage-контур, который:

- достаточно современный для event/data platform;
- не требует сразу тяжелого distributed lakehouse;
- хорошо подходит для time-series, replay и research datasets;
- разворачивается в self-hosted/российском контуре.

## Решение

На foundation-фазе принять следующий стек:

- `MinIO` для объектного хранения;
- `Parquet` для raw и research datasets;
- `ClickHouse` для fast analytical serving;
- `Postgres` для control plane.

Полноценный catalog-heavy lakehouse формат вводить только после реальной потребности в multi-engine concurrent table management.

## Почему так

- Parquet и MinIO закрывают replay и dataset versioning без лишней сложности.
- ClickHouse отлично подходит для screening, fast time-series analytics и monitoring workloads.
- Postgres естественно закрывает metadata/config/audit registry use cases.
- Такой стек проще внедрить и сопровождать в ранней стадии платформы.

## Последствия

Положительные:

- быстрый старт без платформенного оверинжиниринга;
- хороший баланс между research и production needs;
- понятная траектория эволюции к более тяжелому lakehouse позднее.

Отрицательные:

- часть table-governance возможностей придется реализовывать дисциплиной и tooling;
- при росте числа writers/readers позже может понадобиться более формальный catalog layer.
