# Domain Contracts

Статус: foundation

Этот каталог фиксирует канонические MVP-контракты для `statarb/` в формате `docs-first`.

## Правила

- source-of-truth: эти документы и согласованный MVP design, а не ad hoc код.
- compatibility-mode: additive-first; breaking и boundary changes требуют отдельного approval.
- replay-rule: execution и ops-события должны сохранять shape, пригодный для paper, replay и adapter mapping.
- live-rule: live-специфика допустима только через явные поля `contour` или `environment`; скрытых live-веток в контрактах быть не должно.

## Индекс

- `agent-operating-system.md` - общие contract conventions, authority hierarchy, envelope semantics.
- `domain-data.md` - reference/data contracts для instrument и dataset lineage.
- `domain-research.md` - research run и research intent contracts.
- `domain-execution.md` - portfolio target, order/fill lifecycle, position/account snapshots.
- `domain-ops.md` - replay, approvals, operator-state contracts.

## Python binding

- `statarb/domain/instruments.py`
- `statarb/domain/research.py`
- `statarb/domain/orders.py`
- `statarb/domain/positions.py`
- `statarb/domain/events.py`

## Change policy

- additive clarification: update docs and tests in the same change.
- breaking contract: stop for approval before code.
- new runtime behavior: update the matching contract doc before implementation.
