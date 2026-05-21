# Project Map

## Purpose

This file gives agents a compact map of the repository, the intended MVP boundaries, and the default placement rules for new work.

## Working Context

- This repository is operated by one founder with AI agents.
- The current slice is the `agent-operated` MVP, starting with the governance layer.
- `core/` remains the legacy research/prototype contour.
- `statarb/` is the reserved new platform shell for MVP code that follows the approved contracts and boundaries.
- `docs/superpowers/` is the home of the `Agent Operating System` artifacts.

## Source Of Truth

Use this order when deciding what is allowed:

1. Accepted `ADR` documents in `docs/adr/`.
2. MVP design constraints in `docs/superpowers/specs/2026-05-21-agent-operated-mvp-design.md`.
3. Governance artifacts in `docs/superpowers/`, especially this file and `approval-matrix.md`.
4. Execution sequencing in `docs/superpowers/plans/2026-05-21-agent-operated-mvp-implementation-plan.md`.
5. Existing code.

Until dedicated contract docs exist, the design spec plus the governance artifacts define the working contract for Phase 1.

## Repository Zones

### `core/`

- Legacy research/prototype package.
- Safe work: contained bug fixes, behavior-preserving cleanup, documentation, tests, and reference reading for migration.
- Not the home for new `agent-operated` platform layers.

### `statarb/`

- New platform shell for upcoming `statarb` modules.
- New MVP runtime, adapters, contracts-aligned services, and control-plane code should land here in later phases.
- Do not split a new architectural slice across `core/` and `statarb/` without a documented migration reason.

### `docs/`

- Durable architecture, specs, plans, ADRs, and operating docs.
- If architecture or runtime semantics move, the matching docs move in the same change.

### `docs/superpowers/`

- Governance layer for agent work.
- Expected contents: `project-map`, `approval-matrix`, contracts, runbooks, and validation playbooks.
- New agent-facing governance artifacts belong here by default.

### `.cursor/rules/`

- Persistent project-local invariants for Cursor agents.
- Keep rules short, practical, and aligned with approved docs.

### `.cursor/skills/`

- Repeatable project-local workflows that are costly to restate each session.
- Keep only repo-specific workflows here; do not duplicate generic built-in skills.

### `tests/`

- Verification layer for both the legacy code and the new shell as it appears.
- New behavior should bring focused tests or an explicit reason why a test is not yet valuable.

### `notebooks/`

- Research workspace, not the source of truth for architecture or runtime semantics.
- Promote durable conclusions into docs, contracts, or code before treating them as project policy.

### `data/`

- Local and historical data artifacts.
- Use it as storage input/output, not as the place where architectural decisions live.

## Placement Rules

- New governance docs, contracts, runbooks, and playbooks: `docs/superpowers/`
- Durable architectural decisions: `docs/adr/`
- New MVP platform modules: `statarb/`
- Legacy behavior fixes needed to preserve existing research flow: `core/`
- Project-local Cursor rules: `.cursor/rules/`
- Project-local agent workflows: `.cursor/skills/`

Do not add a new top-level directory without explicit approval.

## Escalate Before Doing

- Changing the `core/` versus `statarb/` boundary.
- Introducing a new platform layer or cross-cutting abstraction boundary.
- Changing runtime semantics without the matching governance updates.
- Any work that could imply live trading enablement or weaker guardrails.

## Session Start For Agents

1. Read `docs/README.md`.
2. Read the current MVP design spec and implementation plan in `docs/superpowers/`.
3. Read `docs/superpowers/project-map.md` and `docs/superpowers/approval-matrix.md`.
4. Check whether the task touches architecture, contracts, risk policy, or live-related behavior.
5. Update the governance layer first when the task changes repo boundaries or operating rules.
