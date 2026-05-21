---
name: contract-change-workflow
description: Govern contract, schema, event-shape, boundary, and env-naming changes for the statarb MVP. Use when editing contracts, DTOs, event envelopes, approval semantics, operator actions, project-map boundaries, or any change that could break replay, paper, adapter, or control-plane compatibility.
---

# Contract Change Workflow

## Default Rule

Contract and boundary changes are `docs-first`. Do not start from code and backfill the docs later.

## Workflow

1. Classify the change:
   - non-breaking clarification;
   - additive change;
   - breaking change;
   - boundary change.
2. Identify the affected surfaces:
   - `docs/superpowers/contracts/`;
   - `docs/superpowers/project-map.md`;
   - `docs/superpowers/approval-matrix.md`;
   - `docs/adr/`;
   - code and tests.
3. Update the highest-order artifact first.
4. State compatibility explicitly:
   - what stays valid;
   - what breaks;
   - whether migration or dual-shape support is needed.
5. Require approval before proceeding if the change is:
   - breaking;
   - boundary-moving;
   - live-affecting;
   - changing env naming or approval policy.
6. After docs are aligned, update code, fixtures, and tests.

## Handoff Template

```markdown
## Change type
- clarification | additive | breaking | boundary

## Affected artifacts
- ...

## Compatibility
- ...

## Required approval
- none | needed because ...

## Verification
- docs:
- tests:
- intentionally not run:
```

If you cannot name the affected artifacts before coding, stop and ask.
