---
name: research-alpha-review
description: Review research papers, alpha proposals, and strategy notes against the statarb MVP scope. Use when the user mentions a paper review, alpha proposal, pairs trading, baskets, cointegration, VECM, regime filters, or asks whether a research idea fits the approved MVP.
---

# Research Alpha Review

## Goal

Turn a paper, note, or alpha idea into a short project decision: `fits now`, `later`, or `reject for MVP`.

## Workflow

1. Normalize the idea:
   - alpha family;
   - target horizon and instruments;
   - required inputs and outputs;
   - whether it changes research, risk, execution, or ops scope.
2. Check fit against the current MVP:
   - default family is `pairs/baskets mean reversion + cointegration/VECM + simple regime filters`;
   - paper-first and replay-first remain mandatory;
   - anything that implies a new alpha family or live-policy change requires approval.
3. Separate three layers:
   - research-only implications;
   - contract or data implications;
   - runtime or risk implications.
4. Flag the approval gates explicitly if the idea changes:
   - alpha family;
   - target-position logic;
   - risk assumptions;
   - live scope;
   - storage or infra dependencies.

## Output

Use this structure:

```markdown
## Verdict
fits now | later | reject for MVP

## Thesis
- ...

## Required artifacts
- docs/contracts:
- code/tests:
- runbook/ops:

## Risks
- ...

## Approval gates
- ...

## Recommended next step
- ...
```

Keep the output concise and decision-oriented.
