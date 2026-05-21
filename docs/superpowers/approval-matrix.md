# Approval Matrix

## Purpose

This file defines what agents may change autonomously, what requires founder approval first, and which actions remain explicitly off-limits without written approval.

## Operating Rule

- `Founder` means the user.
- If a change fits more than one category, follow the strictest category.
- If approval is required, stop before implementation or before running the risky validation step.

## Autonomous By Default

Agents may proceed without additional approval when the work stays inside the already approved MVP boundaries:

- clarify or improve docs, rules, skills, `project-map`, and this approval matrix;
- draft or refine contracts, runbooks, and validation playbooks that do not expand architecture or trading scope;
- refactor inside an existing boundary without changing runtime semantics;
- add focused tests, fixtures, and validation checks;
- prepare research reviews, alpha notes, and proposal writeups without changing the approved alpha family;
- improve paper-only or read-only operator workflows.

## Approval Required Before Change

### Architecture And Boundaries

Approval is required before:

- adding a new top-level directory, package root, or platform layer;
- moving responsibility between `core/` and `statarb/`;
- introducing a new mandatory abstraction boundary, broker layer, or control-plane dependency;
- putting UI or notebook logic onto the runtime critical path.

### Contracts And Runtime Semantics

Approval is required before:

- adding a new domain entity or changing an event, state, or envelope shape in a breaking way;
- changing approval semantics, audit semantics, replay semantics, or operator action semantics;
- changing env naming conventions or the MVP secret-handling policy;
- widening the allowed behavior of broker-facing flows.

### Integrations, Tooling, And Infra

Approval is required before:

- adding new external services, storage systems, or mandatory infrastructure dependencies;
- adding a frontend toolchain or expanding UI scope beyond the already approved MVP slice;
- running any validation step that leaves safe read-only semantics for broker integration;
- performing a destructive migration or rewrite without a clear rollback path.

### Research And Trading Policy

Approval is required before:

- introducing a new alpha family or materially changing the approved research direction;
- changing target-position logic, cost assumptions, slippage assumptions, or risk limits;
- changing the intended scope of paper versus live behavior.

## Explicit Founder-Only Gates

Agents must not do the following without explicit written approval:

- enable live trading or expand live contour scope;
- run broker commands that can place, route, or alter real orders;
- raise risk limits, weaken kill-switch behavior, or reduce audit and replay coverage;
- bypass required governance updates for speed;
- publish, rotate, or relocate secrets outside the approved local workflow.

## Required Companion Artifacts

When approval is needed, prepare the smallest matching artifact set first:

- boundary or architecture change: update `project-map`, `approval-matrix`, and create or update an `ADR` if the decision is durable;
- contract or schema change: update the relevant contract docs plus tests or fixtures;
- runtime semantic change: update contracts, runbook notes, and verification coverage;
- broker workflow change: update the runbook and validation notes before any risky execution step.

## Handoff Requirements

Before calling work ready for review, the agent should state:

1. whether the work was autonomous or approval-first;
2. which docs, rules, skills, contracts, or tests changed;
3. what was verified and what was intentionally not executed;
4. which approval gates remain open.
