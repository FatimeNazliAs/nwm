# ADR-0003 — Stages never import from each other

**Status:** accepted · **Date:** 2026-08-18

## Context

Consecutive stages overlap: T3 encodes latents that T4 consumes, T4's model is
what T5 loops over. Importing T3's probe from T4's would avoid duplication.

## Decision

`debug/t<N>/` may import from `debug/common/` and from upstream, **never from
`debug/t<M>/`**. Anything two stages need moves into `debug/common/`
(`page.py`, `scene.py`, `report.py`, `update.sh`).

## Consequences

- A finished, published stage cannot be broken by work on a later one. This is
  the main point: stages are published to advisors and must stay stable.
- Genuinely shared logic must be promoted deliberately rather than reached for.
- Where a later stage needs an earlier stage's operation (T4 needs T5's
  `q_sample` to build a realistic mid-loop input), it calls **upstream** directly
  and labels the borrowing on the page.
