# ADR-0002 — Probes import the real inference path, never re-implement it

**Status:** accepted · **Date:** 2026-08-18

## Context

A probe needs the same dataset, the same model construction and the same
diffusion schedule the real evaluation uses. Copying those few lines into each
probe is quicker to write and easier to read in isolation.

## Decision

Probes **import** from the upstream path (`isolated_nwm_infer.get_dataset_eval`,
`models`, `diffusion.create_diffusion`). Shared setup lives in
`debug/common/scene.py`. No probe re-implements upstream behaviour, even when the
re-implementation would be three lines.

## Consequences

- If upstream changes how it builds the dataset or the schedule, every probe
  follows automatically and the walkthrough cannot document behaviour the code no
  longer has.
- Probes inherit upstream's import cost and its constants.
- Values that upstream computes must be **read**, not assumed. The concrete case:
  the 250-step inference schedule has a stride of 999/249 = 4.012, so it visits
  0, 4, 8 … 995, 999 and **never visits 996**. An earlier probe assumed a round
  stride and published a wrong number. Probes now read
  `diffusion.timestep_map` and reject values that are not on it.
