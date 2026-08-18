# ADR-0001 — Facts are the only channel between a probe and a page

**Status:** accepted · **Date:** 2026-08-18

## Context

Each walkthrough stage produces two things: measurements taken from the real
model, and a published page describing them. The obvious implementation is one
script that measures and emits HTML in the same pass.

Two problems with that. Every page tweak would re-run a billion-parameter model
(~10 minutes cold), and any number quoted in prose could silently drift away from
what the code does — which is fatal for a walkthrough whose only value is being
true.

## Decision

A stage is split at the **facts** seam.

- `probe.py` runs the model and writes `<scene>/*_facts.json` plus PNGs.
- `build_page.py` reads **only** those files. **It never imports the model.**

Facts are the interface. Anything the page states must exist as a measured key.

## Consequences

- Page iteration costs a second instead of ten minutes.
- A number cannot appear on a page unless a probe measured it.
- The page cannot show anything the probe did not think to record — adding a
  visual sometimes means a probe change and a re-run. Accepted.
- Facts are a real interface and are therefore validated — see
  [ADR-0004](0004-facts-schema.md).
