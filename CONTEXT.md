# CONTEXT — domain language

This repo is the upstream **NWM (Navigation World Models)** research code plus a
`debug/` tree that is not part of NWM: it is a **walkthrough** built to understand
the model by running it. The glossary below covers the walkthrough. Upstream
model code (`models.py`, `diffusion/`, `isolated_nwm_infer.py`) keeps the paper's
own vocabulary and is not renamed here.

## The walkthrough

**Walkthrough** — the T1–T6 sequence that follows one sample through the NWM
pipeline in data-flow order. It is a *learning* exercise, not a feature: every
claim it makes must come from code that actually ran.

**Stage** — one step of the walkthrough, `T1`…`T6`, living in `debug/t<N>/`.
Stages are independent: **a stage never imports from another stage.** Anything
two stages both need moves to `debug/common/`.

| Stage | Subject |
| --- | --- |
| T1 | map of the pipeline |
| T2 | the data layer |
| T3 | the VAE |
| T4 | CDiT, the model that predicts |
| T5 | the diffusion loop |
| T6 | full inference and metrics |

## The three artifacts of a stage

**Probe** (`debug/t<N>/probe.py`) — builds the real model, runs real forward
passes, and writes what it measured. A probe never fabricates a number and never
re-implements upstream behaviour; it imports it.

**Facts** (`<scene>/cdit_facts.json` and friends) — the measured record a probe
writes. **Facts are the interface between a probe and everything downstream.**
See [ADR-0001](docs/adr/0001-probe-page-split.md).

**Page** (`debug/t<N>/build_page.py` → `debug/out/t<N>/latest.html`) — the
published story of the stage, told with real images. It reads facts and PNGs and
**never runs the model.**

## Data vocabulary

**Scene** — *which* sample the walkthrough is looking at, named by three fields
in a stage's `config.yaml`: `trajectory` (which drive), `time` (the frame that
counts as "now"), and `sec` (how far ahead to predict). Resolved by
`debug/common/scene.py`.

**Context frames** — the 4 real frames ending at `time`. Clean; read, never
modified.

**Target frame** — the frame at `time + sec × fps` that the model is asked to
produce. During denoising it exists only as a **noisy latent**.

**Horizon** — `sec`, how far ahead the prediction reaches. RECON eval uses
1, 2, 4, 8 or 16 seconds.

**Latent** — a frame after the VAE encoder: `(4, 28, 28)` instead of
`(3, 224, 224)`. CDiT works only in latent space.

**Token / square** — one 2×2 tile of a latent, expanded to 1152 numbers. A frame
is 196 tokens. The pages say "square" where the code says "token".

## Scene tooling — two similarly named things

- `debug/common/scene.py` — **resolves** the scene a config names.
- `debug/t4/find_scenes.py` — **searches** all 500 eval scenes for interesting
  ones. A lookup tool; no probe depends on it.

## Decisions on record

| ADR | Decision |
| --- | --- |
| [0001](docs/adr/0001-probe-page-split.md) | Facts are the only channel between a probe and a page |
| [0002](docs/adr/0002-import-the-real-path.md) | Probes import the real inference path, never re-implement it |
| [0003](docs/adr/0003-stages-are-independent.md) | Stages never import from each other |
| [0004](docs/adr/0004-facts-schema.md) | The facts file is a validated, versioned interface |
| [0005](docs/adr/0005-probe-stays-linear.md) | A probe stays one linear narrated procedure |

## Running the tests

```bash
python debug/tests/test_t4_page.py
```

No GPU and no checkpoint — the page is built from a recorded facts file.
