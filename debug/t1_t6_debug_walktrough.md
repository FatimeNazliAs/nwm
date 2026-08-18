# NWM code walkthrough — T1 to T6 — state of play

**What this file is:** a point-in-time summary of the `debug/` walkthrough — what each stage does,
where its code, Notion page and published page live, and what is still open. It is a snapshot, not
a living index: the authoritative sources are `CONTEXT.md` (vocabulary), `../docs/adr/` (decisions)
and the git history. Re-date it when you revise it.

**Written:** 2026-08-18 · **Status:** all six stages built. The walkthrough proper is complete.

---

## What this project is

A **learning** track, not a feature build. The goal is to understand the
[NWM (Navigation World Models)](https://arxiv.org/abs/2412.03572) research codebase by following
**one RECON sample** through the pipeline in data-flow order, running the real code at every step
and producing one visual per step.

Scope, fixed at the start: **Predict / one-shot / RECON**. Deferred and still deferred: rollout,
standalone planning, ranking, go_stanford.

The pipeline being followed:
`data → VAE encode → CDiT denoise → diffusion loop → VAE decode → predicted frame → metrics`

---

## Where things live

| | |
| --- | --- |
| Repo | `/home/nazli/projects/nwm` — https://github.com/FatimeNazliAs/nwm |
| Branch | `debug/nwm-walkthrough` (off `integration`, which is the main branch) |
| Last code commit | `637fcdd` — *feat(debug): add the T6 inference-and-metrics walkthrough* |
| Container | `nwm_debug` — persistent and running. Repo at `/app`, data at `/data`. **Do not rebuild.** |
| Checkpoint | `0100000.pth.tar` (4 GB, CDiT-XL/2, 1012 M params) at `/data/logs/nwm_cdit_xl/checkpoints/` |
| Plan file | `~/.claude/plans/nwm-debug-walkthrough.md` — the T1–T6 plan and every convention |
| Glossary | `CONTEXT.md` in the repo root |
| Decisions | `docs/adr/0001`…`0005` |
| Notion — NWM root | https://app.notion.com/p/NWM-Navigational-World-Models-393d0eaafe5580dcb5d9d0c446a2129c |
| Notion — Debug root | https://app.notion.com/p/Debug-Phase-5-with-cladue-chat-in-vs-code-3c0d0eaafe55802cbc72cad5e41ce8d6 |
| Notion — General plan | https://app.notion.com/p/General-Plan-T1-T6-3a2d0eaafe55803e9f49e904b4b50c00 |

**Notion structure.** *Debug (Phase 5 with cladue chat in vs code)* is now its own page under the
NWM root, and it holds *General Plan - T1…T6* plus the six stage pages T1–T6. Every stage page keeps
its own ID, so the per-stage links below are unaffected by the move.

Thirteen commits on the branch, `13b994b` (T1) through `637fcdd` (T6). All six stages are
committed and pushed to `origin/debug/nwm-walkthrough`, which is a **public** fork of
`facebookresearch/nwm` — so the Notion and artifact links below are visible to anyone who finds the
repo. The pages themselves stay private; only the URLs are public.

---

## The stages at a glance

| Stage | Subject | Code | Notion | Webpage |
| --- | --- | --- | --- | --- |
| **T0** | The paper in one page — what NWM is, before any code | `debug/t0/` | — | [artifact](https://claude.ai/code/artifact/ea04dacd-05a3-44d4-9e91-13e80950f2a0) |
| **T1** | Bird's-eye map of the pipeline | `debug/t1/trace.py` | [T1](https://app.notion.com/p/3a5d0eaafe558052a58bf581ce4e12dc) | [artifact](https://claude.ai/code/artifact/2a64ac6a-da9c-431d-9a3f-f41c29d26f3d) |
| **T2** | What one RECON sample is | `debug/t2/` | [T2](https://app.notion.com/p/3a8d0eaafe558090a3e3d5dd7caf8478) | [artifact](https://claude.ai/code/artifact/f25c8202-9bc3-4672-aed3-6963633a5d96) |
| **T3** | The VAE — pixels ↔ latents | `debug/t3/` | [T3](https://app.notion.com/p/3a8d0eaafe55818aad25e0e42be7e723) | [artifact](https://claude.ai/code/artifact/67ff8417-b8ab-47b7-9a95-b27f9c8af427) |
| **T4** | CDiT — the model that predicts | `debug/t4/` | [T4](https://app.notion.com/p/3abd0eaafe55809781b9db47f33ef023) | [artifact](https://claude.ai/code/artifact/b6cd5f71-733b-43a3-bc73-f7fd52739035) |
| **T5** | The diffusion loop | `debug/t5/` | [T5](https://app.notion.com/p/3c0d0eaafe5580178880c90713877429) | [artifact](https://claude.ai/code/artifact/9c9379da-52b3-4163-8a6f-89f4472aa5a4) |
| **T6** | Full inference and metrics | `debug/t6/` | [T6](https://app.notion.com/p/3c0d0eaafe5580629420d69a358ca3b8) | [artifact](https://claude.ai/code/artifact/30a9d5de-f7a6-4da1-bad8-d2c66370eeaa) |

**T0 is not a probe stage.** Its page is hand-written and static — no `config.yaml`, no `probe.py`,
no `build_page.py`. `./debug/t0/update.sh` only copies `debug/t0/latest.html` into `debug/out/t0/`
and starts the same server, so it is still viewed at port 8000, path `/t0/latest.html`.

**Artifact URLs are fixed per stage.** Always republish with `url=` or the link already given to
advisors silently keeps showing the old run. There is a stale duplicate T4 artifact at
`029fd697-a5dd-41dc-88c1-8f4ff8cfb14c` (minted 2026-08-05) — **do not use it**; `b6cd5f71` is live.

---

## What each stage actually does

**T1 — the map.** `trace.py` runs one sample through every boundary of the pipeline, printing shape
and range at each and saving images. Written before the probe/page split existed, so it is a single
script with no page builder.

**T2 — the data.** Loads one RECON sample and prints every field: 4 context frames, the target
frame, the action (relative pose + yaw), the time offset. Writes `scene_facts.json`.

**T3 — the VAE.** Encodes one frame to a `(4, 28, 28)` latent and decodes it back, side by side with
the original. Names the model (`stabilityai/sd-vae-ft-ema`, frozen, **not** trained by NWM). Writes
`vae_facts.json`.

**T4 — CDiT.** Instantiates CDiT-XL/2 with the real checkpoint, runs one forward pass on real
latents, hooks the attention blocks and traces shapes through them. Includes `find_scenes.py`, a
search tool over all 500 eval scenes. Writes `cdit_facts.json` (~85 keys).

**T5 — the diffusion loop.** Runs the real 250-pass reverse sampler three times on one scene: once
normally, once with the action zeroed, once from different starting noise. Keeps all 251
intermediate latents and decodes a filmstrip. Writes `loop_facts.json` (~71 keys).

**T6 — inference and metrics.** Predicts one scene at all five horizons (1, 2, 4, 8, 16 s) through
the real inference call, writes the PNGs with the real writer, then scores **all 500 finished test
scenes** with the evaluation's own scoring loop. Writes `eval_facts.json` (~56 keys).

---

## The architecture that emerged

Every stage from T2 onwards is split at the **facts** seam:

- `probe.py` runs the real model and writes `<scene>/*_facts.json` plus PNGs.
- `build_page.py` reads **only** those files and writes HTML. It never imports the model.

That split is [ADR-0001](../docs/adr/0001-probe-page-split.md). It exists because page iteration used
to cost a ten-minute model load, and because a number quoted in prose could silently drift from
what the code does.

**Shared machinery** in `debug/common/` (used by T4, T5, T6):

| Module | What it gives |
| --- | --- |
| `model.py` | `load_config`, `build_cdit`, `build_vae`, `build_diffusion`, `encode`/`decode`, `to_display`, `context_latents`, `action_and_horizon`, and every upstream constant |
| `facts.py` | The only module allowed to read or write a facts file. Versioned, validated, tracks which keys were read |
| `images.py` | `Saver` — writes PNGs and builds the manifest **as a side effect of writing**, so it cannot disagree with what is on disk |
| `scene.py` | Builds the real eval dataset and resolves which scene a config names |
| `page.py` | The shared page shell — one look across T1–T6 |
| `update.sh` | The whole docker/conda/serve dance; each stage has a one-line wrapper |

**The five decisions on record** (`docs/adr/`):

1. Facts are the only channel between a probe and a page.
2. Probes import the real inference path, never re-implement it.
3. Stages never import from each other — anything shared moves to `debug/common/`.
4. The facts file is a validated, versioned interface.
5. A probe stays one linear narrated procedure (its console output is a deliverable, not logging).

An architecture pass over T6 (2026-08-18) applied three fixes: `check_run` no longer returns a
value it can never deliver, the example-folder lookup filters to directories the way `check_run`
already does, and a `both(a, b)` closure absorbs the single-element-list wrapping that upstream's
batch-shaped loss functions forced on eight call sites. The probe was re-run afterwards and the
facts file came out identical apart from wall-clock timings, which is the refactoring check
ADR-0005 prescribes.

---

## Tests

```bash
./debug/tests/run_all.sh        # 40 tests, 7 files, ~13 seconds
```

No GPU, no checkpoint, no dataset — every test runs against a committed facts file from a real probe
run, with stub PNGs.

| File | Tests | Covers |
| --- | --- | --- |
| `test_t4_page.py` | 4 | T4 page from fixture |
| `test_t5_page.py` | 4 | T5 page from fixture |
| `test_t5_config.py` | 7 | every T5 knob refused before the GPU |
| `test_t5_loop.py` | 6 | the Run/Pass alignment, driven by a fake diffusion |
| `test_t6_page.py` | 5 | T6 page; includes a test that the winning metric is read from facts, not hard-coded |
| `test_t6_config.py` | 6 | every T6 knob, including that only the five real horizons are accepted |
| `test_t6_run.py` | 8 | run-completeness check, metric readout against a fake logger, pixel diff |

---

## The headline result (T6)

The full RECON `time` evaluation, 500 scenes × 5 horizons = 2,500 pairs, scored in ~75 seconds:

| how far ahead | LPIPS | DreamSim | FID |
| --- | --- | --- | --- |
| 1 s | 0.2568 | 0.0934 | 25.0 |
| 2 s | 0.2815 | 0.0995 | 25.4 |
| 4 s | 0.3059 | 0.1068 | 26.4 |
| 8 s | 0.3407 | 0.1204 | 26.4 |
| 16 s | 0.4133 | 0.1698 | 31.2 |

Lower is better throughout. From 1 s to 16 s: LPIPS +61 %, DreamSim +82 %, FID +25 %. The measure
that cares about layout degrades fastest; the measure that only cares about realism degrades
slowest.

**The argument T6 is built around.** The same moment predicted three times with one input changed
each time, scored against the true frame:

| | plain pixel difference | LPIPS | DreamSim |
| --- | --- | --- | --- |
| the real prediction | 0.0889 | **0.2942** | **0.0864** |
| the action removed | **0.0798** | 0.3035 | 0.0871 |
| a different random start | 0.1055 | 0.3573 | 0.0894 |

Subtracting the pictures picks the run that was told **nothing** about where the robot went — which
cannot be the best account of where the journey ended. LPIPS and DreamSim, built independently,
both pick the real prediction. That is why the evaluation uses them and not a plain difference.

Two structural facts worth keeping: inference and scoring never share a tensor (the handover is PNG
files on disk, which is why 5 hours of prediction scores in 75 seconds), and the 500-scene run is
**already on disk** at `/data/logs/nwm_eval/` — nothing regenerates it.

---

## Conventions that bind any future session

These are recorded in the plan file and in the `walkthrough-doc-convention` memory. The user has
explained them many times and should not have to again.

**How to explain anything**
1. Ordered, not topical — a numbered walk through what happens to the data, including steps owned by
   other stages marked "(not this step)".
2. Real measured instance first, then why it is built that way.
3. No forward references. Unavoidable dependencies get one line plus a "Deferred on purpose → TN".
4. Always name the specific model and checkpoint, and whether NWM trains it.
5. Show the real multiplicity — 4 context frames means show 4.
8. **Prose does not work. Numbered lists only.** Said many times.
7. Headings must say where you physically are, not name an idea.
8. Define a term before using it.
9. Every comparison picture needs its reference beside it, and the caption must say what to compare
   with what.

**The webpage**
- Open with the visual. A diagram of labelled boxes is not an architecture.
- Structure the user converged on over four rounds on T5: a short "In short" section first (three
  real images, five numbered lines) → a worked example before step 1 → the numbered beats, 3–6 items
  each → `<details>` for supporting argument only, never as structure.
- Roughly 800 visible words. T6 is the exception — it covers the whole process and may be longer.
- No fundamentals, no how-to-run, no commands.
- The pages say **"pass"** where the code says "step", and **"square"** where the code says "token".

**Notion tree, per stage**
```
T<N> — <title>            ← general idea only, plus the artifact link
  ├── Overview — what X is and does
  ├── How it works — X in detail        ← the numbered walk
  ├── Implementation — how to run T<N>  ← short parent + Republish section
  │     ├── Run it with one command (T<N>)
  │     └── Run it by hand (T<N>)
  └── Pick a different data example (T<N>)
```
- Notion appends children in creation order, so create Implementation **before** "Pick a different
  data example".
- **No code on Overview or How it works** — no snippets, no `file.py:NN`. Code lives only on
  Implementation pages.
- Every Implementation page needs a Republish section with **both** the file path and the fixed
  artifact URL.
- Never put `localhost:` links in Notion.
- **Never publish problems** (review findings, error reports) as a webpage — report in chat.
- Do **not** create `debug/tN/NOTION_implementation.md`. T3 and T4 have one; the user declined it
  for T5 and T6.

**Operating a stage**
Edit the three lines in `debug/tN/config.yaml` → run `./debug/tN/update.sh` → refresh the browser at
port 8000, path `/tN/latest.html`. VS Code Remote-SSH forwards the port; `file://` cannot work
because the file is on the server and the browser is on the laptop.

---

## Open items

1. **`whole()` is duplicated three times.** The same six-line integer-knob reader sits
   byte-identical in `debug/t4/probe.py`, `debug/t5/probe.py` and `debug/t6/probe.py`. Three users
   is well past the "two users make a real seam" line ADR-0005 closes on, so it wants promoting to
   a `debug/common/config.py` alongside the `folder()` check and the shared error phrasing. Not
   done, because it is all-or-nothing: promoting it for T6 alone would leave four copies rather
   than three, and fixing all three touches two committed stages, one of which (T4) is published.
   Needs a decision before anyone starts.
2. **The `sweep` record's shape is defined in three places** in `debug/t6/probe.py` — built in
   section 2, given its scores in section 4, and stripped of its path keys when the facts are
   written. The ordering is deliberate (the metric networks load after prediction on purpose), so
   this is a readability cost rather than a bug. Low value, listed so it is not rediscovered.
3. **T2 and T3 predate the shared machinery.** Both still use raw `json.dump`/`json.load` instead of
   `debug/common/facts.py`, and neither has a committed fixture or a page test. They work; they are
   just the last two stages not on the current architecture.
4. **T4's page still says "step" where T5 and T6 say "pass"** — a known, accepted inconsistency.
   Sweeping it needs permission, because T4's page is already published to advisors.
5. **The paper is not cited anywhere.** The convention says to anchor "How it works" to the paper's
   own figure. T5 and T6 both skipped it: the PDF is not in the repo and the arXiv HTML invents
   figure labels. To do it properly, fetch the **PDF** (arXiv 2412.03572) and read it with
   `Read(pages=…)`. Never cite a section or equation number you have not read.
6. **T5's Notion page may contain a wrong claim.** Its "How it works" says the action-removed run
   "produces something close to the starting view: building still large, still near the middle."
   T6 measured this on the real inference path and it does **not** hold — all three comparison
   predictions sit further from the starting view (DreamSim 0.156–0.182) than the true frame does
   (0.118). T5's probe runs fp32 and T6 runs the real bf16 path, so this may be arithmetic or may be
   a claim that was eyeballed and never true. **Not yet investigated.** T6 makes no such claim.
7. **32 of T6's 56 facts are read by no code** (`python debug/t6/build_page.py --audit`). Not dead —
   several are quoted by hand when writing Notion, and `truth_from_start` exists purely as a guard
   against item 6 creeping back in. Do not "clean them up" without asking.
8. **Remaining optional work:** Appendix A (training, read-only) and Appendix B (planning,
   read-only). Both are out of the main sequence and neither is started.

---

## Two mistakes made during this work — do not repeat them

1. **Describing images without opening them.** A caption was written claiming the action-removed run
   produced "a different place, a pole appears". Opening the PNGs showed the opposite. **Open every
   PNG and look before writing a caption about it.**
2. **Writing a conclusion the numbers contradict.** A sentence said "the last third barely moves the
   answer" when the measured split was 55.7 / 23.1 / **21.2**. **Read your own table before writing
   the sentence under it.**

---

## Suggested skills

- **`/phased-implementation`** — the standing way to run any T-step or follow-up: do the one current
  phase, show what changed, stop for review, never auto-advance, never commit unasked.
- **`/explain-in-steps`** — for every why / where-are-we / what's-next answer. The user finds
  abstract paragraphs frustrating; this makes answers numbered and concrete.
- **`/commit`** — only after explicit approval. Split unrelated concerns into separate commits.
- **`/grilling`** — the user used this to walk the whole T5 architecture decision tree and it worked
  well: one question at a time, each with a recommendation.
- **`/improve-codebase-architecture`** — worth running now that T6 has landed, particularly on
  items 3 and 4 above. Report findings **in chat**, never as a webpage.
- **`/code-review`** — for the uncommitted T6 diff before it is committed.
