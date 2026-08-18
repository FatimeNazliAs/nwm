# ADR-0005 — A probe stays one linear narrated procedure

**Status:** accepted · **Date:** 2026-08-18

## Context

`debug/t4/probe.py`'s `main()` is ~1000 lines. The obvious criticism is that it
should be split into per-section functions — `section_model()`,
`section_patchify()`, `section_attention()` and so on — with a context object
threading the shared state.

That was considered and **rejected**.

## Decision

Probes keep a single linear `main()`. Only genuinely independent helpers are
lifted to module level.

Applying the deletion test to a proposed `section_*` function: deleting it would
put its body back where it is used, which is where it already is. It does not
concentrate complexity — it moves it and adds a call. The sections are not
independent: they share the model, the VAE, the diffusion schedule, `x`,
`x_cond`, `t`, `y`, `rel_t`, the grid geometry and the output directory. A
context object carrying that would have an interface about as complex as the
code behind it — a **shallow module**, which is the thing this repo's design
vocabulary says to avoid.

The probe is also read top to bottom by a person learning the model. Its console
output is a deliverable, not debug logging. **Locality** is the point.

What *was* extracted, because each is independent, reusable and testable on its
own:

- `parse_query_patch` — config string → grid index
- `self_attn_weights` — re-run timm attention to recover weights it discards
- `cross_attn_weights` — same for cross-attention, from hook captures
- `per_query_cos` — similarity between two attention patterns
- `overlay` — a heat map painted on a frame at a caller-supplied scale

Five closures remain inside `main()` (`rec`, `grab_self`, `grab_cross`,
`attention_profile`, `ablate`). Each is a hook or a re-run bound to this run's
model and tensors; hoisting them would mean passing six to eight arguments to
recreate the scope they already have.

## Consequences

- `main()` stays long. Navigation is by the `hr(...)` section banners, which
  match the numbered sections in the console output and on the page.
- Refactoring safety comes from the facts file, not from unit tests of sections:
  the whole probe is re-run and `cdit_facts.json` must come out **byte-identical**.
  That check caught nothing on the extraction above — the output was unchanged —
  and it is the check to repeat for any future change here.
- If a later stage (T5) needs the same model-and-scene setup, that is the point
  to extract it into `debug/common/` — two users make a real seam, one does not.
