# ADR-0004 — The facts file is a validated, versioned interface

**Status:** accepted · **Date:** 2026-08-18

## Context

[ADR-0001](0001-probe-page-split.md) made `*_facts.json` the only channel between
a probe and a page, but nothing enforced it. Both sides used `json.load` /
`json.dump` on a bare dict of ~85 keys. Consequences seen in practice:

- Renaming a key in a probe broke the page only at render time, only for scenes
  rebuilt afterwards. Older scene folders kept working, so the break was
  invisible until someone re-ran an old scene.
- Building a page needed a GPU, the checkpoint and a full probe run, so the
  fast half of the split was never actually fast to iterate on.
- The probe measured far more than any page displayed, and there was no way to
  see how much. An audit found **46 of 85 keys read by no code at all.**

## Decision

`debug/common/facts.py` is the only module allowed to read or write a facts
file. It provides `write()`, `load(path, required=...)` and `unused()`.

- Every file carries `_schema_version` and `_stage`.
- `load()` fails with a message naming the missing key and saying to re-run the
  probe. A page declares only the handful of keys it cannot start without —
  **not** all 85, which would make every probe change a page change.
- `load()` returns a mapping that records which keys were read, so
  `build_page.py --audit` can list what the probe measured and the page ignores.
- A real probe run is checked in at `debug/t4/fixtures/cdit_facts.json`, and
  `debug/tests/test_t4_page.py` builds the whole page from it with stub PNGs —
  no GPU, no checkpoint, well under a second.

## Consequences

- Page work is now testable and fast, which is what ADR-0001 was for.
- The 46 unread keys are visible. They are **not** dead: several are consumed by
  hand when writing the Notion pages. The audit makes that second, manual
  consumer explicit instead of invisible; it is a prompt to decide, not a
  failure to fix.
- One more indirection between a probe and its output. Accepted: the file is a
  published contract, not an implementation detail.
