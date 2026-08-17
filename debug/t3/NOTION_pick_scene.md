# Pick a different data example (T3)

Source of truth for the Notion child page of the same name. Edit here, paste there.

How to point T3 at a different RECON scene. Four steps, no rebuild.

---

## 1. Open the config

`debug/t3/config.yaml`

## 2. Choose the scene — two ways

**A — by row number** of the 500-row eval index (`data_splits/recon/test/time.pkl`):

```yaml
sample: 0            # 0..499
trajectory:          # leave blank
time:
```

**B — by trajectory name.** Setting `trajectory` **overrides** `sample`:

```yaml
trajectory: jackal_2019-08-02-16-28-30_5_r01
time: 6              # the "now" frame
```

Names come from `data_splits/recon/test/traj_names.txt`.

## 3. Two rules for a scene of your own

Both are checked before anything runs — you get a clear message, not a crash.

- **`time >= 3`** — four context frames are needed, ending at `time`.
- **`time + 64 <= last frame`** — the dataset always builds the full 16 s horizon,
  even though T3 only uses the context frames.

## 4. Optional — which frame gets measured

T3 encodes all 4 context frames, but the detailed round-trip measurement (PSNR,
error map) is done on one of them:

```yaml
frame: now           # "now" = the last context frame, or 0..3
```

`0` is the oldest, `3` is "now" when `context_size = 4`.

## 5. Re-run

```bash
cd /app && python debug/t3/probe.py && python debug/t3/build_page.py
```

See **Implementation** for the full run procedure.

---

The scene keys are the **same as T2's** — both stages read the same resolver
(`debug/common/scene.py`), so you can point T2 and T3 at the same scene and
compare them frame for frame.

Outputs land in a new folder per scene: `debug/out/t3/<trajectory>__t<time>/`.
Nothing overwrites earlier runs.
