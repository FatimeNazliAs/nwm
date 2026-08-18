"""
Find good RECON scenes to point a probe at.

A prediction is only interesting to look at if something *changes* between "now"
and the frame we ask for. This script scans the 500-row RECON eval index and
measures, for every row, what the robot actually did over the next 16 seconds:

  * how far it drove              (path length, metres)
  * how much it turned            (total |yaw change|, degrees)
  * how different the picture got (mean |pixel difference| between now and target)

then prints the scenes that stand out, with a ready-to-paste config block.

Poses come straight from each trajectory's traj_data.pkl, so the whole scan is
cheap. Images are only loaded for the shortlist.

    cd /app && python debug/t4/scenes.py              # default: top 5 per category
    cd /app && python debug/t4/scenes.py --sec 8      # judge over an 8 s horizon
    cd /app && python debug/t4/scenes.py --top 10

Nothing here is required to run a probe -- it is a lookup tool. Any scene you
pick yourself works exactly the same, as long as it passes the two validity
rules the probes already enforce (printed at the end).
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import yaml
from PIL import Image

import misc
from datasets import get_data_path
from debug.common.report import hr
from debug.common.scene import build_recon_eval_dataset

INPUT_FPS = 4          # isolated_nwm_infer.py argparse default


def wrap(a):
    """Angle difference into (-pi, pi], so a spin never counts as ~2*pi."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=int, default=16,
                    help="horizon to judge scenes over, in seconds (max 16)")
    ap.add_argument("--top", type=int, default=5, help="how many to show per category")
    args = ap.parse_args()

    with open("config/eval_config.yaml") as f:
        base = yaml.safe_load(f)
    with open("config/nwm_cdit_xl.yaml") as f:
        base.update(yaml.safe_load(f))

    ds = build_recon_eval_dataset(base)
    ts = args.sec * INPUT_FPS
    if ts > ds.len_traj_pred:
        raise SystemExit(f"--sec {args.sec} needs {ts} future steps but the dataset only builds "
                         f"{ds.len_traj_pred} ({ds.len_traj_pred // INPUT_FPS} s). Use --sec "
                         f"{ds.len_traj_pred // INPUT_FPS} or less.")

    hr(f"SCANNING {len(ds)} RECON EVAL SCENES  (horizon {args.sec} s = {ts} steps)")

    traj_cache = {}
    rows = []
    for i in range(len(ds)):
        f_curr, curr_time, _, _ = ds.index_to_data[i]
        curr_time = int(curr_time)
        if f_curr not in traj_cache:
            traj_cache[f_curr] = ds._get_trajectory(f_curr)
        td = traj_cache[f_curr]
        pos, yaw = np.asarray(td["position"]), np.asarray(td["yaw"]).squeeze()
        if curr_time + ts >= len(pos):
            continue
        seg = pos[curr_time:curr_time + ts + 1]
        yseg = yaw[curr_time:curr_time + ts + 1]
        steps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        turns = np.abs(wrap(np.diff(yseg)))
        rows.append({
            "row": i,
            "traj": f_curr,
            "time": curr_time,
            "path_m": float(steps.sum()),
            "net_m": float(np.linalg.norm(seg[-1] - seg[0])),
            "turn_deg": float(np.degrees(np.abs(wrap(yseg[-1] - yseg[0])))),
            "wiggle_deg": float(np.degrees(turns.sum())),
        })

    print(f"  usable rows: {len(rows)} of {len(ds)}")
    p = np.array([r["path_m"] for r in rows])
    t = np.array([r["turn_deg"] for r in rows])
    print(f"  distance driven in {args.sec}s : median {np.median(p):.1f} m, "
          f"max {p.max():.1f} m")
    print(f"  net turn in {args.sec}s        : median {np.median(t):.0f}°, "
          f"max {t.max():.0f}°")

    # ---- how much does the PICTURE change? only for the shortlist ----
    shortlist = {}
    for key, sel in (
        ("turning the most", sorted(rows, key=lambda r: -r["turn_deg"])[:args.top * 2]),
        ("driving the furthest", sorted(rows, key=lambda r: -r["path_m"])[:args.top * 2]),
        ("barely moving", sorted(rows, key=lambda r: (r["path_m"], r["turn_deg"]))[:args.top * 2]),
    ):
        shortlist[key] = sel
    seen = {}

    def pixel_change(r):
        k = (r["traj"], r["time"])
        if k in seen:
            return seen[k]
        try:
            a = misc.transform(Image.open(get_data_path(ds.data_folder, r["traj"], r["time"])))
            b = misc.transform(Image.open(get_data_path(ds.data_folder, r["traj"], r["time"] + ts)))
            v = float((misc.unnormalize(a) - misc.unnormalize(b)).abs().mean())
        except Exception:
            v = float("nan")
        seen[k] = v
        return v

    for sel in shortlist.values():
        for r in sel:
            r["pix"] = pixel_change(r)

    def show(title, sel, note):
        hr(title)
        print(f"  {note}\n")
        print(f"  {'sample':>6}  {'trajectory':<40} {'t':>4} {'drove':>7} {'turned':>7} {'picture':>8}")
        for r in sel[:args.top]:
            print(f"  {r['row']:>6}  {r['traj']:<40} {r['time']:>4} "
                  f"{r['path_m']:>6.1f}m {r['turn_deg']:>6.0f}° {r['pix']:>7.3f}")

    show("A) BIG TURNS  -- the action has the most to say here",
         sorted([r for r in rows if "pix" in r], key=lambda r: -r["turn_deg"]),
         "The robot swings the camera round. Best for seeing the action matter.")
    show("B) LONG DRIVES  -- the scene translates past the camera",
         sorted([r for r in rows if "pix" in r], key=lambda r: -r["path_m"]),
         "Straight-ish and fast. Good for watching things approach and pass.")
    show("C) THE CALMEST ON OFFER  -- a control",
         sorted([r for r in rows if "pix" in r], key=lambda r: (r["path_m"], r["turn_deg"])),
         f"Not still -- the quietest scene here still drives {p.min():.1f} m in {args.sec} s. "
         f"The eval index only\n  keeps rows with real motion, so there is no static example "
         f"to compare against.")

    # the single best all-rounder: turns AND moves AND looks different
    scored = [r for r in rows if "pix" in r and not np.isnan(r["pix"])]
    for r in scored:
        r["score"] = (r["turn_deg"] / max(t.max(), 1e-9)
                      + r["path_m"] / max(p.max(), 1e-9)
                      + 2 * r["pix"] / max(max(x["pix"] for x in scored), 1e-9))
    best = sorted(scored, key=lambda r: -r["score"])[:args.top]

    hr("RECOMMENDED  -- moves, turns, and looks properly different")
    for r in best:
        print(f"\n  sample {r['row']}  ·  drove {r['path_m']:.1f} m  ·  turned {r['turn_deg']:.0f}°  "
              f"·  picture changed {r['pix']:.3f}")
        print(f"    trajectory: {r['traj']}   time: {r['time']}   "
              f"(target frame {r['time'] + ts})")

    hr("PASTE ONE OF THESE INTO debug/t4/config.yaml")
    r = best[0]
    print(f"""
  trajectory: {r['traj']}
  time: {r['time']}
  sec: {args.sec}
""")
    print("  ...or just point at the row number and leave `trajectory` blank:\n")
    print(f"  sample: {r['row']}\n  trajectory:\n  time:\n  sec: {args.sec}\n")

    hr("PICKING YOUR OWN")
    print(f"  Any scene works. Two rules, both already enforced by the probes:")
    print(f"    1. time >= {ds.context_size - 1}          "
          f"-- needs {ds.context_size} context frames ending at `time`")
    print(f"    2. time + {ds.len_traj_pred} <= last frame  "
          f"-- EvalDataset always builds the full "
          f"{ds.len_traj_pred // INPUT_FPS} s horizon, whatever `sec` you ask for")
    print(f"  Rule 2 does NOT depend on `sec`, so a scene valid at sec=1 is valid at sec=16.")
    print(f"  Break either and the probe stops with a message telling you the valid range.")
    print(f"\n  `sec` may be any of 1, 2, 4, 8, 16 (what the real eval sweeps), or any integer")
    print(f"  up to {ds.len_traj_pred // INPUT_FPS}. It sets ts = sec x {INPUT_FPS}, which drives both")
    print(f"  the summed action and rel_t = ts/128.")
    print(f"\n  Trajectory names live in data_splits/recon/test/traj_names.txt.")


if __name__ == "__main__":
    main()
