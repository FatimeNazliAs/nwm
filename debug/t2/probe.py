"""
T2 data-layer probe for the NWM debug walkthrough.

Loads ONE real RECON eval sample through the *exact same* code path the
inference pipeline uses (EvalDataset + the time.pkl predefined index), then:
  1. prints the raw traj_data.pkl fields for that trajectory,
  2. prints every tensor the dataset returns, with shapes + dtypes,
  3. explains the action (delta) + how time is encoded (curr_delta / rel_t),
  4. renders an annotated visual: 4 context frames + the 5 target frames the
     "time" eval scores (sec = 1,2,4,8,16 -> frame 4,8,16,32,64).

Pick a scene by editing debug/t2/config.yaml (no rebuild), then run inside the
nwm_debug container:
    cd /app && python debug/t2/probe.py
Two ways to choose a scene (see debug/t2/config.yaml):
  * sample: N          -> row N of the predefined index data_splits/recon/test/time.pkl
  * trajectory: <name> -> load that folder directly at frame `time` (overrides sample)
Outputs to debug/out/t2/<scene>/ (gitignored).
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)   # repo root, so `import misc` / `datasets` resolve
os.chdir(ROOT)             # so config/, data_splits/, outputs resolve from repo root

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from debug.common.report import hr, show
from debug.common.scene import (
    build_recon_eval_dataset,
    resolve_scene,
)

import misc

# ---- eval "time" path constants (see isolated_nwm_infer.py) ----
INPUT_FPS = 4                       # recon is 4 fps
SECS = [1, 2, 4, 8, 16]             # 2**i for i in range(num_sec_eval=5)
TIMESTEPS = [s * INPUT_FPS for s in SECS]   # [4, 8, 16, 32, 64]


def main():
    cfg = yaml.safe_load(open("debug/t2/config.yaml")) if os.path.exists("debug/t2/config.yaml") else {}
    cfg = cfg or {}

    with open("config/eval_config.yaml") as f:
        base = yaml.safe_load(f)
    with open("config/nwm_cdit_xl.yaml") as f:
        base.update(yaml.safe_load(f))

    ds = build_recon_eval_dataset(base)
    n_predef = len(ds)   # size of the predefined time.pkl index (before any override)

    row, f_curr, curr_time, scene_tag, mode = resolve_scene(ds, cfg, "debug/t2/config.yaml")

    hr("SCENE SELECTED  (from debug/t2/config.yaml)")
    print(f"  mode         : {mode}")
    print(f"  trajectory   : {f_curr}")
    print(f"  curr_time    : frame {curr_time}   ('now')")
    print(f"  predefined index (data_splits/recon/test/time.pkl) has {n_predef} rows")

    # ---- 1. raw trajectory pkl ----
    hr("1) RAW traj_data.pkl  (what's on disk before any processing)")
    traj = ds._get_trajectory(f_curr)
    for k, v in traj.items():
        v = np.asarray(v)
        print(f"  {k:<10} shape={tuple(v.shape)!s:<12} dtype={v.dtype}  "
              f"e.g. curr_time row = {np.round(v[curr_time], 4)}")
    n_frames = len(traj["position"])
    print(f"  -> trajectory has {n_frames} frames (jpg images 0..{n_frames-1})")

    # ---- 2. the returned sample ----
    hr("2) EvalDataset.__getitem__  ->  (idx, obs_image, pred_image, delta)")
    idx, obs_image, pred_image, delta = ds[row]
    show("idx", idx)
    show("obs_image", obs_image)
    show("pred_image", pred_image)
    show("delta", delta)
    print("\n  obs_image  = context frames  (context_size, 3, 224, 224)")
    print("  pred_image = future frames held out for scoring (len_traj_pred, 3, 224, 224)")
    print("  delta      = per-step action (dx, dy, dyaw), normalized (len_traj_pred, 3)")

    # ---- 3. action + time encoding ----
    hr("3) THE ACTION  (delta) and HOW TIME IS ENCODED")
    print("  delta[:6] (first 6 per-step actions, normalized dx,dy | raw dyaw):")
    for i in range(6):
        print(f"    step {i:>2}: dx={delta[i,0]:+.4f}  dy={delta[i,1]:+.4f}  dyaw={delta[i,2]:+.4f}")
    print("\n  ONE-SHOT time query = sum the first `timestep` deltas (generate_time):")
    print(f"    {'sec':>4} {'timestep':>9} {'rel_t=ts/128':>13}   curr_delta = delta[:ts].sum  (dx, dy, dyaw)")
    for sec, ts in zip(SECS, TIMESTEPS):
        cd = delta[:ts].sum(dim=0)
        print(f"    {sec:>4} {ts:>9} {ts/128.0:>13.4f}   "
              f"({cd[0]:+.4f}, {cd[1]:+.4f}, {cd[2]:+.4f})")
    print("\n  -> larger horizon = more steps summed = bigger displacement +")
    print("     a bigger rel_t scalar. Time enters the model BOTH as the summed")
    print("     action magnitude AND as the rel_t embedding (timestep/128).")

    # ---- 4. visual ----
    hr("4) VISUAL  ->  debug/out/t2")
    out_dir = f"debug/out/t2/{scene_tag}"
    os.makedirs(out_dir, exist_ok=True)

    # undo the [-1,1] normalization for viewing
    obs_v = misc.unnormalize(obs_image).clamp(0, 1)          # (4,3,224,224)
    pred_v = misc.unnormalize(pred_image).clamp(0, 1)        # (64,3,224,224)

    ncols = max(ds.context_size, len(SECS))
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6.4))

    # row 0: context frames
    for c in range(ncols):
        ax = axes[0, c]
        ax.axis("off")
        if c < ds.context_size:
            ax.imshow(obs_v[c].permute(1, 2, 0).numpy())
            rel = c - (ds.context_size - 1)   # -3..0 relative to "now"
            tag = "now (t=0)" if rel == 0 else f"t={rel} ({rel/INPUT_FPS:+.2f}s)"
            ax.set_title(f"context {c}\nframe {curr_time - (ds.context_size - 1) + c}  |  {tag}",
                         fontsize=10)

    # row 1: the 5 scored target frames
    for c in range(ncols):
        ax = axes[1, c]
        ax.axis("off")
        if c < len(SECS):
            ts = TIMESTEPS[c]
            fidx = ts - 1                      # pred index for that horizon
            ax.imshow(pred_v[fidx].permute(1, 2, 0).numpy())
            ax.set_title(f"real frame @ {SECS[c]}s (GT)\nframe {curr_time + ts}  (gt idx {fidx})",
                         fontsize=10)

    fig.suptitle(
        f"RECON scene  |  traj '{f_curr}'  curr_time={curr_time}\n"
        f"TOP: 4 context frames (input)      BOTTOM: 5 future frames the model must predict (held-out GT)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    viz_path = os.path.join(out_dir, "context_and_targets.png")
    fig.savefig(viz_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {viz_path}")

    # ---- 5. machine-readable scene facts (so the webpage can't drift from data) ----
    pos = np.asarray(traj["position"])
    yaw = np.asarray(traj["yaw"])
    facts = {
        "trajectory": f_curr,
        "curr_time": int(curr_time),
        "n_frames": int(n_frames),
        "context_size": int(ds.context_size),
        "len_traj_pred": int(ds.len_traj_pred),
        "input_fps": INPUT_FPS,
        "context_frames": [int(curr_time - (ds.context_size - 1) + c) for c in range(ds.context_size)],
        "pred_first": int(curr_time + 1),
        "pred_last": int(curr_time + ds.len_traj_pred),
        "position_now": [round(float(pos[curr_time][0]), 4), round(float(pos[curr_time][1]), 4)],
        "yaw_now": round(float(yaw[curr_time]), 4),
        "delta1": [round(float(delta[1, 0]), 4), round(float(delta[1, 1]), 4), round(float(delta[1, 2]), 4)],
        "time_table": [
            {"sec": int(sec), "steps": int(ts), "rel_t": round(ts / 128.0, 3),
             "curr_delta": [round(float(delta[:ts, 0].sum()), 2),
                            round(float(delta[:ts, 1].sum()), 2),
                            round(float(delta[:ts, 2].sum()), 2)],
             "target_frame": int(curr_time + ts), "gt_idx": int(ts - 1)}
            for sec, ts in zip(SECS, TIMESTEPS)
        ],
    }
    with open(os.path.join(out_dir, "scene_facts.json"), "w") as f:
        json.dump(facts, f, indent=2)
    print(f"  wrote {out_dir}/scene_facts.json")
    print("\nDONE.")


if __name__ == "__main__":
    main()
