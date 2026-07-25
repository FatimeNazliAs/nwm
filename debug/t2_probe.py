"""
T2 data-layer probe for the NWM debug walkthrough.

Loads ONE real RECON eval sample through the *exact same* code path the
inference pipeline uses (EvalDataset + the time.pkl predefined index), then:
  1. prints the raw traj_data.pkl fields for that trajectory,
  2. prints every tensor the dataset returns, with shapes + dtypes,
  3. explains the action (delta) + how time is encoded (curr_delta / rel_t),
  4. renders an annotated visual: 4 context frames + the 5 target frames the
     "time" eval scores (sec = 1,2,4,8,16 -> frame 4,8,16,32,64).

Pick a scene by editing debug/t2_config.yaml (no rebuild), then run inside the
nwm_debug container:
    cd /app && python debug/t2_probe.py
Two ways to choose a scene (see debug/t2_config.yaml):
  * sample: N          -> row N of the predefined index data_splits/recon/test/time.pkl
  * trajectory: <name> -> load that folder directly at frame `time` (overrides sample)
Outputs to debug/t2_trace_out/<scene>/ (gitignored).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # repo root, so `import misc` / `datasets` resolve
os.chdir(ROOT)             # so config/, data_splits/, outputs resolve from repo root

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

import misc
from datasets import EvalDataset

# ---- eval "time" path constants (see isolated_nwm_infer.py) ----
INPUT_FPS = 4                       # recon is 4 fps
SECS = [1, 2, 4, 8, 16]             # 2**i for i in range(num_sec_eval=5)
TIMESTEPS = [s * INPUT_FPS for s in SECS]   # [4, 8, 16, 32, 64]


def build_recon_eval_dataset(config):
    """Mirror isolated_nwm_infer.get_dataset_eval for recon / time.pkl."""
    dc = config["eval_datasets"]["recon"]
    return EvalDataset(
        data_folder=dc["data_folder"],
        data_split_folder=dc["test"],
        dataset_name="recon",
        image_size=config["image_size"],
        min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
        max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
        len_traj_pred=config["eval_len_traj_pred"],
        traj_stride=config["traj_stride"],
        context_size=config["eval_context_size"],
        normalize=config["normalize"],
        transform=misc.transform,
        goals_per_obs=4,
        predefined_index="data_splits/recon/test/time.pkl",
        traj_names="traj_names.txt",
    )


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(name, t):
    if torch.is_tensor(t):
        print(f"  {name:<14} shape={tuple(t.shape)!s:<22} dtype={str(t.dtype):<15} "
              f"min={t.min().item():+.3f} max={t.max().item():+.3f}")
    else:
        print(f"  {name:<14} {t}")


def resolve_scene(ds, cfg):
    """
    Decide which RECON scene to load from debug/t2_config.yaml.

    Returns (row, f_curr, curr_time, scene_tag, mode) where `row` is the index
    to pass to EvalDataset.__getitem__ so the *real* dataset code still does all
    the image loading / action processing.

    Two modes:
      * trajectory mode (cfg['trajectory'] set): load that folder at cfg['time']
        directly, bypassing the 500-row time.pkl index. EvalDataset can index any
        folder listed in traj_names.txt (predefined_index=None builds exactly such
        an index); here we validate the folder + frame ourselves and splice a
        single synthetic row into ds.index_to_data, so __getitem__ is untouched.
      * sample mode (otherwise): use row cfg['sample'] of the time.pkl index
        (the original behavior).
    """
    traj = cfg.get("trajectory")
    if traj:  # ------------------------------------------------ trajectory mode
        t = cfg.get("time")
        if t is None:
            raise SystemExit("t2_config: `trajectory` is set but `time` is missing.\n"
                             "  Add `time: <frame>` (the curr_time / 'now' frame index).")
        t = int(t)
        traj = str(traj)
        folder = os.path.join(ds.data_folder, traj)
        if not os.path.isfile(os.path.join(folder, "traj_data.pkl")):
            raise SystemExit(f"t2_config: trajectory folder not found (no traj_data.pkl):\n"
                             f"    {folder}\n"
                             f"  Check the name against data_splits/recon/test/traj_names.txt.")
        traj_data = ds._get_trajectory(traj)
        n = len(traj_data["position"])
        need_before = ds.context_size - 1      # frames t-3..t must exist  -> t >= 3
        need_after = ds.len_traj_pred          # EvalDataset always builds the full 64-step horizon
        if t < need_before:
            raise SystemExit(f"t2_config: time={t} too small for '{traj}'.\n"
                             f"  Need {ds.context_size} context frames ending at t "
                             f"(frames {t - need_before}..{t}); earliest valid time is {need_before}.")
        if t + need_after > n - 1:
            raise SystemExit(f"t2_config: time={t} too large for '{traj}' ({n} frames).\n"
                             f"  Need {need_after} future frames after t (up to frame {t + need_after}); "
                             f"latest valid time is {n - 1 - need_after}.")
        in_split = traj in ds.traj_names
        note = "in test split" if in_split else "NOT in test traj_names.txt (loading from disk anyway)"
        # __getitem__ only reads (f_curr, curr_time) from the tuple; min/max are ignored.
        ds.index_to_data = [(traj, t, 0, 0)]
        return 0, traj, t, f"{traj}__t{t}", f"trajectory ({note})"

    # ---------------------------------------------------------------- sample mode
    sample = int(cfg.get("sample", 0) or 0)
    if not (0 <= sample < len(ds)):
        raise SystemExit(f"t2_config: sample={sample} out of range 0..{len(ds) - 1}.")
    f_curr, curr_time, _, _ = ds.index_to_data[sample]
    return sample, f_curr, int(curr_time), f"sample{sample}", "sample (row of time.pkl)"


def main():
    cfg = yaml.safe_load(open("debug/t2_config.yaml")) if os.path.exists("debug/t2_config.yaml") else {}
    cfg = cfg or {}

    with open("config/eval_config.yaml") as f:
        base = yaml.safe_load(f)
    with open("config/nwm_cdit_xl.yaml") as f:
        base.update(yaml.safe_load(f))

    ds = build_recon_eval_dataset(base)
    n_predef = len(ds)   # size of the predefined time.pkl index (before any override)

    row, f_curr, curr_time, scene_tag, mode = resolve_scene(ds, cfg)

    hr("SCENE SELECTED  (from debug/t2_config.yaml)")
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
    hr("4) VISUAL  ->  debug/t2_trace_out")
    out_dir = f"debug/t2_trace_out/{scene_tag}"
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
    print("\nDONE.")


if __name__ == "__main__":
    main()
