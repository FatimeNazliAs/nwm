"""Choosing which RECON sample the walkthrough looks at.

Every stage from T2 onwards needs the same two things: an EvalDataset built the
way the real eval builds it, and a way to say "look at THIS scene". Both live
here so that a stage never has to import from another stage.

Scene selection has two modes, driven by the stage's own config.yaml:
  * sample: N          -> row N of the predefined index data_splits/recon/test/time.pkl
  * trajectory: <name> -> load that folder directly at frame `time` (overrides sample)

Callers pass `cfg_name` purely so error messages can point at the right file.
"""
import os

# The REAL eval's dataset builder -- imported, never re-implemented. If the eval
# path changes how it constructs EvalDataset, every probe follows automatically
# and the walkthrough cannot quietly document behaviour the code no longer has.
from isolated_nwm_infer import get_dataset_eval


def build_recon_eval_dataset(config):
    """The RECON 'time' eval dataset, exactly as isolated_nwm_infer builds it.

    predefined_index=True makes it use data_splits/recon/test/time.pkl -- the
    same 500-row index the real evaluation iterates over.
    """
    return get_dataset_eval(config, "recon", "time", predefined_index=True)


def resolve_scene(ds, cfg, cfg_name="the stage config"):
    """
    Decide which RECON scene to load.

    Returns (row, f_curr, curr_time, scene_tag, mode) where `row` is the index
    to pass to EvalDataset.__getitem__ so the *real* dataset code still does all
    the image loading / action processing.

    Two modes:
      * trajectory mode (cfg['trajectory'] set): load that folder at cfg['time']
        directly, bypassing the 500-row time.pkl index. EvalDataset can index any
        folder listed in traj_names.txt (predefined_index=None builds exactly such
        an index); here we validate the folder + frame ourselves and splice a
        single synthetic row into ds.index_to_data, so __getitem__ is untouched.
      * sample mode (otherwise): use row cfg['sample'] of the time.pkl index.
    """
    traj = cfg.get("trajectory")
    if traj:  # ------------------------------------------------ trajectory mode
        t = cfg.get("time")
        if t is None:
            raise SystemExit(f"{cfg_name}: `trajectory` is set but `time` is missing.\n"
                             "  Add `time: <frame>` (the curr_time / 'now' frame index).")
        t = int(t)
        traj = str(traj)
        folder = os.path.join(ds.data_folder, traj)
        if not os.path.isfile(os.path.join(folder, "traj_data.pkl")):
            raise SystemExit(f"{cfg_name}: trajectory folder not found (no traj_data.pkl):\n"
                             f"    {folder}\n"
                             f"  Check the name against data_splits/recon/test/traj_names.txt.")
        traj_data = ds._get_trajectory(traj)
        n = len(traj_data["position"])
        need_before = ds.context_size - 1      # frames t-3..t must exist  -> t >= 3
        need_after = ds.len_traj_pred          # EvalDataset always builds the full 64-step horizon
        if t < need_before:
            raise SystemExit(f"{cfg_name}: time={t} too small for '{traj}'.\n"
                             f"  Need {ds.context_size} context frames ending at t "
                             f"(frames {t - need_before}..{t}); earliest valid time is {need_before}.")
        if t + need_after > n - 1:
            raise SystemExit(f"{cfg_name}: time={t} too large for '{traj}' ({n} frames).\n"
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
        raise SystemExit(f"{cfg_name}: sample={sample} out of range 0..{len(ds) - 1}.")
    f_curr, curr_time, _, _ = ds.index_to_data[sample]
    return sample, f_curr, int(curr_time), f"sample{sample}", "sample (row of time.pkl)"
