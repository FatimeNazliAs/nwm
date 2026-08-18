"""
T6 full-inference-and-metrics probe for the NWM debug walkthrough.

T5 was one prediction being made. This is the evaluation that makes 2500 of
them and puts a number on the result -- the two programs the Phase-4 run
actually executed:

    isolated_nwm_infer.py:215   elif args.eval_type == 'time'
    isolated_nwm_infer.py:108   generate_time  -- secs = [1, 2, 4, 8, 16]
    isolated_nwm_infer.py:66    model_forward_wrapper -- the prediction itself
    isolated_nwm_infer.py:118   visualize_preds -> id_<N>/<sec>.png
    isolated_nwm_eval.py:24     get_loss_fn -- lpips / dreamsim / fid
    isolated_nwm_eval.py:77     evaluate -- the scoring loop

The structural fact this stage is built around: **the two halves never share a
tensor.** Inference writes PNG files to disk and exits; scoring is a separate
program that reads those files back by path and scores them. The seam between
predicting and scoring is the filesystem, and the probe shows it as one.

It reports, with real numbers:
  1. one scene predicted at all five horizons, through the real call the eval
     makes, written into the real folder layout,
  2. the files that appeared, which is the entire interface to the scorer,
  3. what LPIPS, DreamSim and FID each measure, run on those five pairs,
  4. the question T5's page ended on: the prediction, the same prediction with
     the action removed, and the same action from different noise -- scored
     against the true frame three ways, to see which measures agree with what
     the pictures show,
  5. the finished 500-scene run scored by the eval's own `evaluate`,
  6. debug/out/t6/<scene>/eval_facts.json so the write-up cannot drift.

Nothing here regenerates the 500-scene run: it is already on disk from Phase 4
(five hours of GPU), and re-running it to publish a page would be absurd. The
predictions this probe makes are one scene's worth, to show the path end to end.

Pick a scene by editing debug/t6/config.yaml, then run inside the nwm_debug
container:
    cd /app && python debug/t6/probe.py
Outputs to debug/out/t6/<scene>/ (gitignored).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)   # repo root, so `import models` / `datasets` resolve
os.chdir(ROOT)             # so config/, data_splits/, outputs resolve from repo root

import time as clock
from typing import NamedTuple

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import distributed as dist
# The real inference path and the real scoring path, imported rather than
# rebuilt (ADR-0002). Everything measured below runs through these.
from isolated_nwm_infer import model_forward_wrapper, save_image
from isolated_nwm_eval import evaluate, get_loss_fn

from debug.common import facts as facts_io
from debug.common import images as images_io
from debug.common import model as nwm
from debug.common.report import hr
from debug.common.scene import build_recon_eval_dataset, resolve_scene

CONFIG = "debug/t6/config.yaml"
TEAL, AMBER, GREY = "#0f8f8b", "#b06d13", "#5c6a76"

# What the eval calls the two halves on disk. `exp_name` is the basename of the
# --exp config file (isolated_nwm_infer.py:139), which for this walkthrough is
# always config/nwm_cdit_xl.yaml; the held-out frames are written by a second
# run of the same script with --gt 1 (:137).
EXP_NAME = "nwm_cdit_xl"
DATASET = "recon"
EVAL_TYPE = "time"


class Settings(NamedTuple):
    """Everything debug/t6/config.yaml asks for, already checked.

    `raw` is kept because resolve_scene reads the scene keys itself and reports
    its own errors against the same file; everything else is resolved here.
    """
    raw: dict
    sec: int
    seed: int
    compare_seed: int
    predictions_dir: str
    ground_truth_dir: str
    score_batch_size: int


def read_config(path=CONFIG):
    """Read and check every knob BEFORE anything expensive is built.

    This stage's expensive part is very expensive -- a checkpoint, a VAE, seven
    250-pass predictions, three metric networks and 2500 scored pairs. A typo in
    a config file must not cost any of that, so every knob is resolved here,
    first line of main(), while nothing is loaded yet.

    Independent of the model, the scene and every tensor, so it is also the one
    part of the probe that can be tested without a GPU (ADR-0005 allows exactly
    this kind of helper).
    """
    cfg = yaml.safe_load(open(path)) or {}

    def whole(name, default):
        try:
            return int(cfg.get(name, default) if cfg.get(name) is not None else default)
        except (TypeError, ValueError):
            raise SystemExit(f"{path}: {name}={cfg.get(name)!r} is not a whole number.")

    sec = whole("sec", nwm.SECS_SWEPT[-1])
    if sec not in nwm.SECS_SWEPT:
        raise SystemExit(
            f"{path}: sec={sec} is not one of the horizons the evaluation uses.\n"
            f"  It sweeps {nwm.SECS_SWEPT} seconds and nothing else -- "
            f"generate_time() builds them as secs = [2**i for i in range(5)].")

    batch = whole("score_batch_size", 64)
    if batch < 1:
        raise SystemExit(f"{path}: score_batch_size={batch} must be at least 1.")

    def folder(name, default):
        got = str(cfg.get(name) or default)
        if not os.path.isdir(got):
            raise SystemExit(
                f"{path}: {name} is not a folder:\n    {got}\n"
                f"  This stage scores the finished 500-scene run; it does not "
                f"produce it. Check the path, or run Phase 4 first.")
        return got

    return Settings(
        raw=cfg,
        sec=sec,
        seed=whole("seed", 0),
        compare_seed=whole("compare_seed", 7),
        predictions_dir=folder("predictions_dir",
                               "/data/logs/nwm_eval/nwm_cdit_xl/recon/time"),
        ground_truth_dir=folder("ground_truth_dir", "/data/logs/nwm_eval/gt/recon/time"),
        score_batch_size=batch,
    )


class _EvalArgs(NamedTuple):
    """The one attribute isolated_nwm_eval.evaluate reads off `args` (:89).

    A named type rather than a SimpleNamespace so that a future upstream change
    reaching for another attribute fails by name here, instead of silently
    picking up whatever a loose object happened to carry.
    """
    batch_size: int


def check_run(gt_dir, exp_dir, secs):
    """Confirm the finished run is complete before spending a minute scoring it.

    `evaluate` (isolated_nwm_eval.py:87) trusts the two folders to match: it
    lists the ground-truth side, opens the same names on the prediction side,
    and a missing file surfaces as a PIL error most of the way through. This
    checks first, and says which side is short.

    Returns (scenes, pairs). It raises rather than reporting what is missing,
    because there is no sensible way to carry on: scoring an incomplete run
    would quietly average fewer pairs and publish the result as if it were the
    whole thing.
    """
    gt_eps = sorted(d for d in os.listdir(gt_dir)
                    if os.path.isdir(os.path.join(gt_dir, d)))
    exp_eps = set(d for d in os.listdir(exp_dir)
                  if os.path.isdir(os.path.join(exp_dir, d)))
    if not gt_eps:
        raise SystemExit(f"no scene folders in {gt_dir} -- expected id_0, id_1, ...")

    missing = []
    for ep in gt_eps:
        if ep not in exp_eps:
            missing.append(f"{ep} (no prediction folder)")
            continue
        for sec in secs:
            for side, root in (("prediction", exp_dir), ("truth", gt_dir)):
                if not os.path.isfile(os.path.join(root, ep, f"{sec}.png")):
                    missing.append(f"{ep}/{sec}.png ({side})")
    if missing:
        raise SystemExit(
            f"the finished run is incomplete: {len(missing)} file(s) missing, "
            f"first few:\n    " + "\n    ".join(missing[:5]) + "\n"
            f"  Scoring an incomplete run would quietly average fewer pairs.")
    return len(gt_eps), len(gt_eps) * len(secs)


def per_horizon(metric_logger, dataset_name, eval_name, secs):
    """The three scores per horizon, read out of the meters `evaluate` filled.

    `evaluate` does not return anything -- it writes into
    metric_logger.meters under names built from the dataset, the eval type, the
    metric and the horizon (isolated_nwm_eval.py:122-132). Rebuilding those
    names by hand at the call site is how a silently-empty row happens, so it is
    done once, here, and tested against a fake logger.
    """
    rows = []
    for sec in secs:
        row = {"sec": int(sec)}
        for metric in ("lpips", "dreamsim", "fid"):
            key = f"{dataset_name}_{eval_name}_{metric}_{sec}s"
            if key not in metric_logger.meters:
                raise SystemExit(
                    f"the scoring loop recorded no {metric} for {sec}s "
                    f"(meter {key!r} is missing).")
            row[metric] = float(metric_logger.meters[key].global_avg)
        rows.append(row)
    return rows


def pixel_mae(path_a, path_b):
    """Mean absolute difference between two saved frames, on the files themselves.

    Deliberately reads the PNGs rather than the tensors they came from: LPIPS and
    DreamSim are handed the same paths, so all three numbers describe the same
    bytes and none of them can be flattered by a different rounding.
    """
    a, b = plt.imread(path_a)[..., :3], plt.imread(path_b)[..., :3]
    return float(np.abs(a - b).mean())


def main():
    # Every knob is checked here, before the dataset, the checkpoint or the GPU.
    cfg = read_config()
    base = nwm.load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    secs = np.array(nwm.SECS_SWEPT)          # generate_time(): [2**i for i in range(5)]

    # ---- 0. the scene, and what the evaluation asks of it ----
    ds = build_recon_eval_dataset(base)
    row, f_curr, curr_time, scene_tag, mode = resolve_scene(ds, cfg.raw, CONFIG)
    _, obs_image, gt_image, delta = ds[row]

    T = ds.context_size
    latent_size = base["image_size"] // 8
    sec = cfg.sec
    ts = sec * nwm.INPUT_FPS
    out_dir = f"debug/out/t6/{scene_tag}"
    save = images_io.Saver(out_dir)     # writes the PNGs and records the manifest

    hr("0) WHAT THE EVALUATION ASKS  (debug/t6/config.yaml)")
    print(f"  mode          : {mode}")
    print(f"  trajectory    : {f_curr}")
    print(f"  curr_time     : frame {curr_time}   ('now', the last context frame)")
    print(f"  horizons      : {list(nwm.SECS_SWEPT)} seconds -- FIVE predictions from")
    print(f"                  the same {T} frames, asking for five different futures")
    print(f"  highlighted   : {sec} s -> ts = {sec} x {nwm.INPUT_FPS} fps = {ts} dataset steps,")
    print(f"                  target frame {curr_time + ts}, held out")
    print(f"  device        : {device}")
    print(f"\n  The real run does this for all 500 rows of data_splits/{DATASET}/test/{EVAL_TYPE}.pkl.")
    print(f"  This probe does it for one, and then scores all 500 from the finished run.")

    # ---- 1. what one prediction is built from ----
    hr("1) WHAT ONE PREDICTION IS BUILT FROM  (all of T3, T4 and T5)")
    model, ckpt = nwm.build_cdit(base, context_size=T, latent_size=latent_size, device=device)
    vae = nwm.build_vae(device)
    diffusion = nwm.build_diffusion()
    print(f"  model      : {base['model']}, {ckpt['params_total']/1e6:.0f} M params, "
          f"checkpoint {nwm.CKP} key '{ckpt['key']}' (step {ckpt['train_step']})")
    print(f"  vae        : {nwm.VAE_NAME}, frozen")
    print(f"  diffusion  : create_diffusion(str({nwm.RESPACED_STEPS})), "
          f"num_timesteps = {diffusion.num_timesteps}")
    print(f"  One prediction = one {nwm.RESPACED_STEPS}-pass loop (T5). Five horizons means five")
    print(f"  separate loops from five separate screens of noise -- they share the context")
    print(f"  frames and nothing else. Nothing is reused between them, and no prediction is")
    print(f"  fed back into the next: this is the 'time' eval, not 'rollout' (infer.py:210).")
    print(f"\n  Unlike T5's probe this runs the real wrapper, so it is bf16 autocast")
    print(f"  (infer.py:71) -- the arithmetic the published numbers were made with.")

    # ---- 2. the five predictions, through the real call ----
    hr("2) PREDICTING  (model_forward_wrapper, infer.py:66)")
    # The real run's folder layout, rebuilt exactly: <exp>/<dataset>/time/id_<N>/<sec>.png
    # for the predictions and gt/<dataset>/time/id_<N>/<sec>.png for the held-out
    # frames, from a second run of the same script with --gt 1 (infer.py:136-140).
    run_root = os.path.join(out_dir, "run")
    pred_dir = os.path.join(run_root, EXP_NAME, DATASET, EVAL_TYPE, f"id_{row}")
    truth_dir = os.path.join(run_root, "gt", DATASET, EVAL_TYPE, f"id_{row}")
    cmp_dir = os.path.join(run_root, "compare", f"id_{row}")
    for d in (pred_dir, truth_dir, cmp_dir):
        os.makedirs(d, exist_ok=True)

    obs = obs_image[-T:].unsqueeze(0).to(device)      # (1, T, 3, 224, 224), infer.py:207
    gt_b = gt_image.unsqueeze(0)                      # (1, 64, 3, 224, 224)
    delta_b = delta.unsqueeze(0)                      # (1, 64, 3)

    print(f"  generate_time (infer.py:108) is a five-line loop over the horizons. For each:")
    print(f"    curr_delta = delta[:, :ts].sum(dim=1, keepdim=True)   where the robot ends up")
    print(f"    model_forward_wrapper(..., num_timesteps=ts)          the {nwm.RESPACED_STEPS}-pass loop")
    print(f"    visualize_preds(...)                                  -> id_<N>/<sec>.png")
    print(f"  The probe calls the middle two directly. visualize_preds cannot run here: it")
    print(f"  does `for i, s in enumerate(idxs.squeeze())` (:119), and a batch of one squeezes")
    print(f"  to a 0-d tensor that will not iterate. The real run uses batches of 16.")
    print(f"  The seeding is the probe's own -- the real run does not seed -- so that the")
    print(f"  comparison in section 5 starts from exactly the same noise as the prediction.")
    print()

    sweep, t0 = [], clock.time()
    for s in nwm.SECS_SWEPT:
        ts_i = int(s * nwm.INPUT_FPS)
        curr_delta = delta_b[:, :ts_i].sum(dim=1, keepdim=True)     # infer.py:111
        torch.manual_seed(cfg.seed)
        t1 = clock.time()
        px = model_forward_wrapper((model, diffusion, vae), obs, curr_delta, ts_i,
                                   latent_size, device=device, num_cond=T)
        took = clock.time() - t1
        p_path = os.path.join(pred_dir, f"{s}.png")
        t_path = os.path.join(truth_dir, f"{s}.png")
        save_image(p_path, px[0], True)                             # infer.py:124
        save_image(t_path, gt_b[0, ts_i - 1], True)                 # the --gt 1 branch, :113
        act = curr_delta[0, 0]
        sweep.append({"sec": int(s), "ts": ts_i,
                      "target_frame": int(curr_time + ts_i),
                      "action": [round(float(v), 4) for v in act],
                      "seconds": round(took, 1),
                      "pred_png": p_path, "truth_png": t_path,
                      "pred_bytes": os.path.getsize(p_path)})
        print(f"  {s:>2} s ahead -> frame {curr_time + ts_i:>4}  "
              f"action ({act[0]:+.3f}, {act[1]:+.3f}, {act[2]:+.3f})  "
              f"{took:>5.1f} s  -> {os.path.basename(p_path)}")
    predict_secs = clock.time() - t0
    print(f"\n  five predictions in {predict_secs:.0f} s, one scene.")

    # ---- 3. the seam ----
    hr("3) THE SEAM: EVERYTHING BETWEEN THE TWO HALVES IS FILES")
    print(f"  Inference has now finished for this scene and is holding nothing. What it")
    print(f"  leaves behind is this, and only this:\n")
    for d, what in ((pred_dir, "predicted"), (truth_dir, "held out, true")):
        print(f"    {d}/")
        for fn in sorted(os.listdir(d), key=lambda n: int(n.split('.')[0])):
            print(f"      {fn:<8} {os.path.getsize(os.path.join(d, fn)):>7,} bytes   {what}")
    print(f"\n  isolated_nwm_eval.py is a separate program run afterwards. It never imports")
    print(f"  the model, never loads the checkpoint and never sees a latent. It lists the")
    print(f"  ground-truth folders (:87), opens the matching prediction by path (:108-111)")
    print(f"  and scores the two images. The names ARE the interface: id_<row> says which")
    print(f"  scene, <sec>.png says which horizon.")
    print(f"  That is why the Phase-4 run could take five hours on one day and be scored in")
    print(f"  seventy seconds on another, and why the same scorer works on any model's output.")

    # ---- 4. the three measures ----
    hr("4) THE THREE MEASURES  (get_loss_fn, eval.py:24)")
    print(f"  Building the three, exactly as the eval builds them ...")
    lpips_fn = get_loss_fn("lpips", secs, device)          # eval.py:26, AlexNet features
    dreamsim_fn = get_loss_fn("dreamsim", secs, device)    # eval.py:46

    def both(a, b):
        """The two per-pair measures on one pair of files, as plain floats.

        Upstream's loss functions score a whole BATCH: they take two lists of
        paths and return the mean (eval.py:27, :47). Everything in this probe
        scores one pair at a time, so every call site was wrapping single files
        in single-element lists and unwrapping a tensor afterwards -- eight
        times, with the two file paths buried between the brackets. This is that
        adaptation, written once. A closure rather than a module-level helper
        because it is bound to the two loss functions just built, exactly like
        the hook closures in T4's probe (ADR-0005).
        """
        return {"lpips": float(lpips_fn([a], [b])),
                "dreamsim": float(dreamsim_fn([a], [b]))}
    print(f"\n  LPIPS     lpips.LPIPS(net='alex'). Runs both frames through AlexNet and")
    print(f"            compares the feature maps layer by layer. Lower is better; it")
    print(f"            forgives a leaf in the wrong place and punishes a wall in the")
    print(f"            wrong place. Takes FILE PATHS, one pair at a time.")
    print(f"  DreamSim  dreamsim(pretrained=True). An ensemble tuned on human judgements")
    print(f"            of 'which of these two images is more like that one'. Lower is")
    print(f"            better, and it weighs layout and objects over texture.")
    print(f"  FID       FrechetInceptionDistance(feature_dim=2048). NOT a per-pair score:")
    print(f"            it fits a Gaussian to the Inception features of ALL the predictions")
    print(f"            and another to ALL the true frames, and measures the distance")
    print(f"            between those two clouds. It cannot say whether prediction 7")
    print(f"            matches truth 7 -- only whether the predictions look like the")
    print(f"            right kind of picture as a set. That is why it is computed once")
    print(f"            per horizon over 500 scenes and never on a single pair.")
    print(f"\n  This scene, scored pair by pair:\n")
    print(f"    {'horizon':>8} {'frame':>7} {'LPIPS':>9} {'DreamSim':>10}")
    for r in sweep:
        r.update(both(r["truth_png"], r["pred_png"]))
        print(f"    {r['sec']:>6} s {r['target_frame']:>7} {r['lpips']:>9.4f} "
              f"{r['dreamsim']:>10.4f}")
    print(f"\n  One scene is one scene -- these five numbers are not the result. The result")
    print(f"  is section 6, the same two measures averaged over 500 of them.")

    # ---- 5. the question T5 ended on ----
    hr("5) THE QUESTION T5 ENDED ON: WHICH MEASURE AGREES WITH YOUR EYES?")
    print(f"  T5 compared three finished predictions with a plain distance, twice: once")
    print(f"  between latents, once between pixels. The two rankings came out in opposite")
    print(f"  orders, so neither could be trusted to say which prediction was better.")
    print(f"  Same three pictures here, scored against the TRUE frame the way the")
    print(f"  evaluation does it.\n")
    cmp_delta = delta_b[:, :ts].sum(dim=1, keepdim=True)

    torch.manual_seed(cfg.seed)                    # identical noise to the sweep's `sec` run
    px_noact = model_forward_wrapper((model, diffusion, vae), obs,
                                     torch.zeros_like(cmp_delta), ts,
                                     latent_size, device=device, num_cond=T)
    torch.manual_seed(cfg.compare_seed)            # same action, different starting noise
    px_seed = model_forward_wrapper((model, diffusion, vae), obs, cmp_delta, ts,
                                    latent_size, device=device, num_cond=T)
    noact_png = os.path.join(cmp_dir, f"noaction_{sec}.png")
    seed_png = os.path.join(cmp_dir, f"otherseed_{sec}.png")
    # The starting view, written through the same save_image so that every number
    # below is measured on files produced identically.
    start_png = os.path.join(cmp_dir, "start.png")
    save_image(noact_png, px_noact[0], True)
    save_image(seed_png, px_seed[0], True)
    save_image(start_png, obs_image[-1], True)

    at_sec = next(r for r in sweep if r["sec"] == sec)
    truth_png, pred_png = at_sec["truth_png"], at_sec["pred_png"]
    compare = []
    for label, path in (("the prediction", pred_png),
                        ("action removed", noact_png),
                        ("different starting noise", seed_png)):
        # ...and against where the robot STARTED, so "did it move the camera?"
        # is a measurement rather than an impression of the pictures.
        from_start = both(start_png, path)
        compare.append({
            "label": label, "png": path,
            "pixel_mae": pixel_mae(truth_png, path),
            **both(truth_png, path),
            "lpips_vs_start": from_start["lpips"],
            "dreamsim_vs_start": from_start["dreamsim"],
        })
    print(f"  {'compared with the TRUE frame':<28} {'pixel diff':>11} {'LPIPS':>9} {'DreamSim':>10}")
    for c in compare:
        print(f"  {c['label']:<28} {c['pixel_mae']:>11.4f} {c['lpips']:>9.4f} "
              f"{c['dreamsim']:>10.4f}")

    # Did the camera actually travel? The truth is a long way from the starting
    # view; a prediction that stayed put is not.
    truth_from_start = both(start_png, truth_png)
    print(f"\n  ...and how far each one travelled from the STARTING view "
          f"(frame {curr_time}):")
    print(f"  {'':<28} {'':>11} {'LPIPS':>9} {'DreamSim':>10}")
    print(f"  {'what really happened':<28} {'':>11} {truth_from_start['lpips']:>9.4f} "
          f"{truth_from_start['dreamsim']:>10.4f}")
    for c in compare:
        print(f"  {c['label']:<28} {'':>11} {c['lpips_vs_start']:>9.4f} "
              f"{c['dreamsim_vs_start']:>10.4f}")
    stayed = [c["label"] for c in compare
              if c["dreamsim_vs_start"] < truth_from_start["dreamsim"]]
    print(f"  Read this before explaining the table above it. "
          f"{'None of the three' if not stayed else 'Only ' + ', '.join(stayed)} "
          f"{'is' if len(stayed) < 2 else 'are'} closer to the starting view than the true")
    print(f"  frame is, so {'no run here is' if not stayed else 'not every run is'} "
          f"a repeat of where it began. What separates these three is WHAT they")
    print(f"  contain, not how much they changed -- which is exactly what a per-pixel")
    print(f"  average cannot see.")

    # Which measure ranks the three the way the pictures do? Read off the numbers
    # rather than asserted: the ranking is a property of this scene's run.
    ranks = {m: [c["label"] for c in sorted(compare, key=lambda c: c[m])]
             for m in ("pixel_mae", "lpips", "dreamsim")}
    print(f"\n  best-first, by each measure:")
    for m in ("pixel_mae", "lpips", "dreamsim"):
        print(f"    {m:<10} {'  <  '.join(ranks[m])}")
    agree = ranks["lpips"] == ranks["dreamsim"]
    print(f"\n  LPIPS and DreamSim {'agree' if agree else 'DISAGREE'} with each other here.")
    print(f"  The plain pixel difference {'agrees too' if ranks['pixel_mae'] == ranks['lpips'] else 'puts them in a different order'}.")

    # ...and the pair T5 measured directly: two equally valid futures of one drive.
    seed_vs_pred = {"pixel_mae": pixel_mae(pred_png, seed_png),
                    **both(pred_png, seed_png)}
    print(f"\n  and the two runs that differ only by their starting noise, against each other:")
    print(f"    pixel diff {seed_vs_pred['pixel_mae']:.4f}   LPIPS {seed_vs_pred['lpips']:.4f}   "
          f"DreamSim {seed_vs_pred['dreamsim']:.4f}")

    # ---- 6. the whole run ----
    hr("6) THE WHOLE RUN, SCORED  (evaluate, eval.py:77)")
    scenes, pairs = check_run(cfg.ground_truth_dir, cfg.predictions_dir, nwm.SECS_SWEPT)
    print(f"  predictions : {cfg.predictions_dir}")
    print(f"  truth       : {cfg.ground_truth_dir}")
    print(f"  {scenes} scene folders x {len(nwm.SECS_SWEPT)} horizons = {pairs} pairs, "
          f"{pairs * 2} files.")
    print(f"  Produced by the Phase-4 inference run; this probe only reads them.")
    # One real folder out of the 500, measured rather than described: the page
    # shows this listing, and it must be the finished run's own files, not the
    # probe's one-scene copy of the same layout.
    example_folder = sorted((d for d in os.listdir(cfg.predictions_dir)
                             if os.path.isdir(os.path.join(cfg.predictions_dir, d))),
                            key=lambda n: int(n.split("_")[-1]))[0]
    example_files = [
        {"sec": int(s),
         "bytes": os.path.getsize(os.path.join(cfg.predictions_dir, example_folder, f"{s}.png"))}
        for s in nwm.SECS_SWEPT]
    print(f"\n  one of those {scenes} folders, {example_folder}/:")
    for e in example_files:
        print(f"    {str(e['sec']) + '.png':<8} {e['bytes']:>7,} bytes   "
              f"predicted {e['sec']} s ahead")
    print(f"\n  Building five FID accumulators (one per horizon) and scoring ...")
    fid_fns = get_loss_fn("fid", secs, device)             # eval.py:65-70
    mlog = dist.MetricLogger(delimiter="  ")
    args = _EvalArgs(batch_size=cfg.score_batch_size)
    t0 = clock.time()
    with torch.no_grad():
        evaluate(args, DATASET, EVAL_TYPE, mlog, (lpips_fn, dreamsim_fn, fid_fns),
                 cfg.ground_truth_dir, cfg.predictions_dir, secs, None)
    score_secs = clock.time() - t0
    rows = per_horizon(mlog, DATASET, EVAL_TYPE, nwm.SECS_SWEPT)
    n_batches = (scenes + cfg.score_batch_size - 1) // cfg.score_batch_size
    print(f"\n  scored {pairs} pairs in {score_secs:.0f} s "
          f"({n_batches} batches of {cfg.score_batch_size}).\n")
    print(f"    {'horizon':>8} {'LPIPS':>9} {'DreamSim':>10} {'FID':>9}")
    for r in rows:
        print(f"    {r['sec']:>6} s {r['lpips']:>9.4f} {r['dreamsim']:>10.4f} {r['fid']:>9.2f}")
    worst, best = rows[-1], rows[0]
    print(f"\n  All three get worse as the horizon grows: over {best['sec']} s to "
          f"{worst['sec']} s, LPIPS")
    print(f"  rises {(worst['lpips']/best['lpips'] - 1)*100:.0f}%, DreamSim "
          f"{(worst['dreamsim']/best['dreamsim'] - 1)*100:.0f}% and FID "
          f"{(worst['fid']/best['fid'] - 1)*100:.0f}%.")
    print(f"  DreamSim moves the most, and that is the informative one: it is the measure")
    print(f"  that cares about layout, and layout is exactly what a longer drive gets wrong.")
    print(f"  A note on the average: evaluate() updates each meter once per BATCH with")
    print(f"  n=1 (:122), so the published figure is the mean of {n_batches} batch means, and")
    print(f"  the last batch holds {scenes - (n_batches - 1) * cfg.score_batch_size} scenes rather than "
          f"{cfg.score_batch_size}. Upstream's arithmetic, kept as it is.")

    # ---- 7. visuals ----
    hr("7) VISUALS  ->  " + out_dir)
    # The page shows the very files that were scored, read back rather than
    # re-rendered from the tensors: a picture and its number must be the same bytes.
    for i in range(T):
        save.image(f"ctx_f{i}", nwm.to_display(obs_image[-T:][i]))
    for r in sweep:
        save.image(f"pred_{r['sec']}s", plt.imread(r["pred_png"]))
        save.image(f"truth_{r['sec']}s", plt.imread(r["truth_png"]))
    save.image("cmp_noaction", plt.imread(noact_png))
    save.image("cmp_otherseed", plt.imread(seed_png))
    save.image("cmp_start", plt.imread(start_png))

    # --- figure: every score against the horizon, over all 500 scenes ---
    xs = [r["sec"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.0))
    ax1.plot(xs, [r["lpips"] for r in rows], "o-", color=TEAL, lw=2, label="LPIPS")
    ax1.plot(xs, [r["dreamsim"] for r in rows], "o-", color=AMBER, lw=2, label="DreamSim")
    ax1.set_ylabel("distance (lower is better)")
    ax1.legend(fontsize=8, frameon=False)
    ax2.plot(xs, [r["fid"] for r in rows], "o-", color=GREY, lw=2)
    ax2.set_ylabel("FID (lower is better)")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}s" for x in xs])
        ax.set_xlabel(f"how far ahead the prediction reaches   ({scenes} scenes)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save.figure("curve", fig, dpi=130)
    print(f"  wrote curve.png, pred_*.png, truth_*.png, cmp_*.png, ctx_f*.png")

    # ---- 8. machine-readable facts ----
    facts = {
        "trajectory": f_curr,
        "curr_time": int(curr_time),
        "row": int(row),
        "scene_mode": mode,
        "context_size": int(T),
        "context_frame_numbers": [int(curr_time - (T - 1) + i) for i in range(T)],
        "sec": sec,
        "ts": int(ts),
        "input_fps": nwm.INPUT_FPS,
        "target_frame_number": int(curr_time + ts),
        "secs_swept": [int(s) for s in nwm.SECS_SWEPT],
        "image_size": int(base["image_size"]),
        "latent_size": int(latent_size),
        "in_channels": 4,
        "model": base["model"],
        "checkpoint": nwm.CKP,
        "train_step": ckpt["train_step"] if isinstance(ckpt["train_step"], int) else str(ckpt["train_step"]),
        "params_total_m": round(ckpt["params_total"] / 1e6, 1),
        "vae": nwm.VAE_NAME,
        "respaced_steps": nwm.RESPACED_STEPS,
        "eval_type": EVAL_TYPE,
        "dataset": DATASET,
        "seed": cfg.seed,
        "compare_seed": cfg.compare_seed,
        "autocast": "bfloat16 (infer.py:71)",
        # one scene, five horizons
        "sweep": [{k: v for k, v in r.items() if k not in ("pred_png", "truth_png")}
                  for r in sweep],
        "predict_seconds": round(predict_secs, 1),
        # the seam
        "pred_dir": pred_dir,
        "truth_dir": truth_dir,
        "file_pattern": "id_<row>/<sec>.png",
        "files_this_scene": len(sweep) * 2,
        "scorer_imports_model": False,
        # the three measures
        "lpips_net": "alex",
        "fid_feature_dim": 2048,
        "fid_is_per_pair": False,
        # the T5 question
        "compare": [{k: v for k, v in c.items() if k != "png"} for c in compare],
        "compare_ranks": ranks,
        "compare_lpips_dreamsim_agree": bool(agree),
        "truth_from_start": {k: round(v, 4) for k, v in truth_from_start.items()},
        "other_seed_vs_prediction": {k: round(v, 4) for k, v in seed_vs_pred.items()},
        # the whole run
        "run_scenes": int(scenes),
        "run_pairs": int(pairs),
        "run_files": int(pairs * 2),
        "run_predictions_dir": cfg.predictions_dir,
        "run_ground_truth_dir": cfg.ground_truth_dir,
        "run_example_folder": example_folder,
        "run_example_files": example_files,
        "score_seconds": round(score_secs, 1),
        "score_batch_size": cfg.score_batch_size,
        "score_batches": int(n_batches),
        "score_last_batch": int(scenes - (n_batches - 1) * cfg.score_batch_size),
        "per_horizon": [{"sec": r["sec"], "lpips": round(r["lpips"], 4),
                         "dreamsim": round(r["dreamsim"], 4), "fid": round(r["fid"], 2)}
                        for r in rows],
        "growth_lpips_pct": round((worst["lpips"] / best["lpips"] - 1) * 100, 1),
        "growth_dreamsim_pct": round((worst["dreamsim"] / best["dreamsim"] - 1) * 100, 1),
        "growth_fid_pct": round((worst["fid"] / best["fid"] - 1) * 100, 1),
    }
    # Written through debug/common/facts.py, never json.dump directly: that is
    # what stamps the schema version the page checks on load (ADR-0004).
    written = facts_io.write(out_dir, "eval_facts.json", facts, stage="t6",
                             images=save.names)
    print(f"\n  wrote {len(save)} pictures, recorded in the facts file's manifest")
    print(f"  wrote {written}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
