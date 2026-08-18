"""
T5 diffusion-loop probe for the NWM debug walkthrough.

T4 explained the model that answers one question. This is the loop that asks it
250 times and turns noise into a frame -- the call the real evaluation makes for
every single prediction it publishes:

    isolated_nwm_infer.py:81    z = torch.randn(B, 4, 28, 28)
    isolated_nwm_infer.py:84    diffusion.p_sample_loop(model.forward, z.shape, z,
                                    clip_denoised=False, model_kwargs=..., device=...)
    isolated_nwm_infer.py:168   create_diffusion(str(250))
    diffusion/gaussian_diffusion.py:470   p_sample_loop_progressive -- the loop itself
    diffusion/gaussian_diffusion.py:382   p_sample -- one step
    diffusion/respace.py:18               how 1000 training steps become 250 inference steps

Nothing here is a stand-in. T4's filmstrip was built with q_sample -- the FORWARD
process applied to the true future frame -- and was labelled as borrowed. This
probe runs the real reverse loop and decodes what actually came out of it.

It reports, with real numbers:
  1. the schedule: which 250 of the 1000 steps survive, and how much signal each
     one carries,
  2. the loop, run for real, keeping all 251 latents it passes through,
  3. ONE step opened up: the model's guess at the finished frame, the blend that
     makes the next latent, and the fresh noise put back in,
  4. when the frame is actually decided -- the running guess stops moving long
     before the loop ends,
  5. why clip_denoised=False: how much of the guess lives outside [-1, 1],
  6. two more full loops -- same noise without the action, and the same action
     from different noise -- to separate what the noise decides from what the
     action decides,
  7. debug/out/t5/<scene>/loop_facts.json so the write-up cannot drift.

Pick a scene by editing debug/t5/config.yaml, then run inside the nwm_debug
container:
    cd /app && python debug/t5/probe.py
Outputs to debug/out/t5/<scene>/ (gitignored).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)   # repo root, so `import models` / `datasets` resolve
os.chdir(ROOT)             # so config/, data_splits/, outputs resolve from repo root

from typing import NamedTuple

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from debug.common import facts as facts_io
from debug.common import images as images_io
from debug.common import model as nwm
from debug.common.report import hr, show
from debug.common.scene import build_recon_eval_dataset, resolve_scene

CONFIG = "debug/t5/config.yaml"
TEAL, AMBER, GREY = "#0f8f8b", "#b06d13", "#5c6a76"


class Settings(NamedTuple):
    """Everything debug/t5/config.yaml asks for, already checked.

    `raw` is kept because resolve_scene reads the scene keys itself and reports
    its own errors against the same file; everything else is resolved here.
    """
    raw: dict
    sec: int
    step: int
    filmstrip_steps: tuple
    seed: int
    compare_seed: int


def read_config(path=CONFIG):
    """Read and check every knob BEFORE anything expensive is built.

    The old arrangement validated `step` after the first 250-pass loop and
    `filmstrip_steps` after all three, so a typo cost a checkpoint load and about
    a minute of GPU before it was reported. A config file is this stage's
    interface for the person using it; it should fail in under a second.

    Independent of the model, the scene and every tensor, so it is also the one
    part of the probe that can be tested without a GPU -- which is why it is out
    here rather than inside main() (ADR-0005 allows exactly this kind of helper).
    """
    cfg = yaml.safe_load(open(path)) or {}
    last = nwm.RESPACED_STEPS - 1

    def whole(name, default):
        try:
            return int(cfg.get(name, default) if cfg.get(name) is not None else default)
        except (TypeError, ValueError):
            raise SystemExit(f"{path}: {name}={cfg.get(name)!r} is not a whole number.")

    sec = whole("sec", 1)
    if not (1 <= sec <= nwm.SECS_SWEPT[-1]):
        raise SystemExit(
            f"{path}: sec={sec} is outside 1..{nwm.SECS_SWEPT[-1]}.\n"
            f"  The dataset only builds {nwm.SECS_SWEPT[-1]} seconds of future frames.")

    step = whole("step", 125)
    if not (0 <= step <= last):
        raise SystemExit(
            f"{path}: step={step} is outside 0..{last}.\n"
            f"  {last} is the loop's first pass, 0 its last.")

    want = cfg.get("filmstrip_steps") or [249, 200, 150, 100, 60, 30, 10, 0]
    try:
        strip = [int(v) for v in want]
    except (TypeError, ValueError):
        raise SystemExit(f"{path}: filmstrip_steps={want!r} is not a list of whole numbers.")
    bad = [v for v in strip if not (0 <= v <= last)]
    if bad:
        raise SystemExit(
            f"{path}: filmstrip_steps has {bad}, outside 0..{last}.")
    if len(set(strip)) < 3:
        raise SystemExit(
            f"{path}: filmstrip_steps needs at least 3 different passes, got {strip}.\n"
            f"  The page reads the strip as a story from first pass to last.")
    # Sorted high to low because that is loop order, and the page's captions say
    # "left to right". Silently accepting another order would make them false.
    strip = tuple(sorted(set(strip), reverse=True))

    return Settings(raw=cfg, sec=sec, step=step, filmstrip_steps=strip,
                    seed=whole("seed", 0), compare_seed=whole("compare_seed", 7))


class Pass(NamedTuple):
    """One turn of the reverse loop: the model called once, x nudged once.

    `step` is the loop's own index, 249 down to 0. `t` is the noise level the
    model is told about, 999 down to 0 -- a different number, on a different
    scale, which upstream also calls a timestep. Keeping both here is what stops
    the two being confused at the call sites.
    """
    step: int
    t: int
    x_in: object        # what went in
    x_out: object       # what came out, i.e. the next pass's x_in
    guess: object       # pred_xstart: this pass's guess at the FINISHED latent


class Run:
    """One complete execution of the loop: every Pass, and what it ended on.

    Exists because the honest raw result is two lists of different lengths -- 251
    latents and 250 guesses -- whose alignment could only be stated in prose.
    Callers were doing `steps.index(s)` and `xs[k], xs[k+1], guesses[k]` in eight
    places, and the off-by-one was available to get wrong at every one of them.
    Addressing a pass by its step number makes that impossible.
    """

    def __init__(self, passes, final):
        self.passes = tuple(passes)
        self.final = final
        self._by_step = {p.step: p for p in self.passes}

    @property
    def steps(self):
        """Step numbers in the order the loop visits them: 249 down to 0."""
        return tuple(p.step for p in self.passes)

    def at(self, step):
        """The pass numbered `step`. Raises with the valid range, not KeyError."""
        try:
            return self._by_step[int(step)]
        except KeyError:
            raise SystemExit(
                f"no pass numbered {step} in this run; it has "
                f"{self.steps[0]} down to {self.steps[-1]}.") from None

    def __iter__(self):
        return iter(self.passes)

    def __len__(self):
        return len(self.passes)


def sample_loop(diffusion, model, z, model_kwargs, device):
    """Run the real reverse loop and keep every latent it passes through.

    p_sample_loop (infer.py:84) is precisely this generator drained to its last
    value -- see gaussian_diffusion.py:425-468, whose whole body is a for-loop
    that throws the intermediates away. Iterating it instead is the only way to
    see the 250 states in between, and changes nothing about what is computed.

    Everything is moved to CPU float32 as it arrives -- 251 latents is 3 MB, and
    keeping them is what lets the measurements below be arithmetic on the real
    run rather than on a second, differently-seeded one.
    """
    t_map = diffusion.timestep_map
    n = diffusion.num_timesteps
    passes, x_in = [], z.detach().float().cpu()
    for i, out in enumerate(diffusion.p_sample_loop_progressive(
            model.forward, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, device=device)):
        x_out = out["sample"].detach().float().cpu()
        step = n - 1 - i                       # the loop counts down: 249 .. 0
        passes.append(Pass(step=step, t=int(t_map[step]), x_in=x_in, x_out=x_out,
                           guess=out["pred_xstart"].detach().float().cpu()))
        x_in = x_out                           # this pass's output is the next one's input
    return Run(passes, final=x_in)


def rel(a, b, scale):
    """How far apart two latents are, as a percentage of `scale`."""
    return float((a - b).norm()) / scale * 100


def main():
    # Every knob is checked here, before the dataset, the checkpoint or the GPU.
    cfg = read_config()
    seed, compare_seed = cfg.seed, cfg.compare_seed
    base = nwm.load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 0. the scene, and what exactly we are asking for ----
    ds = build_recon_eval_dataset(base)
    row, f_curr, curr_time, scene_tag, mode = resolve_scene(ds, cfg.raw, CONFIG)
    _, obs_image, gt_image, delta = ds[row]

    sec = cfg.sec
    ts = sec * nwm.INPUT_FPS                  # generate_time(): eval_timesteps = sec*input_fps
    T = ds.context_size
    latent_size = base["image_size"] // 8
    IMG = base["image_size"]
    out_dir = f"debug/out/t5/{scene_tag}"
    save = images_io.Saver(out_dir)     # writes the PNGs and records the manifest

    hr("0) THE QUESTION WE ARE ASKING  (debug/t5/config.yaml)")
    print(f"  mode          : {mode}")
    print(f"  trajectory    : {f_curr}")
    print(f"  curr_time     : frame {curr_time}   ('now', the last context frame)")
    print(f"  horizon       : {sec} s ahead -> ts = {sec} x {nwm.INPUT_FPS} fps = {ts} dataset steps")
    print(f"  target        : frame {curr_time + ts}, held out")
    print(f"  device        : {device}")

    # ---- 1. the three things the loop needs ----
    hr("1) WHAT THE LOOP IS BUILT FROM")
    model, ckpt = nwm.build_cdit(base, context_size=T, latent_size=latent_size, device=device)
    print(f"  model      : {base['model']}, {ckpt['params_total']/1e6:.0f} M params, "
          f"checkpoint {nwm.CKP} key '{ckpt['key']}' (step {ckpt['train_step']}) -- all of T4")
    vae = nwm.build_vae(device)
    print(f"  vae        : {nwm.VAE_NAME}, frozen -- all of T3")
    diffusion = nwm.build_diffusion()
    print(f"  diffusion  : create_diffusion(str({nwm.RESPACED_STEPS})) -> "
          f"{type(diffusion).__name__}, num_timesteps = {diffusion.num_timesteps}")
    print(f"  The model has NO loop in it and no memory. The loop is arithmetic that lives")
    print(f"  entirely in diffusion/gaussian_diffusion.py; the model is called from inside it.")
    print(f"  (The real eval also wraps the model in torch.compile + DDP and runs the loop")
    print(f"  under bf16 autocast -- speed, not maths. This probe runs it in fp32 so the")
    print(f"  per-step differences below are the loop's, not the arithmetic's.)")

    # the inputs that stay FIXED for all 250 steps
    x_cond, ctx_px = nwm.context_latents(vae, obs_image, T, device)
    y, rel_t = nwm.action_and_horizon(delta, ts, device)
    model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
    print(f"\n  handed in once and never changed again:")
    show("    x_cond", x_cond[0])
    print(f"    y            = ({y[0,0]:+.4f}, {y[0,1]:+.4f}, {y[0,2]:+.4f})   the action")
    print(f"    rel_t        = {ts}/128 = {rel_t.item():.5f}   the horizon")
    print(f"  the ONE thing that changes per step is x, and the step number t.")

    # ---- 2. the schedule ----
    hr(f"2) THE SCHEDULE  ({nwm.DIFFUSION_STEPS} TRAINING STEPS -> {nwm.RESPACED_STEPS} INFERENCE STEPS)")
    t_map = list(diffusion.timestep_map)
    abar = diffusion.alphas_cumprod                     # respaced: 250 entries
    signal = np.sqrt(abar)                              # q_sample's coefficient on the frame
    noise_lvl = np.sqrt(1.0 - abar)                     # ...and on the noise
    gaps = sorted(set(t_map[i + 1] - t_map[i] for i in range(len(t_map) - 1)))
    stride = (nwm.DIFFUSION_STEPS - 1) / (nwm.RESPACED_STEPS - 1)
    skipped = [v for v in (995, 996, 997) if v not in t_map]
    print(f"  Training used {nwm.DIFFUSION_STEPS} noise levels. Sampling uses {nwm.RESPACED_STEPS} of them,")
    print(f"  picked by space_timesteps (respace.py:18) and listed in diffusion.timestep_map:")
    print(f"    {t_map[:6]} ... {t_map[-3:]}")
    print(f"  stride {nwm.DIFFUSION_STEPS - 1}/{nwm.RESPACED_STEPS - 1} = {stride:.3f}, NOT a round 4:")
    print(f"  the gaps are {gaps} and {skipped[0]} is never visited. Read the map, never assume it.")
    print(f"\n  betas run {diffusion.betas[0]:.5f} -> {diffusion.betas[-1]:.5f} (linear, then respaced).")
    print(f"  What the model is looking at, step by step:\n")
    print(f"  {'step':>5} {'t':>5} {'signal':>9} {'noise':>9}   how much of the true frame is left")
    for s in [249, 225, 200, 175, 150, 125, 100, 75, 50, 25, 10, 0]:
        bar = "#" * int(round(signal[s] * 40))
        print(f"  {s:>5} {t_map[s]:>5} {signal[s]:>8.3f}x {noise_lvl[s]:>8.3f}x   {bar}")
    print(f"\n  Read the loop DOWNWARDS: it starts at step {nwm.RESPACED_STEPS-1} (t={t_map[-1]}), where the")
    print(f"  true frame is multiplied by {signal[-1]:.4f} -- i.e. gone -- and ends at step 0.")

    # ---- 3. the loop ----
    hr("3) THE LOOP, RUN FOR REAL  (p_sample_loop, infer.py:84)")
    torch.manual_seed(seed)
    z = torch.randn(1, 4, latent_size, latent_size, device=device)      # infer.py:81
    print(f"  x starts as pure Gaussian noise, seed {seed}:")
    show("    z", z[0])
    print(f"  then {nwm.RESPACED_STEPS} times: call the model, take one step towards the answer.")
    print(f"  running {nwm.RESPACED_STEPS} steps ...")
    run = sample_loop(diffusion, model, z, model_kwargs, device)
    steps, final = list(run.steps), run.final
    fscale = float(final.norm())
    print(f"  done. kept {len(run)} passes (each with what went in, what came out, and the guess)")
    show("    final", final[0])
    print(f"\n  Each step's dict has two things (gaussian_diffusion.py:420):")
    print(f"    'sample'      the next x -- what the loop carries forward")
    print(f"    'pred_xstart' the model's guess at the FINISHED latent, made fresh")
    print(f"                  every step and then mostly thrown away")
    print(f"  The second is the interesting one: the model has an opinion about the whole")
    print(f"  frame from step {steps[0]} onwards. The loop is it changing its mind {nwm.RESPACED_STEPS} times.")

    # per-pass record -- all arithmetic on the run above, no extra forward passes
    rec, previous = [], None
    for p in run:
        rec.append({
            "step": p.step,
            "t": p.t,
            "signal": round(float(signal[p.step]), 4),
            "x_norm": round(float(p.x_in.norm()), 2),
            "guess_norm": round(float(p.guess.norm()), 2),
            "x_move_pct": round(rel(p.x_out, p.x_in, fscale), 2),
            "guess_move_pct": None if previous is None else round(
                rel(p.guess, previous.guess, fscale), 2),
            "guess_to_final_pct": round(rel(p.guess, final, fscale), 1),
            "guess_cos_final": round(float(torch.nn.functional.cosine_similarity(
                p.guess.flatten(), final.flatten(), dim=0)), 3),
            "outside_pm1_pct": round(float((p.guess.abs() > 1).float().mean()) * 100, 1),
        })
        previous = p
    by_step = {r["step"]: r for r in rec}
    print(f"\n  {'step':>5} {'t':>5} {'signal':>7} {'|x|':>8} {'x moved':>9} "
          f"{'guess moved':>12} {'guess->final':>13} {'cos':>6}")
    for r in [r for r in rec if r["step"] in (249, 240, 225, 200, 175, 150, 125,
                                              100, 75, 50, 25, 10, 5, 1, 0)]:
        gm = f"{r['guess_move_pct']:>11.2f}%" if r["guess_move_pct"] is not None else f"{'--':>12}"
        print(f"  {r['step']:>5} {r['t']:>5} {r['signal']:>7.3f} {r['x_norm']:>8.1f} "
              f"{r['x_move_pct']:>8.2f}% {gm} {r['guess_to_final_pct']:>12.1f}% "
              f"{r['guess_cos_final']:>6.3f}")

    # ---- 4. one pass, opened up ----
    opened = run.at(cfg.step)
    step_s, t_s = opened.step, opened.t
    hr(f"4) ONE PASS, OPENED UP  (step {step_s} of {nwm.RESPACED_STEPS}, t = {t_s})")
    x_in = opened.x_in.to(device)
    x_out = opened.x_out.to(device)
    t_idx = torch.tensor([step_s], device=device)      # the loop indexes 0..249, not 0..999

    # p_mean_variance is what p_sample calls (gaussian_diffusion.py:405). Calling it
    # again on the SAME x is free of side effects -- the model is deterministic -- so
    # this reproduces exactly the step the loop took, without re-running the loop.
    with torch.no_grad():
        pm = diffusion.p_mean_variance(model.forward, x_in, t_idx,
                                       clip_denoised=False, model_kwargs=model_kwargs)
        # ...and the raw model output, so the two halves of it can be named. The model
        # takes the ORIGINAL t; SpacedDiffusion's _WrappedModel (respace.py:126) is what
        # translates step 125 into t=501 on the way in.
        raw = model(x_in, torch.full((1,), float(t_s), device=device), y, x_cond, rel_t)
    eps, var_raw = raw[:, :4], raw[:, 4:]
    mean, pred = pm["mean"], pm["pred_xstart"]
    noise_put_back = x_out - mean

    c1 = float(diffusion.posterior_mean_coef1[step_s])
    c2 = float(diffusion.posterior_mean_coef2[step_s])
    recon = c1 * pred + c2 * x_in
    r1 = float(diffusion.sqrt_recip_alphas_cumprod[step_s])
    r2 = float(diffusion.sqrt_recipm1_alphas_cumprod[step_s])
    pred_from_eps = r1 * x_in - r2 * eps

    print(f"  1. call the model once.  forward(x, t={t_s}, y, x_cond, rel_t)")
    show("     eps  (channels 0-3)", eps[0])
    show("     var  (channels 4-7)", var_raw[0])
    print(f"     eps is a guess at the NOISE inside x. Not a picture -- that was T4's point.")
    print(f"\n  2. turn that into a guess at the FINISHED latent  (_predict_xstart_from_eps, :340)")
    print(f"       pred_xstart = {r1:.4f} * x  -  {r2:.4f} * eps")
    show("     pred_xstart", pred[0])
    print(f"     matches the model output to {float((pred - pred_from_eps).abs().max()):.2e} "
          f"(same formula, checked)")
    print(f"     THIS is decodable. It is the model's answer if you stopped right now.")
    print(f"\n  3. blend that guess with where you already are  (q_posterior_mean_variance, :238)")
    print(f"       mean = {c1:.4f} * pred_xstart  +  {c2:.4f} * x")
    print(f"     residual vs the real mean: {float((mean - recon).abs().max()):.2e}")
    print(f"     At step {step_s} the step is {c1*100:.1f}% new guess and {c2*100:.1f}% where it was.")
    print(f"     That ratio is the whole reason it takes {nwm.RESPACED_STEPS} steps: each one only")
    print(f"     moves a fraction of the way towards the model's current answer.")

    # the variance the model chose, between the two bounds it is allowed
    frac = ((var_raw + 1) / 2).mean().item()
    min_log = float(diffusion.posterior_log_variance_clipped[step_s])
    max_log = float(np.log(diffusion.betas)[step_s])
    drift = float((mean - x_in).norm())
    jog = float(noise_put_back.norm())
    print(f"\n  4. put fresh noise back in  (p_sample, :415)")
    print(f"       x_next = mean + exp(0.5 * log_variance) * randn_like(x)")
    print(f"     The variance is the model's other 4 channels (ModelVarType.LEARNED_RANGE):")
    print(f"     it picks a point between a small bound ({np.exp(min_log):.5f}) and a large one")
    print(f"     ({np.exp(max_log):.5f}), and sat {frac*100:.0f}% of the way up. At this step those")
    print(f"     two bounds differ by {abs(np.exp(max_log)/np.exp(min_log) - 1)*100:.1f}%, so the choice barely matters here;")
    print(f"     they only pull apart near the end of the loop, where the posterior variance")
    print(f"     collapses towards 0 and beta does not.")
    print(f"\n     |mean - x|       = {drift:>6.2f}   the deliberate part of the step")
    print(f"     |noise put back| = {jog:>6.2f}   the fresh random part")
    print(f"     The random part is {jog/max(drift,1e-9):.0f}x the deliberate one. One step is mostly a")
    print(f"     random jog with a small deliberate drift in it -- {nwm.RESPACED_STEPS} jogs whose noise")
    print(f"     cancels and whose drift does not. The old noise is also shrunk every step:")
    print(f"     the two blend weights sum to {(c1 + c2)*100:.1f}%, i.e. x itself is scaled down by")
    print(f"     {100 - (c1 + c2)*100:.1f}% each time round.")
    print(f"     Putting noise back at all is what makes this DDPM ancestral sampling and not")
    print(f"     DDIM: the file has ddim_sample_loop (:606) and the eval does not call it. The")
    print(f"     nonzero_mask at :411 switches the noise off for step 0 only, so the loop's")
    print(f"     last output is clean.")

    # ---- 5. when is the frame actually decided? ----
    hr("5) WHEN IS THE FRAME DECIDED?")
    to_final = np.array([r["guess_to_final_pct"] for r in rec])
    moves = np.array([r["guess_move_pct"] or 0.0 for r in rec])
    marks = {}
    for thr in (50, 25, 10):
        hit = [r for r in rec if r["guess_to_final_pct"] <= thr]
        marks[thr] = hit[0]["step"] if hit else None
    thirds = [moves[:83].sum(), moves[83:166].sum(), moves[166:].sum()]
    thirds = [v / max(moves.sum(), 1e-9) * 100 for v in thirds]
    print(f"  Distance from the running guess to the finished latent, as the loop runs:")
    for thr in (50, 25, 10):
        s_ = marks[thr]
        print(f"    within {thr:>2}% of the answer by step {s_:>3}  "
              f"({(nwm.RESPACED_STEPS - 1 - s_) / (nwm.RESPACED_STEPS - 1) * 100:>4.0f}% of the loop done)"
              if s_ is not None else f"    never gets within {thr}%")
    print(f"\n  Where the guess actually moves (total movement split over the loop):")
    for nm, v in zip(["first third (steps 249-167)", "middle third (166-84)",
                      "last third (83-0)"], thirds):
        print(f"    {nm:<30} {v:>5.1f}%  {'#' * int(round(v / 2))}")
    print(f"\n  Careful with those two readings -- they say different things and both are true.")
    print(f"  {thirds[0]:.0f}% of all the movement happens in the first third, but the guess only gets")
    print(f"  within 25% of its own final answer at step {marks[25]}, {(nwm.RESPACED_STEPS-1-marks[25])/(nwm.RESPACED_STEPS-1)*100:.0f}% of the way through. It never")
    print(f"  stops moving; it just stops moving FAR.")
    print(f"  The filmstrip is what settles the reading: by step ~100 -- still {[r for r in rec if r['step']==100][0]['guess_to_final_pct']:.0f}% away by")
    print(f"  this measure -- the tree, the building and the horizon are already in their")
    print(f"  final places. Everything after that is sharpness and colour. A latent norm")
    print(f"  cannot tell 'redrew the scene' from 'resolved the grass', so read the number")
    print(f"  and the pictures together. Choosing the scene is early; finishing it is late.")

    # ---- 6. clip_denoised=False ----
    hr("6) WHY clip_denoised=False  (infer.py:85)")
    outside = np.array([r["outside_pm1_pct"] for r in rec])
    gmax = max(float(p.guess.abs().max()) for p in run)
    print(f"  p_sample_loop's default is clip_denoised=True: clamp every guess at the")
    print(f"  finished frame into [-1, 1]. That default is for PIXELS. Here x is a VAE")
    print(f"  latent (T3), scaled by {nwm.SCALING} -- it is roughly unit-variance, so a real")
    print(f"  share of it lives outside [-1, 1] legitimately.")
    print(f"    largest value any guess reached : {gmax:.2f}")
    print(f"    share of the FINAL latent outside [-1, 1] : "
          f"{float((final.abs() > 1).float().mean())*100:.1f}%")
    print(f"    peak share of a guess outside [-1, 1]     : {outside.max():.1f}% "
          f"(step {rec[int(outside.argmax())]['step']})")
    print(f"  Clipping would flatten every one of those values onto the same number.")
    print(f"  The eval passes clip_denoised=False for exactly that reason.")

    # ---- 7. two more loops: what does the action decide, what does the noise decide ----
    hr("7) TWO MORE LOOPS  (same code, one input changed)")
    print(f"  (a) same starting noise, action zeroed -- what does the action decide?")
    torch.manual_seed(seed)                       # identical z AND identical per-step noise
    z_a = torch.randn(1, 4, latent_size, latent_size, device=device)
    final_noact = sample_loop(diffusion, model, z_a,
                              dict(y=torch.zeros_like(y), x_cond=x_cond, rel_t=rel_t),
                              device).final
    print(f"  (b) same action, different starting noise (seed {compare_seed}) -- what does the noise decide?")
    torch.manual_seed(compare_seed)
    z_b = torch.randn(1, 4, latent_size, latent_size, device=device)
    final_seed = sample_loop(diffusion, model, z_b, model_kwargs, device).final

    with torch.no_grad():
        z_gt = nwm.encode(vae, gt_image[ts - 1:ts].to(device)).float().cpu()
    d_noact = rel(final, final_noact, fscale)
    d_seed = rel(final, final_seed, fscale)
    d_gt = rel(final, z_gt, fscale)

    # ...and the same three comparisons in pixels, because the latent norm and the
    # picture disagree here, and only one of them is what a person sees.
    target_v = nwm.to_display(gt_image[ts - 1])
    final_v = {nm: nwm.to_display(nwm.decode(vae, z_.to(device))[0]) for nm, z_ in
               (("final_pred", final), ("final_noaction", final_noact),
                ("final_otherseed", final_seed))}
    px_mae = {nm: float(np.abs(v - final_v["final_pred"]).mean())
              for nm, v in final_v.items() if nm != "final_pred"}
    px_mae["ground_truth"] = float(np.abs(target_v - final_v["final_pred"]).mean())

    print(f"\n  distance from the real prediction, two ways:\n")
    print(f"  {'':<26} {'latent, % of |x|':>17} {'pixels, mean abs':>18}")
    print(f"  {'action zeroed out':<26} {d_noact:>16.1f}% {px_mae['final_noaction']:>18.4f}")
    print(f"  {'different starting noise':<26} {d_seed:>16.1f}% {px_mae['final_otherseed']:>18.4f}")
    print(f"  {'the TRUE future frame':<26} {d_gt:>16.1f}% {px_mae['ground_truth']:>18.4f}")
    print(f"\n  The two columns put them in OPPOSITE orders, and that is the lesson.")
    print(f"  In the latent the different seed looks like the bigger change; in pixels it is")
    print(f"  the smaller one -- smaller even than the gap to the true frame.")
    print(f"  Why: a new seed regrows every blade of grass and every leaf. That is a large")
    print(f"  latent distance and almost no change to what the picture shows. Zeroing the")
    print(f"  action moves the robot somewhere else, which is a modest latent distance and")
    print(f"  a different photograph. Look at finals.png: the seed run keeps the building on")
    print(f"  the left and the trees where they are; the no-action run does not.")
    print(f"  So: the noise decides the texture, the action decides the place. And a plain")
    print(f"  norm is the wrong instrument for saying which prediction is better -- which is")
    print(f"  exactly why the real evaluation uses LPIPS and DreamSim instead (T6).")

    # ---- 8. visuals ----
    hr("8) VISUALS  ->  " + out_dir)
    strip_steps = list(cfg.filmstrip_steps)      # checked in read_config, before the GPU

    ctx_v = [nwm.to_display(p) for p in ctx_px]
    for i, v in enumerate(ctx_v):
        save.image(f"ctx_f{i}", v)
    save.image("ground_truth", target_v)

    # the two filmstrips: what x IS at each pass, and what the model THINKS at each pass
    film = []
    for st in strip_steps:
        p = run.at(st)
        save.image(f"loop_x_s{st}", nwm.to_display(nwm.decode(vae, p.x_in.to(device))[0]))
        save.image(f"loop_guess_s{st}", nwm.to_display(nwm.decode(vae, p.guess.to(device))[0]))
        film.append({"step": p.step, "t": p.t,
                     "signal": round(float(signal[p.step]), 4),
                     "guess_to_final_pct": by_step[p.step]["guess_to_final_pct"]})

    for nm, v in final_v.items():          # decoded in section 7, where they were measured
        save.image(nm, v)

    # the one opened-up pass, panel by panel
    save.image("step_x_in", nwm.to_display(nwm.decode(vae, x_in)[0]))
    save.image("step_guess", nwm.to_display(nwm.decode(vae, pred)[0]))
    save.image("step_x_out", nwm.to_display(nwm.decode(vae, x_out)[0]))
    en = eps[0].float().cpu().numpy()
    for ch in range(4):
        save.image(f"step_eps_ch{ch}", en[ch], cmap="magma")

    # --- figure: the two filmstrips together (the whole story in one picture) ---
    fig, axes = plt.subplots(2, len(strip_steps), figsize=(1.85 * len(strip_steps), 4.6))
    for j, s in enumerate(strip_steps):
        for i, tag in enumerate(["loop_x_s", "loop_guess_s"]):
            ax = axes[i, j]; ax.axis("off")
            ax.imshow(plt.imread(save.path(f"{tag}{s}")))
        axes[0, j].set_title(f"step {s}\nt = {t_map[s]}", fontsize=9)
    fig.text(0.005, 0.72, "x, the latent\nbeing carried", fontsize=9, va="center", rotation=90)
    fig.text(0.005, 0.28, "pred_xstart,\nthe model's guess", fontsize=9, va="center", rotation=90)
    fig.suptitle(f"T5 -- the real {nwm.RESPACED_STEPS}-step loop, decoded. TOP: what x actually is. "
                 f"BOTTOM: what the model says the finished frame is, from that same x.",
                 fontsize=11)
    fig.tight_layout(rect=[0.035, 0, 1, 0.9])
    save.figure("filmstrip", fig, dpi=120, bbox_inches="tight")

    # --- figure: the four finished frames ---
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (v, title) in zip(axes, [
            (target_v, f"what really happened\nframe {curr_time + ts}, held out"),
            (final_v["final_pred"], f"the prediction\n{nwm.RESPACED_STEPS} steps, seed {seed}"),
            (final_v["final_noaction"], f"same noise, NO action\n{d_noact:.0f}% away"),
            (final_v["final_otherseed"], f"same action, seed {compare_seed}\n{d_seed:.0f}% away")]):
        ax.axis("off"); ax.imshow(v); ax.set_title(title, fontsize=10)
    fig.suptitle(f"T5 -- three full loops on one scene: {f_curr}, frame {curr_time}, "
                 f"{sec}s ahead", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save.figure("finals", fig, dpi=120, bbox_inches="tight")

    # --- figure: the schedule ---
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(steps, signal[steps], color=TEAL, lw=2, label="the true frame")
    ax.plot(steps, noise_lvl[steps], color=AMBER, lw=2, label="noise")
    for s in strip_steps:
        ax.axvline(s, color=GREY, lw=0.6, alpha=0.35)
    ax.invert_xaxis()
    ax.set_xlabel(f"step   (the loop runs left to right: {steps[0]} → {steps[-1]})")
    ax.set_ylabel("multiplied by")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save.figure("schedule", fig, dpi=130)

    # --- figure: when the frame is decided ---
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(steps, to_final, color=TEAL, lw=2)
    for thr, col in ((50, GREY), (25, GREY), (10, GREY)):
        if marks[thr] is not None:
            ax.axvline(marks[thr], color=col, ls=":", lw=1)
            ax.text(marks[thr], thr + 4, f" {thr}% @ step {marks[thr]}", fontsize=7.5, color=GREY)
    ax.invert_xaxis()
    ax.set_xlabel(f"step   (the loop runs left to right: {steps[0]} → {steps[-1]})")
    ax.set_ylabel("distance to the answer, %")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save.figure("settle", fig, dpi=130)

    # --- figure: how each step blends the guess with where it already is ---
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(steps, diffusion.posterior_mean_coef1[steps] * 100, color=TEAL, lw=2,
            label="the new guess")
    ax.plot(steps, diffusion.posterior_mean_coef2[steps] * 100, color=AMBER, lw=2,
            label="where it already is")
    ax.axvline(step_s, color=GREY, lw=0.8, alpha=0.6)
    ax.invert_xaxis()
    ax.set_xlabel(f"step   (the loop runs left to right: {steps[0]} → {steps[-1]})")
    ax.set_ylabel("% of the next x")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save.figure("mix", fig, dpi=130)

    print(f"  wrote filmstrip.png, finals.png, schedule.png, settle.png, mix.png,")
    print(f"        loop_x_s*.png, loop_guess_s*.png, final_*.png, ground_truth.png,")
    print(f"        ctx_f*.png, step_*.png")

    # ---- 9. machine-readable facts ----
    facts = {
        "trajectory": f_curr,
        "curr_time": int(curr_time),
        "context_size": int(T),
        "context_frame_numbers": [int(curr_time - (T - 1) + i) for i in range(T)],
        "sec": sec,
        "input_fps": nwm.INPUT_FPS,
        "ts": int(ts),
        "target_frame_number": int(curr_time + ts),
        "image_size": int(IMG),
        "latent_size": int(latent_size),
        "in_channels": 4,
        "model": base["model"],
        "checkpoint": nwm.CKP,
        "train_step": ckpt["train_step"] if isinstance(ckpt["train_step"], int) else str(ckpt["train_step"]),
        "params_total_m": round(ckpt["params_total"] / 1e6, 1),
        "vae": nwm.VAE_NAME,
        "scaling": nwm.SCALING,
        "seed": seed,
        "compare_seed": compare_seed,
        "action": [round(float(v), 4) for v in y[0]],
        "rel_t": round(float(rel_t.item()), 5),
        # the schedule
        "diffusion_steps": nwm.DIFFUSION_STEPS,
        "respaced_steps": nwm.RESPACED_STEPS,
        "sampler": "p_sample_loop (DDPM ancestral)",
        "ddim_available_unused": True,
        "clip_denoised": False,
        "timestep_map_head": t_map[:6],
        "timestep_map_tail": t_map[-3:],
        "timestep_gaps": gaps,
        "timestep_stride": round(stride, 3),
        "timestep_never_visited": skipped,
        "beta_first": round(float(diffusion.betas[0]), 6),
        "beta_last": round(float(diffusion.betas[-1]), 6),
        "signal_at_start": round(float(signal[-1]), 4),
        "signal_at_end": round(float(signal[0]), 4),
        "schedule": [{"step": int(s), "t": int(t_map[s]),
                      "signal": round(float(signal[s]), 4),
                      "noise": round(float(noise_lvl[s]), 4)}
                     for s in [249, 225, 200, 175, 150, 125, 100, 75, 50, 25, 10, 0]],
        # the run
        "loop_steps": rec,
        "final_norm": round(fscale, 2),
        "filmstrip": film,
        # one step opened up
        "step_opened": step_s,
        "step_opened_t": t_s,
        "step_mix_new_guess_pct": round(c1 * 100, 1),
        "step_mix_where_it_is_pct": round(c2 * 100, 1),
        "step_xstart_coef_x": round(r1, 4),
        "step_xstart_coef_eps": round(r2, 4),
        "step_xstart_residual": float(f"{float((pred - pred_from_eps).abs().max()):.2e}"),
        "step_mean_residual": float(f"{float((mean - recon).abs().max()):.2e}"),
        "step_noise_put_back": round(float(noise_put_back.norm()), 2),
        "step_signal_removed": round(float((mean - x_in).norm()), 2),
        "step_variance_frac_of_range": round(frac, 3),
        "step_variance_small_bound": round(float(np.exp(min_log)), 5),
        "step_variance_large_bound": round(float(np.exp(max_log)), 5),
        "step_variance_bounds_differ_pct": round(abs(np.exp(max_log) / np.exp(min_log) - 1) * 100, 2),
        "step_mix_sum_pct": round((c1 + c2) * 100, 1),
        "step_drift": round(drift, 2),
        "step_jog_over_drift": round(jog / max(drift, 1e-9), 1),
        "step_eps_std": round(float(eps.std()), 4),
        "step_guess_to_final_pct": by_step[step_s]["guess_to_final_pct"],
        # when is it decided
        "settle_step_50pct": marks[50],
        "settle_step_25pct": marks[25],
        "settle_step_10pct": marks[10],
        "guess_movement_thirds_pct": [round(v, 1) for v in thirds],
        # clip_denoised
        "guess_max_abs": round(gmax, 2),
        "final_outside_pm1_pct": round(float((final.abs() > 1).float().mean()) * 100, 1),
        "guess_outside_pm1_peak_pct": round(float(outside.max()), 1),
        # the comparison loops
        "dist_no_action_pct": round(d_noact, 1),
        "dist_other_seed_pct": round(d_seed, 1),
        "dist_ground_truth_pct": round(d_gt, 1),
        "pixel_mae_vs_prediction": {k: round(v, 4) for k, v in px_mae.items()},
    }
    # Written through debug/common/facts.py, never json.dump directly: that is
    # what stamps the schema version the page checks on load (ADR-0004).
    written = facts_io.write(out_dir, "loop_facts.json", facts, stage="t5",
                             images=save.names)
    print(f"\n  wrote {len(save)} pictures, recorded in the facts file's manifest")
    print(f"  wrote {written}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
