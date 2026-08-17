"""
T3 VAE probe for the NWM debug walkthrough.

Takes ONE real RECON frame (the same tensor EvalDataset hands the pipeline) and
pushes it through the *exact* two calls inference uses:

    isolated_nwm_infer.py:79   x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    isolated_nwm_infer.py:87   samples = vae.decode(samples / 0.18215).sample

and then reports, with real numbers:
  1. what the VAE is (sd-vae-ft-ema) and how big it is,
  2. the pixel tensor going in (shape / range / element count),
  3. the encode: latent_dist (a Gaussian per cell), the sample, the 0.18215
     rescale -- and why that constant exists,
  4. how much smaller the latent is (what that saves inside CDiT is T4's to
     measure, not T3's),
  5. the decode: reconstruction + per-pixel error (MSE / PSNR / worst pixel),
  6. a visual: original | reconstruction | amplified error, plus the 4 latent
     channels rendered as 28x28 images,
  7. debug/out/t3/<scene>/vae_facts.json so the write-up can't drift.

Pick a scene by editing debug/t3/config.yaml (same keys as T2), then run inside
the nwm_debug container:
    cd /app && python debug/t3/probe.py
Outputs to debug/out/t3/<scene>/ (gitignored).
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
from diffusers.models import AutoencoderKL

import misc
from debug.common.report import hr, show
from debug.common.scene import build_recon_eval_dataset, resolve_scene

# ---- the constants the real inference path uses (isolated_nwm_infer.py) ----
VAE_NAME = "stabilityai/sd-vae-ft-ema"
SCALING = 0.18215          # SD latent scaling constant, hardcoded at lines 79/87
IMAGE_SIZE = 224
PATCH_SIZE = 2             # CDiT-XL/2 patchifies the latent 2x2 (models.py:306);
                           # used only to explain how the 4 latents are combined (T4 owns the cost)
DIFFUSION_STEPS = 250      # create_diffusion(str(250)) in the real eval


def pick_frame(obs_image, cfg, context_size):
    """Which context frame to encode. Default 'now' = obs_image[-1]."""
    want = cfg.get("frame", "now")
    if want in (None, "", "now", "last"):
        i = context_size - 1
        label = "now (last context frame)"
    else:
        i = int(want)
        if not (0 <= i < context_size):
            raise SystemExit(f"t3_config: frame={want} out of range 0..{context_size - 1}.")
        rel = i - (context_size - 1)
        label = "now (last context frame)" if rel == 0 else f"context frame {i} (t={rel})"
    return i, label


def main():
    cfg = yaml.safe_load(open("debug/t3/config.yaml")) if os.path.exists("debug/t3/config.yaml") else {}
    cfg = cfg or {}

    with open("config/eval_config.yaml") as f:
        base = yaml.safe_load(f)
    with open("config/nwm_cdit_xl.yaml") as f:
        base.update(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_recon_eval_dataset(base)
    row, f_curr, curr_time, scene_tag, mode = resolve_scene(ds, cfg, "debug/t3/config.yaml")
    _, obs_image, _, _ = ds[row]                    # (context_size, 3, 224, 224)
    fi, frame_label = pick_frame(obs_image, cfg, ds.context_size)
    frame_number = curr_time - (ds.context_size - 1) + fi

    hr("SCENE + FRAME  (from debug/t3/config.yaml)")
    print(f"  mode         : {mode}")
    print(f"  trajectory   : {f_curr}")
    print(f"  curr_time    : frame {curr_time}   ('now')")
    print(f"  encoding     : obs_image[{fi}] = {frame_label}  -> jpg frame {frame_number}")
    print(f"  device       : {device}")

    # ---- 1. the VAE itself ----
    hr("1) THE VAE  (loaded exactly as isolated_nwm_infer.py:169)")
    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device)
    vae.eval()
    n_enc = sum(p.numel() for p in vae.encoder.parameters())
    n_dec = sum(p.numel() for p in vae.decoder.parameters())
    vc = vae.config
    print(f"  {VAE_NAME}   (AutoencoderKL, frozen -- never trained by NWM)")
    print(f"  latent_channels     : {vc.latent_channels}")
    print(f"  block_out_channels  : {tuple(vc.block_out_channels)}  -> "
          f"{len(vc.block_out_channels) - 1} downsampling stages = 2^{len(vc.block_out_channels) - 1} "
          f"= {2 ** (len(vc.block_out_channels) - 1)}x spatial")
    print(f"  encoder params      : {n_enc/1e6:.1f} M")
    print(f"  decoder params      : {n_dec/1e6:.1f} M")
    print(f"  total params        : {(n_enc + n_dec)/1e6:.1f} M")
    print(f"  scaling used by NWM : {SCALING}  (hardcoded, isolated_nwm_infer.py:79 / :87)")

    # ---- 1b. ALL the context frames, exactly as inference does it ----
    hr("1b) THE REAL CALL: every context frame is encoded, in ONE batch")
    ctx = obs_image.to(device)                       # (T, 3, 224, 224), T = context_size
    T = ctx.shape[0]
    with torch.no_grad():
        # mirrors isolated_nwm_infer.py:78-79 exactly (there B=batch, here B=1):
        #   x = x.flatten(0,1); x = vae.encode(x)...mul_(0.18215).unflatten(0,(B,T))
        ctx_lat = vae.encode(ctx).latent_dist.sample().mul_(SCALING)   # (T, 4, 28, 28)
    print(f"  obs_image           : {tuple(ctx.shape)}   ({T} context frames)")
    print(f"  -> flatten(0,1), ONE vae.encode call on all {T}, then unflatten")
    print(f"  x_cond              : {tuple(ctx_lat.shape)}   ({T} SEPARATE latents")
    print(f"  They are NOT merged by the VAE -- it has no idea the frames are related.")
    print(f"  CDiT is what combines them (models.py:234):")
    tok_per = (ctx_lat.shape[-1] // PATCH_SIZE) * (ctx_lat.shape[-2] // PATCH_SIZE)
    print(f"    each latent -> {tok_per} patches, + a per-slot position embedding")
    print(f"    pos_embed is ({T} + 1, {tok_per}, hidden) -- one slot per context frame,")
    print(f"    plus one for the frame being predicted, so frame ORDER is preserved.")
    print(f"    then flatten(1,2) -> {T} x {tok_per} = {T * tok_per} context tokens in one sequence,")
    print(f"    which the {tok_per} tokens of the predicted frame cross-attend to.")

    # ---- 2. pixels in ----
    hr("2) PIXELS IN  (what EvalDataset produced, misc.transform output)")
    x = obs_image[fi:fi + 1].to(device)             # (1, 3, 224, 224), keep batch axis
    show("x (pixels)", x[0])
    n_pix = x[0].numel()
    print(f"  elements    : 3 x {IMAGE_SIZE} x {IMAGE_SIZE} = {n_pix:,} floats")
    print(f"  range       : [-1, 1] (Normalize(0.5, 0.5) at the end of misc.transform)")
    print(f"  mean/std    : {x.mean().item():+.4f} / {x.std().item():.4f}")

    # ---- 2b. inside the encoder: what each downsampling stage does ----
    hr("2b) INSIDE THE ENCODER  (real output shape after each stage)")
    stages = []
    handles = []

    def rec(tag):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            stages.append((tag, tuple(o.shape[1:])))
        return hook

    for i, blk in enumerate(vae.encoder.down_blocks):
        handles.append(blk.register_forward_hook(rec(f"down_block[{i}]")))
    handles.append(vae.encoder.mid_block.register_forward_hook(rec("mid_block")))
    with torch.no_grad():
        _ = vae.encode(x)
    for h in handles:
        h.remove()
    prev = tuple(x.shape[1:])
    print(f"  {'stage':<16} {'output (C, H, W)':<20} note")
    print(f"  {'input':<16} {str(prev):<20} the RGB frame")
    for tag, shp in stages:
        note = "halves H and W" if shp[1] < prev[1] else "same size, refines features"
        print(f"  {tag:<16} {str(shp):<20} {note}")
        prev = shp
    print(f"  {'conv_out+quant':<16} {'(8, 28, 28)':<20} 8 = 4 mean + 4 log-variance channels")
    print(f"  -> 3 of the 4 down_blocks halve the resolution: 224 -> 112 -> 56 -> 28.")
    print(f"     Channels go UP as resolution goes DOWN, then collapse to 4 at the end.")

    # ---- 3. encode ----
    hr("3) ENCODE  ->  vae.encode(x).latent_dist.sample().mul_(0.18215)")
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        mean, std = posterior.mean, posterior.std
        z_raw = posterior.sample()                  # stochastic draw
        z = z_raw * SCALING                         # what CDiT actually sees
        z_mean_scaled = mean * SCALING

    print("  encode returns a DISTRIBUTION, not a single vector:")
    show("  .mean", mean[0])
    show("  .std", std[0])
    print(f"    -> one Gaussian N(mean, std) per latent cell; .sample() draws from it.")
    print(f"    -> std is tiny (avg {std.mean().item():.4f}), so the draw is basically the mean:")
    print(f"       max |sample - mean| = {(z_raw - mean).abs().max().item():.4f}")
    # one concrete cell, so "a distribution per cell" stops being abstract
    flat_std = std[0].flatten()
    k = int(flat_std.argmax())                       # the WORST cell (most uncertain)
    kc, krem = k // (lat_hw := std.shape[-1] * std.shape[-2]), k % lat_hw
    kh, kw = krem // std.shape[-1], krem % std.shape[-1]
    cell = {
        "c": int(kc), "h": int(kh), "w": int(kw),
        "mean": float(mean[0, kc, kh, kw]),
        "std": float(std[0, kc, kh, kw]),
        "sample": float(z_raw[0, kc, kh, kw]),
    }
    print(f"\n  ONE ACTUAL CELL -- latent[{cell['c']}, {cell['h']}, {cell['w']}] "
          f"(the single most uncertain of the {mean[0].numel():,} cells):")
    print(f"    encoder said  mean = {cell['mean']:+.4f}")
    print(f"    encoder said  std  = {cell['std']:.4f}   <- its uncertainty about that value")
    print(f"    .sample() drew       {cell['sample']:+.4f}   = mean + std * (a random draw)")
    print(f"    -> the noise changed the value by {abs(cell['sample'] - cell['mean']):.4f}, "
          f"i.e. {abs(cell['sample'] - cell['mean']) / max(abs(cell['mean']), 1e-9) * 100:.3f}% of it.")
    print(f"    Values span {mean.min().item():+.1f}..{mean.max().item():+.1f}; the noise is 4 decimals down.")
    print()
    show("  z_raw (sample)", z_raw[0])
    show("  z (x0.18215)", z[0])
    print(f"\n  WHY 0.18215: raw SD latents have std ~= {z_raw.std().item():.3f}; the diffusion")
    print(f"  noise schedule assumes unit-variance data. {SCALING:.5f} ~= 1/{1/SCALING:.2f} rescales")
    print(f"  them to std ~= {z.std().item():.3f}. Decode divides it straight back out.")

    # ---- 4. how much smaller ----
    # Only the size change belongs to T3. How much that saves inside CDiT is
    # measured in T4, where patches and attention are actually introduced --
    # keeping it here would give two stages the same claim to disagree about.
    hr("4) HOW MUCH SMALLER  (real numbers for this frame)")
    lat_c, lat_h, lat_w = z.shape[1], z.shape[2], z.shape[3]
    n_lat = lat_c * lat_h * lat_w
    print(f"  pixel tensor  (3, {IMAGE_SIZE}, {IMAGE_SIZE})  = {n_pix:,} numbers")
    print(f"  latent tensor ({lat_c}, {lat_h}, {lat_w})    = {n_lat:,} numbers")
    print(f"  -> {n_pix / n_lat:.1f}x fewer numbers, {IMAGE_SIZE // lat_h}x smaller per side")
    print(f"  The model that consumes this runs {DIFFUSION_STEPS} times per predicted frame,")
    print(f"  while encode and decode run once each -- see T4 for what that saves.")

    # ---- 5. decode ----
    hr("5) DECODE  ->  vae.decode(z / 0.18215).sample, then clip to [-1, 1]")
    dstages = []
    handles = []
    for i, blk in enumerate(vae.decoder.up_blocks):
        handles.append(blk.register_forward_hook(
            lambda _m, _i, out, i=i: dstages.append((f"up_block[{i}]", tuple(
                (out[0] if isinstance(out, tuple) else out).shape[1:])))))
    with torch.no_grad():
        _ = vae.decode(z / SCALING)
    for h in handles:
        h.remove()
    print("  the decoder mirrors the encoder, doubling instead of halving:")
    print(f"    {'latent in':<16} {str(tuple(z.shape[1:]))}")
    for tag, shp in dstages:
        print(f"    {tag:<16} {shp}")
    print(f"    {'conv_out':<16} (3, {IMAGE_SIZE}, {IMAGE_SIZE})   back to RGB\n")

    with torch.no_grad():
        recon = vae.decode(z / SCALING).sample
        recon = torch.clip(recon, -1.0, 1.0)
        recon_from_mean = torch.clip(vae.decode(z_mean_scaled / SCALING).sample, -1.0, 1.0)

    show("recon (pixels)", recon[0])
    # error in [0,1] viewing space, which is what "1/255 of a grey level" refers to
    a = misc.unnormalize(x[0]).clamp(0, 1)
    b = misc.unnormalize(recon[0]).clamp(0, 1)
    err = (a - b).abs()
    mse = torch.mean((a - b) ** 2).item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"\n  reconstruction error, measured in [0,1] pixel space:")
    print(f"    MSE            : {mse:.6f}")
    print(f"    PSNR           : {psnr:.2f} dB")
    print(f"    mean |error|   : {err.mean().item():.5f}  ({err.mean().item() * 255:.2f} / 255 grey levels)")
    print(f"    worst pixel    : {err.max().item():.5f}  ({err.max().item() * 255:.1f} / 255)")
    print(f"    % pixels off by >2/255 : {(err > 2/255).float().mean().item() * 100:.1f}%")
    samp_vs_mean = (recon - recon_from_mean).abs().max().item()
    print(f"\n  decode(sample) vs decode(mean) max pixel diff: {samp_vs_mean:.5f}")
    print(f"    -> the stochastic .sample() is visually irrelevant; the information")
    print(f"       lives in .mean. The loss you see above is the VAE's own ceiling:")
    print(f"       NWM can never beat it, because every prediction is decoded through it.")

    # ---- 6. visual ----
    hr("6) VISUAL  ->  debug/out/t3")
    out_dir = f"debug/out/t3/{scene_tag}"
    os.makedirs(out_dir, exist_ok=True)

    a_np = a.cpu().permute(1, 2, 0).numpy()
    b_np = b.cpu().permute(1, 2, 0).numpy()
    e_np = err.cpu().mean(0).numpy()                # per-pixel mean abs error

    fig = plt.figure(figsize=(16.5, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.15)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.imshow(a_np)
    ax.set_title(f"ORIGINAL\n(3, {IMAGE_SIZE}, {IMAGE_SIZE}) = {n_pix:,} numbers", fontsize=11)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.imshow(b_np)
    ax.set_title(f"RECONSTRUCTION (encode -> decode)\nPSNR {psnr:.1f} dB, "
                 f"mean err {err.mean().item()*255:.2f}/255", fontsize=11)

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    im = ax.imshow(e_np, cmap="inferno", vmin=0, vmax=max(e_np.max(), 1e-6))
    ax.set_title(f"WHAT WAS LOST (|error|, auto-scaled)\nmax {err.max().item()*255:.1f}/255", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)

    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    ax.text(0.02, 0.97,
            f"pixels   (3, {IMAGE_SIZE}, {IMAGE_SIZE})\n"
            f"           = {n_pix:,} numbers\n\n"
            f"   |  vae.encode(x)\n"
            f"   |  .latent_dist.sample()\n"
            f"   |  .mul_({SCALING})\n"
            f"   v\n\n"
            f"latent   ({lat_c}, {lat_h}, {lat_w})\n"
            f"           = {n_lat:,} numbers\n\n"
            f"   {n_pix / n_lat:.0f}x fewer values\n"
            f"   {IMAGE_SIZE // lat_h}x smaller per side",
            va="top", ha="left", fontsize=11, family="monospace",
            transform=ax.transAxes)

    z_np = z[0].cpu().float().numpy()
    for c in range(lat_c):
        ax = fig.add_subplot(gs[1, c]); ax.axis("off")
        ax.imshow(z_np[c], cmap="viridis")
        ax.set_title(f"latent channel {c}  ({lat_h}x{lat_w})\n"
                     f"min {z_np[c].min():+.2f}  max {z_np[c].max():+.2f}", fontsize=10)

    fig.suptitle(
        f"T3 -- VAE round trip  |  traj '{f_curr}'  frame {frame_number} ({frame_label})\n"
        f"TOP: original -> reconstruction -> what the VAE threw away      "
        f"BOTTOM: the {lat_c} latent channels CDiT actually denoises",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    viz_path = os.path.join(out_dir, "vae_roundtrip.png")
    fig.savefig(viz_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {viz_path}")

    # every context frame + its latent, for the "what happens in order" story
    ctx_v = misc.unnormalize(ctx.cpu()).clamp(0, 1)
    ctx_l = ctx_lat.cpu().float().numpy()
    for t in range(T):
        plt.imsave(os.path.join(out_dir, f"ctx_f{t}.png"), ctx_v[t].permute(1, 2, 0).numpy())
        for c in range(lat_c):
            plt.imsave(os.path.join(out_dir, f"ctx_f{t}_ch{c}.png"), ctx_l[t, c], cmap="viridis")
    print(f"  wrote {out_dir}/ctx_f0..{T-1}.png + ctx_f*_ch*.png ({T} frames x {lat_c} channels)")

    # individual panels too, so debug/t3/build_page.py can lay them out itself
    plt.imsave(os.path.join(out_dir, "original.png"), a_np)
    plt.imsave(os.path.join(out_dir, "recon.png"), b_np)
    plt.imsave(os.path.join(out_dir, "error.png"), e_np, cmap="inferno")
    for c in range(lat_c):
        plt.imsave(os.path.join(out_dir, f"latent_ch{c}.png"), z_np[c], cmap="viridis")
    print(f"  wrote {out_dir}/original.png, recon.png, error.png, latent_ch0..{lat_c-1}.png")

    # ---- 7. machine-readable facts ----
    facts = {
        "trajectory": f_curr,
        "curr_time": int(curr_time),
        "frame_index_in_context": int(fi),
        "frame_number": int(frame_number),
        "frame_label": frame_label,
        "vae": VAE_NAME,
        "vae_class": type(vae).__name__,
        "vae_total_params_m": round((n_enc + n_dec) / 1e6, 1),
        "scaling": SCALING,
        "context_size": int(T),
        "context_frame_numbers": [int(curr_time - (T - 1) + t) for t in range(T)],
        "context_latent_shape": list(ctx_lat.shape),
        "tokens_per_frame": int(tok_per),
        "context_tokens_total": int(T * tok_per),
        "encoder_params_m": round(n_enc / 1e6, 1),
        "decoder_params_m": round(n_dec / 1e6, 1),
        "pixel_shape": list(x.shape[1:]),
        "pixel_numel": int(n_pix),
        "pixel_min": round(float(x.min()), 4),
        "pixel_max": round(float(x.max()), 4),
        "latent_shape": list(z.shape[1:]),
        "latent_numel": int(n_lat),
        "latent_min": round(float(z.min()), 4),
        "latent_max": round(float(z.max()), 4),
        "latent_std_raw": round(float(z_raw.std()), 4),
        "latent_std_scaled": round(float(z.std()), 4),
        "posterior_std_mean": round(float(std.mean()), 4),
        "encoder_stages": [{"stage": t, "shape": list(s)} for t, s in stages],
        "decoder_stages": [{"stage": t, "shape": list(s)} for t, s in dstages],
        "example_cell": {k2: (v2 if isinstance(v2, int) else round(v2, 4)) for k2, v2 in cell.items()},
        "latent_mean_min": round(float(mean.min()), 2),
        "latent_mean_max": round(float(mean.max()), 2),
        "sample_minus_mean_max": round(float((z_raw - mean).abs().max()), 4),
        "compression_ratio": round(n_pix / n_lat, 1),
        "spatial_downsample": IMAGE_SIZE // lat_h,
        "diffusion_steps": DIFFUSION_STEPS,
        "mse": round(mse, 6),
        "psnr_db": round(float(psnr), 2),
        "mean_abs_err": round(float(err.mean()), 5),
        "mean_abs_err_255": round(float(err.mean()) * 255, 2),
        "max_abs_err": round(float(err.max()), 5),
        "max_abs_err_255": round(float(err.max()) * 255, 1),
        "pct_pixels_off_gt_2_255": round(float((err > 2 / 255).float().mean()) * 100, 1),
        "sample_vs_mean_decode_max": round(samp_vs_mean, 5),
    }
    with open(os.path.join(out_dir, "vae_facts.json"), "w") as f:
        json.dump(facts, f, indent=2)
    print(f"  wrote {out_dir}/vae_facts.json")
    print("\nDONE.")


if __name__ == "__main__":
    main()
