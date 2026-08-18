"""Building the real model, the real VAE and the real schedule.

Every stage from T4 onwards needs the same setup before it can measure
anything: the merged eval config, CDiT with the trained checkpoint in it, the
frozen VAE, the respaced diffusion schedule, and the five things forward()
takes for one scene. T4 had all of it inline because it was the only user.
T5 is the second, and two users make a real seam -- see the closing paragraph
of docs/adr/0005-probe-stays-linear.md, which named this exact moment.

Nothing here re-implements upstream (ADR-0002). The constants are read off the
real inference path and cited line by line; the dataset half lives next door in
debug/common/scene.py.

    from debug.common import model as nwm
    base = nwm.load_config()
    model, ckpt = nwm.build_cdit(base, context_size=4, latent_size=28, device=dev)
    vae = nwm.build_vae(dev)
    diffusion = nwm.build_diffusion()
"""
import os

import torch
import yaml
from diffusers.models import AutoencoderKL

import misc
from diffusion import create_diffusion
from models import CDiT_models

# ---- constants the real inference path uses ----
VAE_NAME = "stabilityai/sd-vae-ft-ema"
SCALING = 0.18215          # isolated_nwm_infer.py:79 / :87
CKP = "0100000"            # isolated_nwm_infer.py argparse default --ckp
INPUT_FPS = 4              # isolated_nwm_infer.py argparse default --input_fps
SECS_SWEPT = [1, 2, 4, 8, 16]   # generate_time(): secs = [2**i for i in range(num_sec_eval)]
DIFFUSION_STEPS = 1000     # create_diffusion default diffusion_steps
RESPACED_STEPS = 250       # create_diffusion(str(250)) at isolated_nwm_infer.py:168


def load_config():
    """The eval config the real run uses: the defaults, updated by the model file.

    isolated_nwm_infer.py:145-151 does exactly this -- reads config/eval_config.yaml,
    then overwrites it with the experiment file passed as --exp. For the walkthrough
    that file is always config/nwm_cdit_xl.yaml, the CDiT-XL run whose checkpoint
    we have.
    """
    with open("config/eval_config.yaml") as f:
        base = yaml.safe_load(f)
    with open("config/nwm_cdit_xl.yaml") as f:
        base.update(yaml.safe_load(f))
    return base


def build_cdit(base, context_size, latent_size, device):
    """CDiT with the trained weights in it, exactly as isolated_nwm_infer.py:162-166.

    Returns (model, info). `info` carries what the checkpoint file says about
    itself, which a probe reports and a page quotes: the path, which key was
    loaded, and how many training steps produced it.

    The real eval additionally wraps this in torch.compile and DistributedDataParallel
    (infer.py:167, :170). Both are speed, not maths, and both make hooks and
    per-step capture harder to read, so a probe leaves them off and says so.
    """
    model = CDiT_models[base["model"]](
        context_size=context_size, input_size=latent_size, in_channels=4)
    ckp_path = f'{base["results_dir"]}/{base["run_name"]}/checkpoints/{CKP}.pth.tar'
    if not os.path.isfile(ckp_path):
        raise SystemExit(f"no checkpoint at {ckp_path}\n"
                         f"  This is the trained CDiT-XL. Nothing downstream can run without it.")
    ckp = torch.load(ckp_path, map_location="cpu", weights_only=False)
    load_msg = model.load_state_dict(ckp["ema"], strict=True)
    info = {
        "path": ckp_path,
        "key": "ema",                                   # the smoothed copy kept while training
        "keys_in_file": sorted(ckp.keys()),
        "train_step": ckp.get("step", ckp.get("train_steps", "?")),
        "load_msg": str(load_msg),
        "params_total": sum(p.numel() for p in model.parameters()),
    }
    del ckp
    model.eval().to(device)
    return model, info


def build_vae(device):
    """The frozen autoencoder from T3. NWM does not train it (infer.py:169)."""
    return AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()


def build_diffusion():
    """The respaced schedule the real eval samples with (infer.py:168).

    250 of the 1000 training steps, chosen by diffusion/respace.py. The kept
    steps are `diffusion.timestep_map`; read it rather than assuming a stride
    (ADR-0002 -- the stride is 4.012, not 4).
    """
    return create_diffusion(str(RESPACED_STEPS))


@torch.no_grad()
def encode(vae, px):
    """Pixels -> latents, scaled the way the model expects (infer.py:79)."""
    return vae.encode(px).latent_dist.sample().mul_(SCALING)


@torch.no_grad()
def decode(vae, z):
    """Latents -> pixels, unscaled and clipped the way the real eval does (infer.py:87-89)."""
    return torch.clip(vae.decode(z / SCALING).sample, -1., 1.)


def to_display(px):
    """One pixel tensor (3, H, W) -> an (H, W, 3) array matplotlib can save.

    misc.unnormalize undoes the dataset's normalisation; the clamp only guards
    against a value a hair outside [0, 1] making imsave wrap around.
    """
    return misc.unnormalize(px.detach().cpu()).clamp(0, 1).permute(1, 2, 0).numpy()


def context_latents(vae, obs_image, context_size, device):
    """The T context frames as ONE batched conditioning tensor (1, T, 4, h, w).

    infer.py:80 encodes all frames in one call and infer.py:82 slices the first
    `num_cond` off as x_cond. Same thing here, with B = 1.
    """
    ctx_px = obs_image[-context_size:].to(device)
    return encode(vae, ctx_px).unsqueeze(0), ctx_px


def action_and_horizon(delta, ts, device):
    """The two ways the request is expressed, both from the same `ts` (infer.py:75-76, :111).

    y      the whole horizon's motion added up into one (dx, dy, dyaw): WHERE the
           robot ends up, not the path it takes.
    rel_t  ts / 128: HOW LONG it takes to get there.
    """
    y = delta[:ts].sum(dim=0, keepdim=True).to(device).float()
    rel_t = torch.full((1,), ts / 128.0, device=device)
    return y, rel_t
