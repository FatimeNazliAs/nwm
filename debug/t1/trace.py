# T1 trace helper. Runs ONE RECON sample through the whole pipeline, printing
# the shape/range at each boundary + saving images. Lives in debug/ but must run
# as if from the repo root, so anchor sys.path + cwd there before importing repo modules.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # so config/, data_splits/, and outputs resolve from the repo root

import yaml, torch
from diffusers.models import AutoencoderKL
from models import CDiT_models
from diffusion import create_diffusion
# reuse the real inference helpers so the trace stays faithful to isolated_nwm_infer.py
from isolated_nwm_infer import get_dataset_eval, save_image

DEVICE = 'cuda'

# --- knobs: read from debug/t1/config.yaml (edit that file, no container rebuild) ---
_cfg = yaml.safe_load(open('debug/t1/config.yaml')) if os.path.exists('debug/t1/config.yaml') else {}
SAMPLE     = int(_cfg.get('sample', 0))       # which RECON scene (0..499)
SEC        = int(_cfg.get('sec', 1))          # horizon in seconds (1..16)
DIFF_STEPS = int(_cfg.get('diff_steps', 100)) # denoising steps (real eval = 250)

# each run gets its OWN folder, so changing sample/sec never overwrites earlier results
OUT = f'debug/out/t1/sample{SAMPLE}_sec{SEC}'; os.makedirs(OUT, exist_ok=True)
print(f"[config] sample={SAMPLE}  sec={SEC}  diff_steps={DIFF_STEPS}  ->  {OUT}/")

def banner(n, txt): print(f"\n{'='*70}\n[{n}] {txt}\n{'='*70}")
def show(name, t):
    print(f"  {name:<22} shape={tuple(t.shape)}  dtype={str(t.dtype).replace('torch.','')}"
          f"  range=[{t.min().item():+.3f}, {t.max().item():+.3f}]")

# ---- config (base eval_config + nwm_cdit_xl override, exactly like infer.py) ----
cfg = yaml.safe_load(open('config/eval_config.yaml'))
cfg.update(yaml.safe_load(open('config/nwm_cdit_xl.yaml')))
image_size = cfg['image_size']; latent_size = image_size // 8; num_cond = cfg['context_size']
input_fps = 4; timestep = SEC * input_fps
print(f"image_size={image_size}  latent_size={latent_size}  context(num_cond)={num_cond}  "
      f"predict {SEC}s ahead => sum {timestep} action steps")

# ================================================================== 0. DATA
banner(0, "DATA  (datasets.py EvalDataset.__getitem__ -> DataLoader)")
ds = get_dataset_eval(cfg, 'recon', 'time')   # same builder isolated_nwm_infer.py uses
print(f"  dataset has {len(ds)} eval samples; taking sample {SAMPLE}")
idxs, obs_image, gt_image, delta = torch.utils.data.default_collate([ds[SAMPLE]])  # add batch dim like DataLoader
show("obs_image (context)", obs_image)   # [B, T, 3, 224, 224]  T past frames
show("gt_image  (futures)", gt_image)     # [B, 64, 3, 224, 224] all future frames
show("delta     (action)", delta)         # [B, 64, action_dim] per-step motion
obs_image = obs_image[:, -num_cond:].to(DEVICE)          # keep last num_cond frames (infer.py:207)
curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True).to(DEVICE)  # generate_time(): sum action over the horizon
show("curr_delta (summed)", curr_delta)   # the single action fed to the model
for j in range(obs_image.shape[1]):                              # save ALL context frames (t-3 ... now)
    save_image(f'{OUT}/00_context_{j}.png', obs_image[0, j], True)
save_image(f'{OUT}/03_ground_truth_{SEC}s.png', gt_image[0, timestep-1], True)  # what really happened at horizon

# ---- load models ----
banner('M', "load VAE + CDiT-XL checkpoint")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE); vae.eval()
model = CDiT_models[cfg['model']](context_size=num_cond, input_size=latent_size, in_channels=4)
ckp = torch.load(f"{cfg['results_dir']}/{cfg['run_name']}/checkpoints/0100000.pth.tar",
                 map_location='cpu', weights_only=False)
print("  load_state_dict:", model.load_state_dict(ckp['ema'], strict=True))
model.eval().to(DEVICE)
diffusion = create_diffusion(str(DIFF_STEPS))

with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
    B, T = obs_image.shape[:2]
    # ============================================================== 1. VAE ENCODE
    banner(1, "VAE ENCODE  pixels -> latent  (infer.py:79  vae.encode * 0.18215)")
    x = obs_image.flatten(0, 1)
    show("pixels in", x)                                   # [B*T, 3, 224, 224]
    x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
    show("latent out", x)                                  # [B, T, 4, 28, 28]
    x_cond = x[:, :num_cond]                               # context latents = CDiT conditioning [N, Tctx, 4, 28, 28]
    show("x_cond (conditioning)", x_cond)

    # ============================================================== 2. NOISE START
    banner(2, "START FROM PURE NOISE  z ~ N(0,1)  (infer.py:81)")
    z = torch.randn(B, 4, latent_size, latent_size, device=DEVICE)
    show("z (noise latent)", z)
    save_image(f'{OUT}/01_noise_decoded.png', vae.decode(z / 0.18215).sample[0].clamp(-1,1), True)

    # ============================================================== 3. CDiT + DIFFUSION
    banner(3, f"CDiT DENOISE via diffusion.p_sample_loop  ({DIFF_STEPS} steps)  (infer.py:84)")
    rel_t = (torch.ones(B) * (1./128.) * timestep).to(DEVICE)
    y = curr_delta.flatten(0, 1)
    show("y (action to model)", y); show("rel_t (time cond)", rel_t)
    samples = diffusion.p_sample_loop(model.forward, z.shape, z, clip_denoised=False,
                model_kwargs=dict(y=y, x_cond=x_cond, rel_t=rel_t), progress=True, device=DEVICE)
    show("denoised latent", samples)                       # [B, 4, 28, 28] clean predicted latent

    # ============================================================== 4. VAE DECODE
    banner(4, "VAE DECODE  latent -> pixels  (infer.py:87)")
    pred = vae.decode(samples / 0.18215).sample
    show("pred pixels", pred)
    pred = torch.clip(pred, -1., 1.)
    save_image(f'{OUT}/02_prediction_{SEC}s.png', pred[0], True)

print(f"\nDONE. Images written to {OUT}/:")
for f in sorted(os.listdir(OUT)): print("   ", f)
