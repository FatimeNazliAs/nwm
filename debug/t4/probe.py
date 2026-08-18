"""
T4 CDiT probe for the NWM debug walkthrough.

CDiT is the only thing NWM trains. Everything else in the pipeline is either
fixed data handling (T2) or a frozen off-the-shelf autoencoder (T3). This probe
builds the real model, loads the real checkpoint, and runs ONE real forward pass
-- the same call diffusion/p_sample_loop makes 250 times per predicted frame:

    isolated_nwm_infer.py:162   model = CDiT_models['CDiT-XL/2'](context_size=4, input_size=28, in_channels=4)
    isolated_nwm_infer.py:164   model.load_state_dict(ckp["ema"], strict=True)
    isolated_nwm_infer.py:84    diffusion.p_sample_loop(model.forward, ..., model_kwargs=dict(y=, x_cond=, rel_t=))
    models.py:226               CDiT.forward(x, t, y, x_cond, rel_t)

and reports, with real numbers:
  1. what CDiT-XL/2 is, how big it is, and where the parameters actually sit,
  2. the five things forward() takes, with this scene's real values,
  3. patchify: latent -> tokens, and the per-slot position embedding that is the
     ONLY reason the model knows which context frame is which,
  4. the conditioning vector c = timestep + horizon + action, and how adaLN-Zero
     sprays it over all 28 blocks as 11 numbers per block,
  5. one block traced layer by layer, with real attention weights: where does a
     patch of the FUTURE frame look in the four PAST frames,
  6. the output: 196 tokens -> (8, 28, 28) = predicted noise + predicted variance,
  7. ablations that prove the conditioning is load-bearing (zero the action,
     reverse the context order, change the horizon -- watch the output move),
  8. debug/out/t4/<scene>/cdit_facts.json so the write-up can't drift.

Pick a scene and the knobs by editing debug/t4/config.yaml, then run inside the
nwm_debug container:
    cd /app && python debug/t4/probe.py
Outputs to debug/out/t4/<scene>/ (gitignored).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)   # repo root, so `import models` / `datasets` resolve
os.chdir(ROOT)             # so config/, data_splits/, outputs resolve from repo root

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import misc
from debug.common import facts as facts_io
from debug.common import images as images_io
from debug.common import model as nwm
from debug.common.report import hr, show
from debug.common.scene import build_recon_eval_dataset, resolve_scene

CONFIG = "debug/t4/config.yaml"

# The constants and the model/VAE/schedule setup used to live here. They now live
# in debug/common/model.py, which T5 also uses -- two users make the seam real
# (docs/adr/0005-probe-stays-linear.md closes on exactly this moment). Aliased so
# the narration below still reads as prose.
VAE_NAME = nwm.VAE_NAME
SCALING = nwm.SCALING
CKP = nwm.CKP
INPUT_FPS = nwm.INPUT_FPS
SECS_SWEPT = nwm.SECS_SWEPT
DIFFUSION_STEPS = nwm.DIFFUSION_STEPS
RESPACED_STEPS = nwm.RESPACED_STEPS

# the 11 things adaLN-Zero produces per block, in the order models.py:105 unpacks them
ADALN_NAMES = [
    ("shift_msa", "self-attention: shift"),
    ("scale_msa", "self-attention: scale"),
    ("gate_msa", "self-attention: how much of it to keep"),
    ("shift_ca_xcond", "cross-attention, the CONTEXT side: shift"),
    ("scale_ca_xcond", "cross-attention, the CONTEXT side: scale"),
    ("shift_ca_x", "cross-attention, the QUERY side: shift"),
    ("scale_ca_x", "cross-attention, the QUERY side: scale"),
    ("gate_ca_x", "cross-attention: how much of it to keep"),
    ("shift_mlp", "feed-forward: shift"),
    ("scale_mlp", "feed-forward: scale"),
    ("gate_mlp", "feed-forward: how much of it to keep"),
]


def read_config(path=CONFIG):
    """Read and check debug/t4/config.yaml before anything expensive is built.

    `diffusion_t` used to be checked after the model, the VAE and the context
    latents were all in memory, and `attn_step` was never checked at all -- an
    out-of-range value became an IndexError minutes in. Both are validated here,
    against the real respaced schedule, which costs nothing: create_diffusion
    only builds numpy arrays.

    Knobs whose valid range depends on the model (`block`, `query_patch`) stay
    where they are; nothing here can know the depth or the grid yet.
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
        raise SystemExit(f"{path}: sec={sec} is outside 1..{nwm.SECS_SWEPT[-1]}.")

    t_map = list(nwm.build_diffusion().timestep_map)
    t_val = whole("diffusion_t", t_map[-1])
    if t_val not in t_map:
        raise SystemExit(
            f"{path}: diffusion_t={t_val} is never used by the real loop.\n"
            f"  create_diffusion(str({nwm.RESPACED_STEPS})) keeps only these {len(t_map)} of "
            f"{nwm.DIFFUSION_STEPS} steps:\n"
            f"    {t_map[:6]} ... {t_map[-3:]}\n"
            f"  Pick one of those (the loop starts at {t_map[-1]} and ends at {t_map[0]}).")

    attn_step = whole("attn_step", 25)
    if not (0 <= attn_step <= last):
        raise SystemExit(
            f"{path}: attn_step={attn_step} is outside 0..{last}.\n"
            f"  It indexes the {nwm.RESPACED_STEPS} steps of the loop, not the "
            f"{nwm.DIFFUSION_STEPS} training ones.")

    return cfg, sec, t_val, attn_step


def parse_query_patch(want, grid):
    """config `query_patch` -> (index, row, col) in the grid x grid patch grid."""
    if want in (None, "", "center", "centre"):
        r = c = grid // 2
    elif isinstance(want, str) and "," in want:
        r, c = (int(v) for v in want.split(","))
    else:
        i = int(want)
        r, c = i // grid, i % grid
    if not (0 <= r < grid and 0 <= c < grid):
        raise SystemExit(f"debug/t4/config.yaml: query_patch={want} outside the {grid}x{grid} grid.")
    return r * grid + c, r, c


def self_attn_weights(attn, x):
    """Re-run timm's Attention by hand to get the weights it never returns.

    Mirrors timm.models.vision_transformer.Attention.forward's non-fused path
    exactly; the model itself uses the fused kernel, which throws the matrix away.
    """
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    w = (q * attn.scale) @ k.transpose(-2, -1)
    return w.softmax(dim=-1)                      # (B, heads, N, N)


def cross_attn_weights(block, grabbed):
    """Re-run one block's cross-attention to get the weights the forward pass discards.

    `grabbed` is what the forward hook captured for this block: the exact query,
    key and value tensors it was called with. Replaying them with
    need_weights=True is the only way to see where the block looked -- the real
    call sets need_weights=False, so the weights are never materialised.
    """
    with torch.no_grad():
        return block.cttn(query=grabbed["query"], key=grabbed["key"],
                          value=grabbed["value"], need_weights=True,
                          average_attn_weights=True)[1][0].float()


def per_query_cos(w1, w2):
    """How alike two attention patterns are, averaged over the 196 queries.

    Compared per query and then averaged, not flattened into one long vector:
    flattening would let a few loud queries speak for all of them.
    """
    return float(torch.nn.functional.cosine_similarity(w1, w2, dim=1).mean())


def overlay(ax, frame_img, heat, img_size, vmin, vmax):
    """A frame with a heat map painted over it, see-through where the heat is low.

    vmin/vmax are passed in rather than taken from `heat` so that a row of
    frames can share one scale -- otherwise each frame self-normalises and they
    stop being comparable, which is the whole point of showing them together.
    """
    ax.imshow(frame_img)
    a = (heat - vmin) / max(vmax - vmin, 1e-9)
    ax.imshow(heat, cmap="inferno", alpha=0.12 + 0.78 * a, interpolation="bilinear",
              extent=[0, img_size, img_size, 0], vmin=vmin, vmax=vmax)
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)


def main():
    # Every knob is checked here, before the dataset, the checkpoint or the GPU.
    cfg, sec, t_val, step_i = read_config()
    base = nwm.load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.get("seed", 0) or 0))

    # ---- 0. the scene, and what exactly we are asking for ----
    ds = build_recon_eval_dataset(base)
    row, f_curr, curr_time, scene_tag, mode = resolve_scene(ds, cfg, CONFIG)
    _, obs_image, gt_image, delta = ds[row]

    ts = sec * INPUT_FPS                      # generate_time(): eval_timesteps = sec*input_fps
    T = ds.context_size
    out_dir = f"debug/out/t4/{scene_tag}"
    save = images_io.Saver(out_dir)     # writes the PNGs and records the manifest

    hr("0) THE QUESTION WE ARE ASKING THE MODEL  (debug/t4/config.yaml)")
    print(f"  mode          : {mode}")
    print(f"  trajectory    : {f_curr}")
    print(f"  curr_time     : frame {curr_time}   ('now', the last context frame)")
    print(f"  horizon       : {sec} s ahead -> ts = {sec} x {INPUT_FPS} fps = {ts} dataset steps")
    print(f"  i.e. 'given frames {curr_time - T + 1}..{curr_time} and this action, "
          f"what does frame {curr_time + ts} look like?'")
    print(f"  device        : {device}")

    # ---- 1. the model ----
    hr("1) THE MODEL  (built exactly as isolated_nwm_infer.py:162)")
    latent_size = base["image_size"] // 8
    model_name = base["model"]
    model, ckpt = nwm.build_cdit(base, context_size=T, latent_size=latent_size, device=device)
    n_total = ckpt["params_total"]
    print(f"  {model_name}   (class {type(model).__name__}) -- NWM TRAINS THIS. It is the only")
    print(f"  part of the pipeline with learned weights of its own.")
    print(f"  depth (blocks)      : {len(model.blocks)}")
    print(f"  hidden size         : {model.blocks[0].attn.qkv.in_features}")
    print(f"  attention heads     : {model.num_heads}")
    print(f"  patch size          : {model.patch_size}")
    print(f"  input_size (latent) : {latent_size} x {latent_size}   (224 / 8, from T3)")
    print(f"  context_size        : {model.context_size} frames")
    print(f"  in / out channels   : {model.in_channels} -> {model.out_channels} "
          f"(learn_sigma={model.learn_sigma}: 4 noise + 4 variance)")
    print(f"  total parameters    : {n_total/1e6:.1f} M")

    print(f"\n  checkpoint          : {ckpt['path']}")
    print(f"  keys in the file    : {ckpt['keys_in_file']}")
    print(f"  load_state_dict     : {ckpt['load_msg']}   <- 'ema', the smoothed copy kept during training")
    train_step = ckpt["train_step"]
    print(f"  training step       : {train_step}")
    print(f"  (the real eval also wraps this in torch.compile + DDP -- speed only, same maths)")

    # where the parameters actually live
    hidden = model.blocks[0].attn.qkv.in_features
    groups = {
        "x_embedder (patchify)": model.x_embedder,
        "t_embedder (diffusion step)": model.t_embedder,
        "y_embedder (action)": model.y_embedder,
        "time_embedder (how far ahead)": model.time_embedder,
        "final_layer": model.final_layer,
    }
    part = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
    part["pos_embed (learned)"] = model.pos_embed.numel()
    blk = model.blocks[0]
    per_block = {
        "self-attention": sum(p.numel() for p in blk.attn.parameters()),
        "cross-attention": sum(p.numel() for p in blk.cttn.parameters()),
        "feed-forward (MLP)": sum(p.numel() for p in blk.mlp.parameters()),
        "adaLN conditioning": sum(p.numel() for p in blk.adaLN_modulation.parameters()),
    }
    n_blocks_total = sum(p.numel() for p in model.blocks.parameters())
    hr("1b) WHERE THE PARAMETERS ARE")
    print(f"  {'part':<32} {'params':>12}   share")
    for k, v in sorted(part.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v/1e6:>10.1f} M   {v/n_total*100:>5.1f}%")
    print(f"  {'the ' + str(len(model.blocks)) + ' CDiT blocks':<32} {n_blocks_total/1e6:>10.1f} M   "
          f"{n_blocks_total/n_total*100:>5.1f}%")
    print(f"\n  inside ONE block ({sum(per_block.values())/1e6:.1f} M, x{len(model.blocks)}):")
    for k, v in sorted(per_block.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<30} {v/1e6:>10.1f} M   {v/sum(per_block.values())*100:>5.1f}% of the block")
    print(f"\n  Note the biggest single piece: adaLN conditioning is one Linear")
    print(f"  {hidden} -> 11 x {hidden} = {11*hidden} (models.py:96). Steering the block")
    print(f"  costs more parameters than the attention that does the looking.")

    # ---- 2. the five inputs ----
    hr("2) THE FIVE THINGS forward() TAKES  (models.py:226)")
    vae = nwm.build_vae(device)
    ctx_px = obs_image[-T:].to(device)                       # (T, 3, 224, 224)
    x_cond, _ = nwm.context_latents(vae, obs_image, T, device)   # (1, T, 4, 28, 28), B=1

    x = torch.randn(1, 4, latent_size, latent_size, device=device)       # infer.py:81

    # The respaced schedule, exactly as isolated_nwm_infer.py:168 builds it. Its
    # timestep_map IS the set of t values the model can ever be asked about, so we
    # read it rather than assuming the stride is a round number -- it is not.
    # t_val was checked against this same map in read_config, before the checkpoint.
    diffusion = nwm.build_diffusion()
    t_map = list(diffusion.timestep_map)
    t = torch.full((1,), t_val, device=device, dtype=torch.float32)
    y = delta[:ts].sum(dim=0, keepdim=True).to(device).float()           # infer.py:111
    rel_t = torch.full((1,), ts / 128.0, device=device)                  # infer.py:75-76

    print("  x       the noisy latent being cleaned up -- pure noise on the first of")
    print(f"          the {RESPACED_STEPS} passes, partly cleaned on later ones (that loop is T5)")
    show("  x", x[0])
    gaps = sorted(set(t_map[i + 1] - t_map[i] for i in range(len(t_map) - 1)))
    print(f"\n  t       which diffusion step we are on. create_diffusion(str({RESPACED_STEPS}))")
    print(f"          respaces {DIFFUSION_STEPS} steps down to {RESPACED_STEPS}, so the model is only")
    print(f"          ever asked about these {len(t_map)} values:")
    print(f"            {t_map[:6]} ... {t_map[-3:]}")
    print(f"          Note the stride is {DIFFUSION_STEPS - 1}/{RESPACED_STEPS - 1} = "
          f"{(DIFFUSION_STEPS - 1) / (RESPACED_STEPS - 1):.3f}, not a round 4 -- gaps are "
          f"{gaps},")
    print(f"          and the top of the range is {t_map[-1]}, not {DIFFUSION_STEPS - 4}.")
    print(f"          The loop walks the list from {t_map[-1]} down to {t_map[0]}.")
    print(f"  t       = {t_val}   (index {t_map.index(t_val)} of {len(t_map)})")
    print("\n  y       the action: where the robot ends up after the whole horizon,")
    print("          delta[:ts].sum(0) -- one (dx, dy, dyaw), not a path (T2)")
    print(f"  y       = ({y[0,0]:+.4f}, {y[0,1]:+.4f}, {y[0,2]:+.4f})   "
          f"(normalized units, config normalize={base['normalize']})")
    print("\n  x_cond  the T3 latents of the 4 past frames, still separate")
    show("  x_cond", x_cond[0])
    print("\n  rel_t   how far ahead, as one scalar: ts/128")
    print(f"  rel_t   = {ts}/128 = {rel_t.item():.5f}")
    print(f"\n  NOTE y and rel_t both encode the horizon, differently: y says WHERE the")
    print(f"  robot goes, rel_t says HOW LONG it takes. Same {sec}s, two channels.")

    # ---- 3. patchify + position embeddings ----
    hr("3) PATCHIFY  ->  tokens  (models.py:233-235)")
    grid = latent_size // model.patch_size
    n_tok = grid * grid
    conv = model.x_embedder.proj
    print(f"  x_embedder is a Conv2d({conv.in_channels}, {conv.out_channels}, "
          f"kernel={tuple(conv.kernel_size)}, stride={tuple(conv.stride)})")
    print(f"  -> it slices the {latent_size}x{latent_size} latent into non-overlapping "
          f"{model.patch_size}x{model.patch_size} tiles:")
    print(f"     {grid} x {grid} = {n_tok} tiles, i.e. {n_tok} TOKENS.")
    print(f"  Each tile is {model.in_channels} x {model.patch_size} x {model.patch_size} = "
          f"{model.in_channels * model.patch_size ** 2} numbers, and the conv turns it into "
          f"{hidden}.")
    print(f"  That is an EXPANSION ({hidden // (model.in_channels * model.patch_size**2)}x): "
          f"the transformer wants room to think, not compression.")

    with torch.no_grad():
        tok_x = model.x_embedder(x)                                   # (1, 196, hidden)
        tok_c = model.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (1, T))   # (1, T, 196, hidden)
    print(f"\n  the frame being predicted : {tuple(x.shape)} -> {tuple(tok_x.shape)}")
    print(f"  the {T} context frames      : {tuple(x_cond.shape)} -> {tuple(tok_c.shape)}")
    print(f"  then .flatten(1,2)          : -> (1, {T} x {n_tok} = {T*n_tok}, {hidden})")
    print(f"  ONE sequence of {T*n_tok} context tokens. This is where the {T} frames finally merge.")

    pe = model.pos_embed                                              # (T+1, 196, hidden)
    print(f"\n  pos_embed {tuple(pe.shape)} -- {T+1} slots: one per context frame + one for")
    print(f"  the frame being predicted. It is a LEARNED nn.Parameter (requires_grad="
          f"{pe.requires_grad}),")
    print(f"  not the sin/cos table the comment at models.py:174 suggests -- "
          f"get_2d_sincos_pos_embed is defined in the file but never called.")
    print(f"  Without it, {T*n_tok} context tokens would be an unordered bag: the model could")
    print(f"  not tell frame t-3 from 'now', nor top-left from bottom-right.")

    pe_slot = pe.detach().flatten(1)                                  # (T+1, 196*hidden)
    pe_n = pe_slot / pe_slot.norm(dim=1, keepdim=True)
    slot_sim = (pe_n @ pe_n.T).cpu().numpy()
    print(f"\n  how different are the {T+1} slots, after training? (cosine similarity)")
    labels = [f"ctx {i}" if i < T else "predicted" for i in range(T + 1)]
    print("  " + " " * 11 + "".join(f"{l:>11}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:<11}" + "".join(f"{slot_sim[i, j]:>11.3f}" for j in range(T + 1)))
    off = slot_sim[~np.eye(T + 1, dtype=bool)]
    ctx_sim = slot_sim[:T, :T][~np.eye(T, dtype=bool)]
    pred_sim = slot_sim[T, :T]
    print(f"  Two things the model taught itself here:")
    print(f"    * neighbouring context slots are the most alike "
          f"(t-1 vs now {slot_sim[T-2, T-1]:+.3f}) and distant ones the least "
          f"({slot_sim[0, T-1]:+.3f}) -- it learned that these slots are a TIME ORDER,")
    print(f"      nobody told it. Context slots average {ctx_sim.mean():+.3f}.")
    print(f"    * the 'predicted' slot is near-orthogonal to all {T} context slots "
          f"({pred_sim.mean():+.3f}):")
    print(f"      'the frame I am inventing' is a different kind of thing from "
          f"'a frame I am reading'.")

    # spatial structure inside the predicted-frame slot
    q_idx, q_r, q_c = parse_query_patch(cfg.get("query_patch", "center"), grid)
    pred_slot = pe[T].detach()                                        # (196, hidden)
    ps_n = pred_slot / pred_slot.norm(dim=1, keepdim=True)
    spatial_sim = (ps_n @ ps_n[q_idx]).reshape(grid, grid).cpu().numpy()
    print(f"\n  and INSIDE one slot, is there spatial structure? similarity of patch "
          f"({q_r},{q_c}) to all {n_tok}:")
    print(f"    nearest-neighbour patches {np.mean([spatial_sim[max(q_r-1,0):q_r+2, max(q_c-1,0):q_c+2]]):+.3f}"
          f"   vs far corners {spatial_sim[[0,0,-1,-1],[0,-1,0,-1]].mean():+.3f}")

    # ---- 4. the conditioning vector ----
    hr("4) THE CONDITIONING VECTOR c  (models.py:236-239)")
    with torch.no_grad():
        t_emb = model.t_embedder(t[..., None])
        y_emb = model.y_embedder(y)
        time_emb = model.time_embedder(rel_t[..., None])
        c = t_emb + time_emb + y_emb
    print(f"  Three scalars-ish become three {hidden}-vectors, then are simply ADDED:")
    print(f"    t_embedder(t)          diffusion step {t_val:>4}  -> {tuple(t_emb.shape)}  "
          f"|v| = {t_emb.norm().item():7.2f}")
    print(f"    time_embedder(rel_t)   horizon  {rel_t.item():.5f}  -> {tuple(time_emb.shape)}  "
          f"|v| = {time_emb.norm().item():7.2f}")
    print(f"    y_embedder(y)          the action        -> {tuple(y_emb.shape)}  "
          f"|v| = {y_emb.norm().item():7.2f}")
    print(f"    c = sum                                  -> {tuple(c.shape)}  "
          f"|v| = {c.norm().item():7.2f}")
    print(f"\n  HOW a number becomes a vector (TimestepEmbedder, models.py:39-58):")
    print(f"    128 frequencies, from 1 down to 1/10000, are each fed the number;")
    print(f"    cos and sin of all of them -> 256 values -> a 2-layer MLP -> {hidden}.")
    print(f"    Slow frequencies say roughly-where, fast ones say exactly-where -- so")
    print(f"    nearby numbers get nearby vectors instead of one arbitrary vector each.")
    print(f"\n  The action gets THREE of those, one per component (models.py:65-77):")
    hs = hidden // 3
    print(f"    dx -> {hs} dims, dy -> {hs} dims, dyaw -> {hidden - 2*hs} dims, concatenated.")
    print(f"    So the model never has to disentangle x from y from yaw -- they arrive")
    print(f"    in separate slices of the vector.")

    # what each of the 11 adaLN numbers does, for the block we trace
    b_idx = int(cfg.get("block", 0) or 0)
    with torch.no_grad():
        chunks = model.blocks[b_idx].adaLN_modulation(c).chunk(11, dim=1)
    hr(f"4b) adaLN-ZERO: c BECOMES 11 CONTROL SIGNALS PER BLOCK  (block {b_idx})")
    print(f"  {'name':<18} {'|mean|':>9} {'std':>9}   what it does")
    for (nm, what), ch in zip(ADALN_NAMES, chunks):
        print(f"  {nm:<18} {ch.mean().item():>+9.4f} {ch.std().item():>9.4f}   {what}")
    print(f"\n  Each is a {hidden}-vector, applied per channel:  x * (1 + scale) + shift")
    print(f"  (models.py:18). 'gate' multiplies the whole sub-layer's output before it")
    print(f"  is added back -- gate 0 means 'this block does nothing this time'. They")
    print(f"  are initialised to exactly 0 (models.py:202) so training starts from the")
    print(f"  identity function and every block has to earn its influence.")

    # gates across depth -- proof the blocks learned to do different amounts
    gates = {"gate_msa": [], "gate_ca_x": [], "gate_mlp": []}
    gate_pos = {"gate_msa": 2, "gate_ca_x": 7, "gate_mlp": 10}
    with torch.no_grad():
        for blk_i in model.blocks:
            ch = blk_i.adaLN_modulation(c).chunk(11, dim=1)
            for k, p in gate_pos.items():
                gates[k].append(ch[p].abs().mean().item())
    print(f"\n  averaged |gate| over all {len(model.blocks)} blocks (0 would mean 'skip me'):")
    for k, v in gates.items():
        print(f"    {k:<11} min {min(v):.4f}  max {max(v):.4f}  mean {np.mean(v):.4f}")

    # ---- 5. one block, traced ----
    hr(f"5) ONE BLOCK, LAYER BY LAYER  (block {b_idx} of {len(model.blocks)}; all {len(model.blocks)} are identical in shape)")
    trace, cap = [], {}
    handles = []

    def rec(name):
        def hook(_m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            i0 = inp[0] if inp else None
            trace.append((name,
                          tuple(i0.shape) if torch.is_tensor(i0) else "-",
                          tuple(o.shape) if torch.is_tensor(o) else "-"))
        return hook

    blk = model.blocks[b_idx]
    for nm, mod in [("adaLN_modulation", blk.adaLN_modulation), ("norm1", blk.norm1),
                    ("attn (self)", blk.attn), ("norm2", blk.norm2), ("norm_cond", blk.norm_cond),
                    ("cttn (cross)", blk.cttn), ("norm3", blk.norm3), ("mlp", blk.mlp)]:
        handles.append(mod.register_forward_hook(rec(nm)))
    handles.append(model.final_layer.register_forward_hook(rec("final_layer")))

    def grab_self(_m, inp):
        cap["self_in"] = inp[0].detach()
    def grab_cross(_m, args, kwargs):
        cap["cross"] = {k: v.detach() for k, v in kwargs.items() if torch.is_tensor(v)}
    handles.append(blk.attn.register_forward_pre_hook(grab_self))
    handles.append(blk.cttn.register_forward_pre_hook(grab_cross, with_kwargs=True))

    with torch.no_grad():
        out = model(x, t, y, x_cond, rel_t)
    for h in handles:
        h.remove()

    print(f"  {'layer':<20} {'in':<22} {'out'}")
    for nm, i, o in trace:
        print(f"  {nm:<20} {str(i):<22} {o}")
    print(f"\n  Read it as: {n_tok} tokens go round once. Self-attention lets the {n_tok} patches")
    print(f"  of the future frame talk to EACH OTHER ({n_tok} x {n_tok}). Cross-attention lets each")
    print(f"  of them read the {T*n_tok} context tokens ({n_tok} x {T*n_tok}) -- one direction only:")
    print(f"  the context is never rewritten. The MLP then thinks per-token. x is the same")
    print(f"  shape at every step because each sub-layer is ADDED to it (a residual stream).")

    # ---- 5b. real attention weights, all 28 blocks, two different inputs ----
    hr("5b) WHERE DOES IT ACTUALLY LOOK?  (real cross-attention weights, every block)")
    n_ctx_tok = T * n_tok

    def attention_profile(x_in, t_in, xc=None):
        """Every block's cross-attention for one input. Returns (per-block stats, captures)."""
        xc = x_cond if xc is None else xc
        grabbed, handles = {}, []

        def make_grab(i):
            def hook(_m, args, kwargs):
                grabbed.setdefault(i, {}).update(
                    {k: v.detach() for k, v in kwargs.items() if torch.is_tensor(v)})
            return hook

        def make_grab_self(i):
            def hook(_m, inp):
                grabbed.setdefault(i, {})["self_in"] = inp[0].detach()
            return hook

        for i, b in enumerate(model.blocks):
            handles.append(b.cttn.register_forward_pre_hook(make_grab(i), with_kwargs=True))
            handles.append(b.attn.register_forward_pre_hook(make_grab_self(i)))
        with torch.no_grad():
            model(x_in, t_in, y, xc, rel_t)
        for h in handles:
            h.remove()

        stats = []
        with torch.no_grad():
            for i, b in enumerate(model.blocks):
                g = grabbed[i]
                w = b.cttn(query=g["query"], key=g["key"], value=g["value"],
                           need_weights=True, average_attn_weights=True)[1][0].float()
                ctx_w = w[:, :n_ctx_tok]
                # does a query look at ITS OWN corner of the past frames? attention-
                # weighted centre of mass in the 'now' frame vs the query's own position
                now_w = ctx_w.reshape(n_tok, T, n_tok)[:, T - 1, :]
                now_w = now_w / now_w.sum(1, keepdim=True).clamp_min(1e-9)
                cr, cc_ = now_w @ pat_r, now_w @ pat_c
                stats.append({
                    "block": i,
                    "per_frame": ctx_w.reshape(n_tok, T, n_tok).sum(2).mean(0).cpu().numpy(),
                    # how focused: share of a query's attention on its best 10 of 784
                    "top10": float(ctx_w.topk(10, dim=-1).values.sum(-1).mean()),
                    "bias": float(w[:, n_ctx_tok:].sum(-1).mean()),
                    "offset": float(torch.sqrt((cr - pat_r) ** 2 + (cc_ - pat_c) ** 2).mean()),
                })
        return stats, grabbed

    # patch coordinates, and what "looking nowhere in particular" would score
    pat_r = torch.arange(grid, device=device).repeat_interleave(grid).float()
    pat_c = torch.arange(grid, device=device).repeat(grid).float()
    mid = (grid - 1) / 2
    uniform_offset = float(torch.sqrt((pat_r - mid) ** 2 + (pat_c - mid) ** 2).mean())

    # (a) the loop's FIRST call: x is pure noise, nothing to go on
    depth_noise, grabbed_noise = attention_profile(x, t)

    # (b) a realistic LATE call. p_sample_loop's x at step i is a partly-cleaned
    # latent, which we cannot produce without running the loop (T5). We can get a
    # stand-in with one formula from the same schedule: take the TRUE future
    # latent and add exactly as much noise as step i carries. That is q_sample --
    # the forward process, no loop, no model.
    t_late = int(t_map[step_i])          # step_i checked in read_config
    with torch.no_grad():
        z_gt = nwm.encode(vae, gt_image[ts - 1:ts].to(device))
        x_late = diffusion.q_sample(z_gt, torch.tensor([step_i], device=device))
    t_late_t = torch.full((1,), t_late, device=device, dtype=torch.float32)
    depth_late, grabbed = attention_profile(x_late, t_late_t)
    uniform_top10 = 10 / n_ctx_tok

    print(f"  Two measurements per block:")
    print(f"    focus  -- share of a query's attention on its best 10 of the {n_ctx_tok} context")
    print(f"              tokens. Spread evenly that would be {uniform_top10*100:.1f}%.")
    print(f"    offset -- how far, in patches, a query's attention centre in the 'now' frame")
    print(f"              sits from the query's OWN position. Looking nowhere in particular")
    print(f"              scores {uniform_offset:.2f}; looking at the matching place scores ~0.")
    print(f"  ...at two inputs, because that is what makes the difference:")
    print(f"    (a) t={t_val}, x = pure noise      -- the loop's FIRST of {RESPACED_STEPS} calls")
    print(f"    (b) t={t_late}, x = the true future latent re-noised to step {step_i}")
    print(f"        (q_sample, one formula from the schedule -- the loop itself is T5)\n")
    print(f"  {'':>4} | {'--- (a) pure noise ----':^22} | {'--- (b) near the end --':^22}")
    print(f"  {'blk':>4} | {'focus':>6} {'now':>6} {'offset':>7} | {'focus':>6} {'now':>6} {'offset':>7}")
    for da, db in zip(depth_noise, depth_late):
        print(f"  {da['block']:>4} | {da['top10']*100:>5.1f}% {da['per_frame'][T-1]*100:>5.1f}% "
              f"{da['offset']:>7.2f} | {db['top10']*100:>5.1f}% {db['per_frame'][T-1]*100:>5.1f}% "
              f"{db['offset']:>7.2f}")

    depth = depth_late
    want_ab = cfg.get("attn_block", "auto")
    if want_ab in (None, "", "auto"):
        a_idx = max(depth, key=lambda d: d["top10"])["block"]
        why = "auto: the most focused of the 28, on input (b)"
    else:
        a_idx = int(want_ab)
        why = "from debug/t4/config.yaml"

    fa = np.mean([d["top10"] for d in depth_noise]) * 100
    fb = np.mean([d["top10"] for d in depth_late]) * 100
    oa = np.mean([d["offset"] for d in depth_noise])
    ob = np.mean([d["offset"] for d in depth_late])
    print(f"\n  averaged over the {len(model.blocks)} blocks:")
    print(f"    focus   (a) {fa:.1f}%   (b) {fb:.1f}%    (evenly spread: {uniform_top10*100:.1f}%)")
    print(f"    offset  (a) {oa:.2f}    (b) {ob:.2f}     (looking nowhere: {uniform_offset:.2f})")

    # Neither number moved much between (a) and (b). Is that because the attention
    # pattern does not depend on the CONTENT of the context at all? Blank the
    # context latents and compare the weights query by query.
    _, grabbed_noc = attention_profile(x_late, t_late_t, torch.zeros_like(x_cond))

    sim_ab, sim_noc = [], []
    for i in range(len(model.blocks)):
        wl = cross_attn_weights(model.blocks[i], grabbed[i])
        sim_ab.append(per_query_cos(
            cross_attn_weights(model.blocks[i], grabbed_noise[i]), wl))
        sim_noc.append(per_query_cos(
            cross_attn_weights(model.blocks[i], grabbed_noc[i]), wl))
    del grabbed_noise, grabbed_noc

    print(f"\n  Neither number moved much. Is the attention pattern reacting to the CONTENT")
    print(f"  of the context at all? Two comparisons, query by query (cosine over the {n_ctx_tok+1}")
    print(f"  weights, averaged over the {n_tok} queries and the {len(model.blocks)} blocks):")
    print(f"    (a) vs (b)  -- totally different x            : {np.mean(sim_ab):.3f}")
    print(f"    (b) vs x_cond BLANKED -- context deleted      : {np.mean(sim_noc):.3f}")
    print(f"    worst block for the blanked test              : {min(sim_noc):.3f} "
          f"(block {int(np.argmin(sim_noc))})")
    print(f"\n  So: CDiT's cross-attention is close to a fixed, position-driven blend.")
    print(f"  A query reads a broad mixture of all {n_ctx_tok} context tokens -- {fb:.1f}% on its best")
    print(f"  10, and its centre of mass sits {ob:.2f} patches from its own position when")
    print(f"  {uniform_offset:.2f} would mean 'no spatial preference whatsoever'. It is NOT tracking a")
    print(f"  bit of scenery from a past frame into the future one.")
    print(f"  The {T} frames get a near-even {100/T:.0f}/{100/T:.0f}/{100/T:.0f}/{100/T:.0f} split with a "
          f"lean toward 'now' in the")
    print(f"  early blocks -- all four are read every time. What the attention DOES is")
    print(f"  hand every query a summary of the whole scene; deciding what to do with it")
    print(f"  is the MLP's and the conditioning's job (that is where the parameters are).")
    print(f"\n  The pictures below use input (b), block {a_idx}")
    print(f"  ({why}) -- the one block that is")
    print(f"  meaningfully more selective than the rest.")

    blk_a = model.blocks[a_idx]
    with torch.no_grad():
        sw = self_attn_weights(blk_a.attn, grabbed[a_idx]["self_in"])      # (1, heads, 196, 196)
        g = grabbed[a_idx]
        cw = blk_a.cttn(query=g["query"], key=g["key"], value=g["value"],
                        need_weights=True, average_attn_weights=True)[1]   # (1, 196, 784[+1])
    self_map = sw[0].mean(0)[q_idx].reshape(grid, grid).float().cpu().numpy()
    cw0 = cw[0, q_idx].float().cpu()
    bias_mass = float(cw0[n_ctx_tok:].sum()) if cw0.numel() > n_ctx_tok else 0.0
    cross_map = cw0[:n_ctx_tok].reshape(T, grid, grid).numpy()
    per_frame = cross_map.reshape(T, -1).sum(1)
    per_frame = per_frame / max(per_frame.sum(), 1e-9)     # of the context, ignoring the sink

    print(f"\n  ---- block {a_idx}, one query ----")
    print(f"  query = patch ({q_r}, {q_c}) of the {grid}x{grid} grid, i.e. token {q_idx} of the")
    print(f"  FUTURE frame -- a patch we are trying to invent.")
    print(f"  cross-attention weights {tuple(cw.shape)}"
          + (f"  = {n_ctx_tok} context tokens + {cw.shape[-1]-n_ctx_tok} sink"
             if cw.shape[-1] > n_ctx_tok else ""))
    print(f"  a softmax: the {cw.shape[-1]} weights sum to {float(cw0.sum()):.3f} "
          f"({bias_mass*100:.1f}% of it on the sink).")
    print(f"\n  how the rest splits over the {T} context frames:")
    for i in range(T):
        tag = "now" if i == T - 1 else f"t-{T-1-i}"
        bar = "#" * int(round(per_frame[i] / max(per_frame.max(), 1e-9) * 40))
        print(f"    context frame {i} ({tag:>4}) : {per_frame[i]*100:5.1f}%  {bar}")
    print(f"\n  peak patch inside each frame (is it looking at the SAME PLACE?):")
    for i in range(T):
        pr, pc = np.unravel_index(cross_map[i].argmax(), (grid, grid))
        print(f"    frame {i}: brightest at ({pr}, {pc})   query is at ({q_r}, {q_c})")
    print(f"\n  self-attention: this patch's own top neighbours in the future frame:")
    top = np.dstack(np.unravel_index(np.argsort(self_map, axis=None)[::-1][:5], (grid, grid)))[0]
    print(f"    {[f'({r},{c})' for r, c in top]}   (query ({q_r},{q_c}))")
    del grabbed

    # ---- 6. the output ----
    hr("6) WHAT COMES OUT  (models.py:243-244)")
    print(f"  final_layer : ({n_tok}, {hidden}) -> ({n_tok}, "
          f"{model.patch_size}*{model.patch_size}*{model.out_channels} = "
          f"{model.patch_size**2 * model.out_channels})")
    print(f"  unpatchify  : {n_tok} tiles reassembled -> {tuple(out.shape)}")
    eps, sigma = out[:, :4], out[:, 4:]
    show("  eps", eps[0])
    show("  sigma", sigma[0])
    print(f"\n  The first 4 channels are the model's guess at the NOISE currently sitting")
    print(f"  in x (ModelMeanType.EPSILON). The other 4 are how sure it is")
    print(f"  (ModelVarType.LEARNED_RANGE, because learn_sigma=True).")
    print(f"  It never outputs a picture. Subtracting a little of eps from x is one")
    print(f"  denoising step -- doing that {RESPACED_STEPS} times is T5.")

    # ---- 7. is the conditioning load-bearing? ----
    hr("7) ABLATIONS -- does the conditioning actually steer anything?")
    # Most of what CDiT outputs is a copy of the noise it was handed: it is asked
    # for eps, and at high t almost all of x IS eps. So ||eps|| is a misleading
    # yardstick. The part that is a guess about THIS scene is eps - x.
    echo = float(torch.nn.functional.cosine_similarity(eps.flatten(), x.flatten(), dim=0))
    base_norm, guess_norm = eps.norm().item(), (eps - x).norm().item()
    print(f"  First, the scale of things. At t={t_val} the output is almost exactly the")
    print(f"  input handed back: cosine(eps, x) = {echo:.3f}, and only "
          f"{guess_norm/base_norm*100:.1f}% of ||eps|| is")
    print(f"  the part that says anything about THIS scene. So we measure every change")
    print(f"  against eps - x, that guess, not against the whole output.\n")

    # "ask for a different horizon" has to stay inside the range the model was
    # trained on, whatever `sec` the config picked. So compare against the far end
    # of the real eval's sweep: 1 s if we are asking for more, 16 s if we are at 1 s.
    alt_sec = 1 if sec != 1 else max(SECS_SWEPT)
    rel_t_alt = torch.full_like(rel_t, alt_sec * INPUT_FPS / 128.0)

    def ablate(x_in, t_in):
        with torch.no_grad():
            e0 = model(x_in, t_in, y, x_cond, rel_t)[:, :4]
            g0 = (e0 - x_in).norm().item()
            outs = {
                "y = 0": model(x_in, t_in, torch.zeros_like(y), x_cond, rel_t)[:, :4],
                "x_cond.flip(1)": model(x_in, t_in, y, x_cond.flip(1), rel_t)[:, :4],
                "no motion history": model(x_in, t_in, y,
                                           x_cond[:, -1:].expand(-1, T, -1, -1, -1).contiguous(),
                                           rel_t)[:, :4],
                "different horizon": model(x_in, t_in, y, x_cond, rel_t_alt)[:, :4],
            }
        return {k: (v - e0).norm().item() / g0 * 100 for k, v in outs.items()}

    ab_a = ablate(x, t)
    ab_b = ablate(x_late, t_late_t)
    rows = [
        ("action zeroed out", "y = 0"),
        ("context frames in reverse order", "x_cond.flip(1)"),
        (f"all {T} context frames = 'now'", "no motion history"),
        (f"ask for {alt_sec}s instead of {sec}s", "different horizon"),
    ]
    print(f"  change one input, keep the rest -- how far does the guess move?\n")
    print(f"  {'changed':<36} {'how':<20} {'(a) t='+str(t_val):>10} {'(b) t='+str(t_late):>10}")
    abl = []
    for nm, how in rows:
        print(f"  {nm:<36} {how:<20} {ab_a[how]:>9.1f}% {ab_b[how]:>9.1f}%")
        abl.append((nm, how, ab_a[how], ab_b[how]))
    with torch.no_grad():
        other = model(torch.randn_like(x), t, y, x_cond, rel_t)[:, :4]
    print(f"\n  For scale: handing it a DIFFERENT random noise moves the whole output by")
    print(f"  {(other - eps).norm().item()/base_norm*100:.0f}% of ||eps||. The noise decides most of what "
          f"comes out; the")
    print(f"  conditioning is what makes it a plausible FUTURE OF THIS SCENE rather than")
    print(f"  a plausible frame in general. That is exactly the job it has.")
    print(f"\n  Reversing the {T} context frames moves the answer at all only because of the")
    print(f"  per-slot position embedding from step 3 -- without it the {n_ctx_tok} context tokens")
    print(f"  would be an unordered bag and that row would read 0.0%.")

    # ...and how that changes as the loop progresses, on REALISTIC inputs
    sweep_steps = [249, 200, 150, 100, 50, 25, 5]
    print(f"\n  Does the steering grow as the frame emerges? Same question at "
          f"{len(sweep_steps)} points of")
    print(f"  the loop, each with a realistic x for that point (q_sample of the true")
    print(f"  future latent, as above):\n")
    print(f"  {'step':>5} {'t':>5} {'echo cos(eps,x)':>16} {'guess size':>11} "
          f"{'action zeroed':>14}")
    sweep = []
    with torch.no_grad():
        for si in sweep_steps:
            tv = int(t_map[si])
            tt = torch.full((1,), tv, device=device, dtype=torch.float32)
            xi = diffusion.q_sample(z_gt, torch.tensor([si], device=device))
            e0 = model(xi, tt, y, x_cond, rel_t)[:, :4]
            e1 = model(xi, tt, torch.zeros_like(y), x_cond, rel_t)[:, :4]
            g0 = (e0 - xi).norm().item()
            row = {
                "step": si, "t": tv,
                "echo": round(float(torch.nn.functional.cosine_similarity(
                    e0.flatten(), xi.flatten(), dim=0)), 3),
                "guess_frac": round(g0 / e0.norm().item() * 100, 1),
                "action_pct_of_guess": round((e1 - e0).norm().item() / g0 * 100, 2),
            }
            sweep.append(row)
            print(f"  {si:>5} {tv:>5} {row['echo']:>16.3f} {row['guess_frac']:>10.1f}% "
                  f"{row['action_pct_of_guess']:>13.1f}%")
    peak = max(sweep, key=lambda s: s["action_pct_of_guess"])
    print(f"\n  (step {RESPACED_STEPS - 1} is the start of the loop, step 0 is the finished frame,")
    print(f"  so read the table top to bottom.) The action's influence is a HUMP, not a")
    print(f"  ramp: it peaks at step {peak['step']} ({peak['action_pct_of_guess']:.0f}%) and falls away "
          f"to {sweep[-1]['action_pct_of_guess']:.0f}% at the end.")
    print(f"  In the middle of the loop the layout of the scene is still up for grabs and")
    print(f"  the action is what settles it. By the end the frame is decided and the model")
    print(f"  is only sharpening texture, which the action has nothing to say about.")
    print(f"  Conditioning is not one big push -- it is {RESPACED_STEPS} small ones, and the ones")
    print(f"  that matter are in the middle.")

    # ---- 8. visuals ----
    hr("8) VISUALS  ->  " + out_dir)
    ctx_v = misc.unnormalize(ctx_px.cpu()).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    target = misc.unnormalize(gt_image[ts - 1]).clamp(0, 1).permute(1, 2, 0).numpy()
    IMG = latent_size * 8                   # 224, the frame size the VAE decodes to
    px = IMG // grid                        # pixels one tile covers, for overlays

    # --- figure 1: attention ---
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.85], hspace=0.3, wspace=0.18)
    for i in range(T):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        overlay(ax, ctx_v[i], cross_map[i], IMG,
                cross_map.min(), cross_map.max())
        tag = "now" if i == T - 1 else f"t-{T-1-i}"
        ax.set_title(f"context frame {i} ({tag})\n{per_frame[i]*100:.1f}% of the attention",
                     fontsize=10)
    ax = fig.add_subplot(gs[0, 4]); ax.axis("off")
    ax.imshow(target)
    ax.add_patch(plt.Rectangle((q_c * px, q_r * px), px, px, fill=False, ec="#00e5d0", lw=2.5))
    ax.set_title(f"the future frame (held out)\nthe patch we asked about: ({q_r}, {q_c})", fontsize=10)

    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    ax.imshow(self_map, cmap="viridis")
    ax.add_patch(plt.Rectangle((q_c - .5, q_r - .5), 1, 1, fill=False, ec="#00e5d0", lw=2))
    ax.set_title(f"self-attention\nwhere patch ({q_r},{q_c}) looks in its OWN frame", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(T), per_frame * 100, color="#0f8f8b")
    ax.set_xticks(range(T)); ax.set_xticklabels([f"f{i}" if i < T-1 else "now" for i in range(T)])
    ax.set_title("% of attention per context frame", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot([d["top10"] * 100 for d in depth_late], color="#0f8f8b", lw=1.8,
            label=f"near the end (t={t_late})")
    ax.plot([d["top10"] * 100 for d in depth_noise], color="#b06d13", lw=1.6,
            label=f"pure noise (t={t_val})")
    ax.axhline(uniform_top10 * 100, color="#5c6a76", ls=":", lw=1.2, label="if spread evenly")
    ax.axvline(a_idx, color="#00b3a4", lw=1, alpha=.5)
    ax.set_xlabel(f"block (0..{len(model.blocks)-1})")
    ax.set_ylabel("focus: % on best 10 tokens")
    ax.set_title("how focused the attention is", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 4]); ax.axis("off")
    im = ax.imshow(slot_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"pos_embed: are the {T+1} slots different?\n(cosine similarity)", fontsize=10)
    for i in range(T + 1):
        for j in range(T + 1):
            ax.text(j, i, f"{slot_sim[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)

    fig.suptitle(
        f"T4 -- CDiT {model_name}, checkpoint {CKP} (ema)  |  traj '{f_curr}' frame {curr_time}, "
        f"predicting {sec}s ahead  ·  attention read off block {a_idx} at step {step_i} of "
        f"{RESPACED_STEPS} (t={t_late})\n"
        f"TOP: one patch of the FUTURE frame, and where it reads the {T} past frames     "
        f"BOTTOM: its own frame, the split, how focused it is, frame-slot identity",
        fontsize=12)
    p1 = save.figure("cdit_attention", fig, dpi=120, bbox_inches="tight")
    print(f"  wrote {p1}")

    # --- figure 2: the inputs and the output ---
    fig = plt.figure(figsize=(16.5, 10.4))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.9], hspace=0.42, wspace=0.28)
    xn = x[0].float().cpu().numpy()
    en = eps[0].float().cpu().numpy()
    for ch in range(4):
        ax = fig.add_subplot(gs[0, ch]); ax.axis("off")
        ax.imshow(xn[ch], cmap="magma")
        ax.set_title(f"x, channel {ch}\n(what goes in: noise)", fontsize=9)
        ax = fig.add_subplot(gs[1, ch]); ax.axis("off")
        ax.imshow(en[ch], cmap="magma")
        ax.set_title(f"eps, channel {ch}\n(the noise it says is there)", fontsize=9)
    ax = fig.add_subplot(gs[0, 4:]); ax.axis("off")
    ax.text(0.0, 0.98,
            f"forward(x, t, y, x_cond, rel_t)\n\n"
            f"  x       (1, 4, {latent_size}, {latent_size})   noisy latent\n"
            f"  t       ({t_val})            diffusion step\n"
            f"  y       ({y[0,0]:+.3f}, {y[0,1]:+.3f}, {y[0,2]:+.3f})  action\n"
            f"  x_cond  (1, {T}, 4, {latent_size}, {latent_size}) past frames\n"
            f"  rel_t   ({rel_t.item():.5f})       horizon\n\n"
            f"  -> patchify   {n_tok} + {T}x{n_tok} = {n_tok + T*n_tok} tokens\n"
            f"  -> c = t + rel_t + y      ({hidden},)\n"
            f"  -> {len(model.blocks)} blocks x (self, cross, mlp)\n"
            f"  -> ({n_tok}, {model.patch_size**2*model.out_channels}) -> "
            f"(1, {model.out_channels}, {latent_size}, {latent_size})\n"
            f"     = 4 noise + 4 variance",
            va="top", ha="left", fontsize=11, family="monospace", transform=ax.transAxes)
    ax = fig.add_subplot(gs[1, 4:])
    short = {"y = 0": "action zeroed", "x_cond.flip(1)": "frames reversed",
             "no motion history": "no motion history",
             "different horizon": f"horizon {alt_sec}s"}
    names = [short[a[1]] for a in abl][::-1]
    ax.barh(names, [a[3] for a in abl][::-1], color="#0f8f8b")
    ax.set_xlabel("% change, measured against eps - x")
    ax.set_title(f"change one input, watch the answer move (t={t_late})", fontsize=10)
    ax.tick_params(labelsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[2, 0:3])
    for k, col in zip(gates, ["#0f8f8b", "#b06d13", "#5c6a76"]):
        ax.plot(gates[k], label=k, color=col, lw=1.8)
    ax.set_xlabel(f"block (0..{len(model.blocks)-1})"); ax.set_ylabel("mean |gate|")
    ax.set_title("how much each block chooses to do (adaLN gates)", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[2, 3:])
    ax.plot([s["t"] for s in sweep], [s["action_pct_of_guess"] for s in sweep],
            "o-", color="#0f8f8b", lw=1.8)
    ax.invert_xaxis()
    ax.set_xlabel(f"diffusion step t   (the loop runs right to left: {t_map[-1]} -> {t_map[0]})")
    ax.set_ylabel("% change if the action is zeroed")
    ax.set_title("the action matters most in the MIDDLE of the loop", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"T4 -- what CDiT is handed and what it hands back  |  {model_name}, "
                 f"{n_total/1e6:.0f} M params, checkpoint {CKP}", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.06)
    p2 = save.figure("cdit_io", fig, dpi=120, bbox_inches="tight")
    print(f"  wrote {p2}")

    # --- individual panels for build_page.py ---
    for i in range(T):
        save.image(f"ctx_f{i}", ctx_v[i])
        fig, ax = plt.subplots(figsize=(2.4, 2.4)); ax.axis("off")
        overlay(ax, ctx_v[i], cross_map[i], IMG,
                cross_map.min(), cross_map.max())
        fig.subplots_adjust(0, 0, 1, 1)
        save.figure(f"attn_f{i}", fig, dpi=110, bbox_inches="tight", pad_inches=0)
    save.image("target", target)
    ctx_l = x_cond[0].float().cpu().numpy()          # (T, 4, 28, 28)
    for i in range(T):
        for ch in range(4):
            save.image(f"ctx_f{i}_ch{ch}", ctx_l[i, ch], cmap="viridis")
    for ch in range(4):
        save.image(f"noise_ch{ch}", xn[ch], cmap="magma")
        save.image(f"eps_ch{ch}", en[ch], cmap="magma")

    # the late-loop latent, and what it looks like as a picture -- the clearest
    # single image of "the frame is being uncovered", decoded with T3's VAE
    xl = x_late[0].float().cpu().numpy()
    with torch.no_grad():
        xl_px = torch.clip(vae.decode(x_late / SCALING).sample, -1, 1)
    xl_v = misc.unnormalize(xl_px[0].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    save.image("xlate_decoded", xl_v)
    for ch in range(4):
        save.image(f"xlate_ch{ch}", xl[ch], cmap="magma")

    # ---- 8b. the three "make it visible" panels ----
    # Everything above renders measurements. These three render the THINGS, so
    # "196 tokens", "250 steps" and "the action steers it" stop being numbers.

    # (i) what one token actually covers, drawn on a real frame
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); ax.axis("off")
    ax.imshow(ctx_v[T - 1], extent=[0, IMG, IMG, 0])
    for k in range(1, grid):
        ax.axhline(k * px, color="w", lw=0.5, alpha=0.55)
        ax.axvline(k * px, color="w", lw=0.5, alpha=0.55)
    ax.add_patch(plt.Rectangle((q_c * px, q_r * px), px, px, fill=False, ec="#00e5d0", lw=3))
    ax.set_xlim(0, IMG); ax.set_ylim(IMG, 0)
    fig.subplots_adjust(0, 0, 1, 1)
    save.figure("tiles_on_frame", fig, dpi=130, bbox_inches="tight", pad_inches=0)

    # (i-b) the SAME cut, on the thing it is actually cut from. The grid is drawn
    # on the photo above only because 14x14 squares are legible there; the tiles
    # are really cut out of the latent, which is what this panel shows.
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); ax.axis("off")
    ax.imshow(ctx_l[T - 1, 0], cmap="viridis", interpolation="nearest")
    for k in range(1, grid):
        ax.axhline(-0.5 + k * model.patch_size, color="w", lw=0.6, alpha=0.6)
        ax.axvline(-0.5 + k * model.patch_size, color="w", lw=0.6, alpha=0.6)
    ax.add_patch(plt.Rectangle((-0.5 + q_c * model.patch_size, -0.5 + q_r * model.patch_size),
                               model.patch_size, model.patch_size,
                               fill=False, ec="#00e5d0", lw=3))
    fig.subplots_adjust(0, 0, 1, 1)
    save.figure("tiles_on_latent", fig, dpi=130, bbox_inches="tight", pad_inches=0)

    # (ii) the loop, as a filmstrip. p_sample_loop's real x we cannot make without
    # running it (T5), but q_sample gives the TRUE latent carrying exactly the noise
    # each step carries -- i.e. what the model is trying to reach at that point.
    strip_steps = [int(s) for s in (cfg.get("filmstrip_steps")
                                    or [249, 200, 150, 100, 50, 25, 10, 0])]
    strip = []
    with torch.no_grad():
        for si in strip_steps:
            xs = diffusion.q_sample(z_gt, torch.tensor([si], device=device))
            ps = torch.clip(vae.decode(xs / SCALING).sample, -1, 1)
            v = misc.unnormalize(ps[0].cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
            strip.append((si, int(t_map[si]), v))
            save.image(f"loop_s{si}", v)

    fig, axes = plt.subplots(1, len(strip), figsize=(2.05 * len(strip), 2.5))
    for ax, (si, tv, v) in zip(np.atleast_1d(axes), strip):
        ax.axis("off"); ax.imshow(v)
        ax.set_title(f"step {si}\nt = {tv}", fontsize=9)
    fig.suptitle(f"How much noise is left at each point of the {RESPACED_STEPS}-step loop "
                 f"(the loop runs left to right)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save.figure("loop_filmstrip", fig, dpi=120, bbox_inches="tight")

    # (iii) WHERE in the frame the action changes the answer. Not a percentage --
    # a map. Same x, same context, different action; look at |delta eps|.
    y_mirror = y * torch.tensor([[1.0, -1.0, -1.0]], device=device)   # steer the other way
    with torch.no_grad():
        e_real = model(x_late, t_late_t, y, x_cond, rel_t)[:, :4]
        e_zero = model(x_late, t_late_t, torch.zeros_like(y), x_cond, rel_t)[:, :4]
        e_mirr = model(x_late, t_late_t, y_mirror, x_cond, rel_t)[:, :4]
    d_zero = (e_real - e_zero)[0].abs().mean(0).float().cpu().numpy()
    d_mirr = (e_real - e_mirr)[0].abs().mean(0).float().cpu().numpy()
    vmax = max(d_zero.max(), d_mirr.max())

    # Painted on the TARGET frame, not on a context frame. The delta is a difference in
    # the model's answer ABOUT THE FUTURE FRAME, so the future frame is the only honest
    # backdrop -- overlaying it on the last past frame put two different moments in one
    # picture, which is what made it unreadable.
    for nm, dmap in (("actiondelta_zero", d_zero), ("actiondelta_mirror", d_mirr)):
        fig, ax = plt.subplots(figsize=(2.6, 2.6)); ax.axis("off")
        ax.imshow(target, extent=[0, IMG, IMG, 0])
        # Normalise each map against ITS OWN max, and make the bottom 40% fully
        # transparent. The previous version used a shared vmax and an alpha floor of
        # 0.15, which tinted the entire frame a flat purple and buried the signal --
        # the picture read as "nothing". Now: invisible where the action changed
        # little, strongly coloured where it changed a lot.
        a = dmap / max(dmap.max(), 1e-9)
        a = np.clip((a - 0.40) / 0.60, 0.0, 1.0) ** 0.75
        ax.imshow(dmap, cmap="inferno", alpha=0.90 * a, interpolation="bilinear",
                  extent=[0, IMG, IMG, 0], vmin=0, vmax=dmap.max())
        ax.set_xlim(0, IMG); ax.set_ylim(IMG, 0)
        fig.subplots_adjust(0, 0, 1, 1)
        save.figure(nm, fig, dpi=120, bbox_inches="tight", pad_inches=0)
    print(f"  wrote tiles_on_frame.png, tiles_on_latent.png, loop_filmstrip.png + loop_s*.png, "
          f"actiondelta_zero.png, actiondelta_mirror.png")

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for k, col in zip(gates, ["#0f8f8b", "#b06d13", "#5c6a76"]):
        ax.plot(gates[k], label=k, color=col, lw=1.6)
    ax.set_xlabel("block"); ax.set_ylabel("mean |gate|"); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save.figure("gates", fig, dpi=130)

    fig, ax = plt.subplots(figsize=(3.0, 2.6)); ax.axis("off")
    ax.imshow(slot_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(T + 1):
        for j in range(T + 1):
            ax.text(j, i, f"{slot_sim[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout(); save.figure("posembed", fig, dpi=130, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot([d["top10"] * 100 for d in depth_late], color="#0f8f8b", lw=1.6, label="near the end")
    ax.plot([d["top10"] * 100 for d in depth_noise], color="#b06d13", lw=1.4, label="pure noise")
    ax.axhline(uniform_top10 * 100, color="#5c6a76", ls=":", lw=1.1, label="even")
    ax.set_xlabel("block"); ax.set_ylabel("focus, % of attention"); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save.figure("focus", fig, dpi=130)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot([s["t"] for s in sweep], [s["action_pct_of_guess"] for s in sweep],
            "o-", color="#0f8f8b", lw=1.6)
    ax.invert_xaxis()
    ax.set_xlabel(f"diffusion step t  ({t_map[-1]} → {t_map[0]})"); ax.set_ylabel("% change, action zeroed")
    ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save.figure("tsweep", fig, dpi=130)

    fig, ax = plt.subplots(figsize=(2.4, 2.4)); ax.axis("off")
    ax.imshow(self_map, cmap="viridis")
    ax.add_patch(plt.Rectangle((q_c - .5, q_r - .5), 1, 1, fill=False, ec="#00e5d0", lw=2))
    fig.subplots_adjust(0, 0, 1, 1)
    save.figure("selfattn", fig, dpi=110, bbox_inches="tight", pad_inches=0)
    print(f"  wrote ctx_f*.png, attn_f*.png, target.png, noise_ch*.png, eps_ch*.png, "
          f"xlate_ch*.png, xlate_decoded.png, gates.png, focus.png, tsweep.png, "
          f"posembed.png, selfattn.png")

    # ---- 9. machine-readable facts ----
    facts = {
        "trajectory": f_curr,
        "curr_time": int(curr_time),
        "context_size": int(T),
        "context_frame_numbers": [int(curr_time - (T - 1) + i) for i in range(T)],
        "sec": sec,
        "input_fps": INPUT_FPS,
        "ts": int(ts),
        "target_frame_number": int(curr_time + ts),
        "model": model_name,
        "model_class": type(model).__name__,
        "checkpoint": CKP,
        "checkpoint_key": "ema",
        "train_step": train_step if isinstance(train_step, int) else str(train_step),
        "trained_by_nwm": True,
        "params_total_m": round(n_total / 1e6, 1),
        "depth": len(model.blocks),
        "hidden": int(hidden),
        "num_heads": int(model.num_heads),
        "patch_size": int(model.patch_size),
        "latent_size": int(latent_size),
        "in_channels": int(model.in_channels),
        "out_channels": int(model.out_channels),
        "learn_sigma": bool(model.learn_sigma),
        "param_groups_m": {k: round(v / 1e6, 1) for k, v in part.items()},
        "params_blocks_m": round(n_blocks_total / 1e6, 1),
        "params_per_block_m": {k: round(v / 1e6, 2) for k, v in per_block.items()},
        "adaln_out_dim": 11 * int(hidden),
        "grid": int(grid),
        "tokens_per_frame": int(n_tok),
        "context_tokens": int(T * n_tok),
        "numbers_per_patch": int(model.in_channels * model.patch_size ** 2),
        "patch_expansion": round(hidden / (model.in_channels * model.patch_size ** 2), 1),
        "pos_embed_shape": list(pe.shape),
        "pos_embed_learned": bool(pe.requires_grad),
        "pos_embed_slot_sim": [[round(float(v), 3) for v in r] for r in slot_sim],
        "pos_embed_slot_sim_offdiag_mean": round(float(off.mean()), 3),
        "diffusion_t": t_val,
        "diffusion_steps": DIFFUSION_STEPS,
        "respaced_steps": RESPACED_STEPS,
        "action": [round(float(v), 4) for v in y[0]],
        "rel_t": round(float(rel_t.item()), 5),
        "cond_norms": {"t_embedder": round(float(t_emb.norm()), 2),
                       "time_embedder": round(float(time_emb.norm()), 2),
                       "y_embedder": round(float(y_emb.norm()), 2),
                       "c": round(float(c.norm()), 2)},
        "action_emb_split": [hs, hs, int(hidden - 2 * hs)],
        "adaln_signals": [{"name": nm, "what": what,
                           "mean": round(float(ch.mean()), 4), "std": round(float(ch.std()), 4)}
                          for (nm, what), ch in zip(ADALN_NAMES, chunks)],
        "block_traced": b_idx,
        "block_trace": [{"layer": nm, "in": list(i) if i != "-" else None,
                         "out": list(o) if o != "-" else None} for nm, i, o in trace],
        "gates": {k: [round(v, 4) for v in vals_] for k, vals_ in gates.items()},
        "query_patch": {"index": int(q_idx), "row": int(q_r), "col": int(q_c)},
        "attn_block": int(a_idx),
        "attn_block_why": why,
        "attn_step": step_i,
        "attn_t": t_late,
        "attention_by_depth": [{"block": d["block"], "focus": round(d["top10"], 4),
                                "sink": round(d["bias"], 4),
                                "per_frame": [round(float(v), 4) for v in d["per_frame"]]}
                               for d in depth_late],
        "attention_by_depth_pure_noise": [{"block": d["block"], "focus": round(d["top10"], 4),
                                           "sink": round(d["bias"], 4)} for d in depth_noise],
        "attention_focus_mean_pure_noise": round(float(fa), 2),
        "attention_focus_mean_late": round(float(fb), 2),
        "attention_uniform_top10": round(uniform_top10, 4),
        "cross_attn_shape": list(cw.shape),
        "cross_attn_bias_slot_mass": round(bias_mass, 4),
        "attention_per_frame": [round(float(v), 4) for v in per_frame],
        "attention_peak_per_frame": [[int(v) for v in np.unravel_index(cross_map[i].argmax(), (grid, grid))]
                                     for i in range(T)],
        "self_attn_top_patches": [[int(r), int(c_)] for r, c_ in top],
        "spatial_sim_near": round(float(np.mean(spatial_sim[max(q_r-1,0):q_r+2, max(q_c-1,0):q_c+2])), 3),
        "spatial_sim_far": round(float(spatial_sim[[0,0,-1,-1],[0,-1,0,-1]].mean()), 3),
        "output_shape": list(out.shape),
        "final_layer_out_dim": int(model.patch_size ** 2 * model.out_channels),
        "eps_std": round(float(eps.std()), 4),
        "sigma_mean": round(float(sigma.mean()), 4),
        "echo_cosine": round(echo, 3),
        "guess_frac_pct": round(guess_norm / base_norm * 100, 1),
        "ablations": [{"changed": nm, "how": how,
                       "pct_pure_noise": round(va, 2), "pct_near_end": round(vb, 2)}
                      for nm, how, va, vb in abl],
        "ablation_reference_new_noise_pct": round((other - eps).norm().item() / base_norm * 100, 1),
        "attention_sim_a_vs_b": round(float(np.mean(sim_ab)), 3),
        "attention_sim_context_blanked": round(float(np.mean(sim_noc)), 3),
        "attention_sim_context_blanked_min": round(float(min(sim_noc)), 3),
        "attention_offset_pure_noise": round(float(oa), 2),
        "attention_offset_late": round(float(ob), 2),
        "attention_offset_uniform": round(uniform_offset, 2),
        "t_sweep": sweep,
        "filmstrip": [{"step": si, "t": tv} for si, tv, _ in strip],
        "action_delta_zero_max": round(float(d_zero.max()), 4),
        "action_delta_mirror_max": round(float(d_mirr.max()), 4),
        "action_mirror": [round(float(v), 4) for v in y_mirror[0]],
        "tile_px": int(px),
        "image_size": int(IMG),
    }
    # Written through debug/common/facts.py, never json.dump directly: that is
    # what stamps the schema version the page checks on load (ADR-0004).
    written = facts_io.write(out_dir, "cdit_facts.json", facts, stage="t4",
                             images=save.names)
    print(f"  wrote {written}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
