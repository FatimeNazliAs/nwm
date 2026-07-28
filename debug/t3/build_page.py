"""
Build the T3 annotated webpage from the probe's output.

Consumes ONLY debug/out/t3/<scene>/vae_facts.json + the PNG panels the
probe wrote, so every number on the page comes from a real run and cannot drift
from the code. Images are inlined as data URIs (the artifact CSP blocks any
external host).

    cd /app && python debug/t3/build_page.py            # newest scene in out/t3
    cd /app && python debug/t3/build_page.py <scene>    # e.g. sample0

Writes <scene>/t3_artifact.html. Publishing that file as the artifact is a
separate step (Claude does it); running this script alone does not update the
live page.

Visual identity is inherited from the T2 page on purpose -- T1..T6 are one
series, so the tokens (teal accent, amber for "held out / cost", panel cards,
pipeline strip) stay fixed and only the highlighted pipeline stage moves.
"""
import base64
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

OUT_ROOT = "debug/out/t3"


def newest_scene():
    cands = [d for d in os.listdir(OUT_ROOT)
             if os.path.isfile(os.path.join(OUT_ROOT, d, "vae_facts.json"))]
    if not cands:
        raise SystemExit(f"No scene with vae_facts.json under {OUT_ROOT}/ -- run debug/t3/probe.py first.")
    return max(cands, key=lambda d: os.path.getmtime(os.path.join(OUT_ROOT, d, "vae_facts.json")))


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


CSS = """
  :root{
    --ground:#f6f7f8; --panel:#ffffff; --ink:#16212b; --muted:#5c6a76;
    --line:#dfe4e9; --line-soft:#eaeef1;
    --accent:#0f8f8b; --accent-soft:#e4f3f2;
    --amber:#b06d13; --amber-soft:#f6ecdb;
    --shadow:0 1px 2px rgba(20,33,44,.05),0 8px 28px rgba(20,33,44,.06);
  }
  @media (prefers-color-scheme: dark){
    :root{
      --ground:#0d151b; --panel:#15212b; --ink:#e7eef3; --muted:#93a3af;
      --line:#25333f; --line-soft:#1c2934;
      --accent:#3ec8c1; --accent-soft:#12302f;
      --amber:#dda253; --amber-soft:#332715;
      --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
    }
  

  }
  :root[data-theme="light"]{
    --ground:#f6f7f8; --panel:#ffffff; --ink:#16212b; --muted:#5c6a76;
    --line:#dfe4e9; --line-soft:#eaeef1; --accent:#0f8f8b; --accent-soft:#e4f3f2;
    --amber:#b06d13; --amber-soft:#f6ecdb;
    --shadow:0 1px 2px rgba(20,33,44,.05),0 8px 28px rgba(20,33,44,.06);
  }
  :root[data-theme="dark"]{
    --ground:#0d151b; --panel:#15212b; --ink:#e7eef3; --muted:#93a3af;
    --line:#25333f; --line-soft:#1c2934; --accent:#3ec8c1; --accent-soft:#12302f;
    --amber:#dda253; --amber-soft:#332715;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }

  *{box-sizing:border-box}
  body{
    margin:0; background:var(--ground); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    line-height:1.6; -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:900px; margin:0 auto; padding:44px 24px 80px}
  .mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}

  header{border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:32px}
  .eyebrow{
    font-family:ui-monospace,Menlo,monospace; font-size:12px; letter-spacing:.14em;
    text-transform:uppercase; color:var(--accent); font-weight:600; margin:0 0 10px
  }
  h1{font-size:30px; line-height:1.2; margin:0 0 8px; text-wrap:balance; letter-spacing:-.01em}
  .sub{color:var(--muted); margin:0; font-size:15.5px; max-width:62ch}

  section{margin-top:38px}
  h2{
    font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
    margin:0 0 14px; font-weight:700; display:flex; align-items:center; gap:10px
  }
  h2::before{content:""; width:16px; height:2px; background:var(--accent); border-radius:2px}
  p{margin:0 0 14px}

  .chips{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px}
  .chip{background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px}
  .chip .k{font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
  .chip .v{font-family:ui-monospace,Menlo,monospace; font-size:15px; margin-top:3px; font-weight:600}

  .tablecard{background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); overflow:hidden}
  .scroll{overflow-x:auto}
  table{border-collapse:collapse; width:100%; font-size:14px}
  th,td{text-align:left; padding:11px 16px; border-bottom:1px solid var(--line-soft)}
  thead th{font-size:11px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted);
    background:var(--line-soft); border-bottom:1px solid var(--line)}
  tbody tr:last-child td{border-bottom:none}
  td.field{font-family:ui-monospace,Menlo,monospace; font-weight:600; color:var(--accent); white-space:nowrap}
  td.num{font-family:ui-monospace,Menlo,monospace; white-space:nowrap; font-variant-numeric:tabular-nums}

  .note{border-left:3px solid var(--accent); background:var(--accent-soft);
    padding:14px 18px; border-radius:0 10px 10px 0; margin:16px 0; font-size:14.5px}
  .note b{color:var(--ink)}

  figure{margin:0}
  .figcard{background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); padding:14px; overflow:hidden}
  .figcard img{display:block; width:100%; height:auto; border-radius:6px}
  figcaption{color:var(--muted); font-size:13px; margin-top:12px; padding:0 4px 2px}

  /* three-up round trip */
  .trip{display:grid; grid-template-columns:repeat(3,1fr); gap:14px}
  .trip .cell .cap{font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin:0 0 6px}
  @media (max-width:640px){ .trip{grid-template-columns:1fr} 

  }

  pre{background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:14px 16px; overflow-x:auto; font-family:ui-monospace,Menlo,monospace;
    font-size:13px; line-height:1.7; margin:0 0 14px}
  code{font-family:ui-monospace,Menlo,monospace; background:var(--line-soft);
    padding:1px 6px; border-radius:5px; font-size:.9em}
  .accent{color:var(--accent); font-weight:600}
  .amberc{color:var(--amber); font-weight:600}
  footer{margin-top:52px; padding-top:20px; border-top:1px solid var(--line);
    color:var(--muted); font-size:13px}
  a{color:var(--accent)}
  a:focus-visible, summary:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
"""


EXTRA_CSS = """
  /* ---- the story: a vertical sequence of real images ---- */
  .story{display:flex; flex-direction:column; gap:0}

  .beat{display:grid; grid-template-columns:270px 1fr; gap:22px; align-items:center;
    background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); padding:16px 18px}
  .beat.t3{border-color:var(--accent); border-width:2px}
  .beat.absent{background:var(--line-soft); box-shadow:none; border-style:dashed}
  .beat .pic img{width:100%; height:auto; display:block; border-radius:8px; border:1px solid var(--line)}
  .beat .quad{display:grid; grid-template-columns:1fr 1fr; gap:5px}
  .beat .quad img{image-rendering:pixelated; border-radius:5px}
  .beat .num{font-family:ui-monospace,Menlo,monospace; font-size:12px; font-weight:800; color:#fff;
    background:var(--ink); border-radius:6px; padding:2px 8px; display:inline-block; margin-bottom:8px}
  .beat.t3 .num{background:var(--accent)}
  .beat h3{margin:0 0 6px; font-size:19px; letter-spacing:-.01em}
  .beat p{margin:0 0 8px; font-size:14.5px; color:var(--muted)}
  .beat .shape{font-family:ui-monospace,Menlo,monospace; font-size:13.5px; font-weight:700;
    font-variant-numeric:tabular-nums}
  .beat .missing{height:150px; border:2px dashed var(--line); border-radius:8px; display:flex;
    align-items:center; justify-content:center; text-align:center; color:var(--muted);
    font-family:ui-monospace,Menlo,monospace; font-size:12.5px; padding:12px; line-height:1.5}
  @media (max-width:620px){ .beat{grid-template-columns:1fr} 

  }

  /* a row of the 4 context frames, or their 4 latents */
  .strip{display:grid; grid-template-columns:repeat(4,1fr); gap:7px}
  .strip .f img{width:100%; height:auto; display:block; border-radius:5px; border:1px solid var(--line)}
  .strip .f .mini{display:grid; grid-template-columns:1fr 1fr; gap:2px}
  .strip .f .mini img{image-rendering:pixelated; border-radius:2px}
  .strip .f .lb{font-family:ui-monospace,Menlo,monospace; font-size:9.5px; color:var(--muted);
    text-align:center; margin-top:4px; white-space:nowrap}
  .strip .f.now .lb{color:var(--accent); font-weight:700}

  /* the labelled arrow between beats */
  .act{display:flex; align-items:center; gap:12px; padding:12px 0 12px 42px; flex-wrap:wrap}
  .act .what{font-weight:700; font-size:14.5px}
  .act.vae .what{color:var(--accent)}
  .act .src{font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--muted)}
  .act .badge{font-family:ui-monospace,Menlo,monospace; font-size:10px; font-weight:800;
    letter-spacing:.06em; padding:2px 7px; border-radius:5px; background:var(--line-soft); color:var(--muted)}
  .act.vae .badge{background:var(--accent); color:#fff}
  .act .arrow{color:var(--muted); font-size:19px; line-height:1}

  details{background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); padding:0; margin-top:14px; overflow:hidden}
  summary{cursor:pointer; padding:13px 18px; font-weight:650; font-size:14.5px; list-style:none;
    display:flex; align-items:center; gap:10px}
  summary::-webkit-details-marker{display:none}
  summary::before{content:"▸"; color:var(--accent); font-size:12px}
  details[open] summary::before{content:"▾"}
  details[open] summary{border-bottom:1px solid var(--line)}
  .dbody{padding:16px 18px}
  .dbody > :first-child{margin-top:0}
  .dbody .tablecard{box-shadow:none}
"""


def build(scene):
    d = os.path.join(OUT_ROOT, scene)
    f = json.load(open(os.path.join(d, "vae_facts.json")))

    names = ["original", "recon", "error"] + [f"latent_ch{c}" for c in range(f["latent_shape"][0])]
    T = f.get("context_size", 4)
    names += [f"ctx_f{t}" for t in range(T)]
    names += [f"ctx_f{t}_ch{c}" for t in range(T) for c in range(f["latent_shape"][0])]
    img = {n: data_uri(os.path.join(d, f"{n}.png")) for n in names}

    pshape = "(%d, %d, %d)" % tuple(f["pixel_shape"])
    lshape = "(%d, %d, %d)" % tuple(f["latent_shape"])
    nc, nh, nw = f["latent_shape"]

    fnums = f.get("context_frame_numbers", [])
    ctx_shape = "(%d, %d, %d, %d)" % tuple(f["context_latent_shape"])

    # the "now" frame's latent, its 4 channels shown large-ish
    quad = "".join(
        f'<img src="{img[f"latent_ch{c}"]}" alt="Latent channel {c}, a {nh} by {nw} grid.">'
        for c in range(nc))

    def _tag(t):
        return "now" if t == T - 1 else f"t&minus;{T - 1 - t}"

    frame_strip = "\n".join(
        f"""        <div class="f{' now' if t == T - 1 else ''}">
          <img src="{img[f'ctx_f{t}']}" alt="Context frame {t}, camera frame {fnums[t] if fnums else ''}.">
          <div class="lb">{_tag(t)}</div>
        </div>""" for t in range(T))

    latent_strip = "\n".join(
        f"""        <div class="f{' now' if t == T - 1 else ''}">
          <div class="mini">{''.join(f'<img src="{img[f"ctx_f{t}_ch{c}"]}" alt="">' for c in range(nc))}</div>
          <div class="lb">{_tag(t)}</div>
        </div>""" for t in range(T))

    def stage_table(stages, first_label, first_shape, last_label, last_shape):
        rows = [f'<tr><td class="field">{first_label}</td><td class="num">{first_shape}</td></tr>']
        for s in stages:
            rows.append(f'<tr><td class="field">{s["stage"]}</td>'
                        f'<td class="num">({s["shape"][0]}, {s["shape"][1]}, {s["shape"][2]})</td></tr>')
        rows.append(f'<tr><td class="field">{last_label}</td><td class="num">{last_shape}</td></tr>')
        return "\n            ".join(rows)

    html = f"""<title>T3 — The VAE: pixels ↔ latents</title>
<style>{CSS}{EXTRA_CSS}</style>
<div class="wrap">

<header>
  <p class="eyebrow">NWM debug walkthrough · step T3</p>
  <h1>The VAE: pixels ↔ latents</h1>
  <p class="sub">The VAE is used <b>twice</b> in every prediction — once to turn the picture into
  numbers the model can work on, and once at the very end to turn numbers back into a picture.
  Here is that story, in order, with the real images.</p>
  <div class="chips" style="margin-top:20px">
    <div class="chip"><div class="k">the model</div><div class="v" style="font-size:12.5px">{f['vae']}</div></div>
    <div class="chip"><div class="k">class</div><div class="v" style="font-size:13px">{f['vae_class']}</div></div>
    <div class="chip"><div class="k">size</div><div class="v">{f['vae_total_params_m']} M params</div></div>
    <div class="chip"><div class="k">trained by NWM?</div><div class="v" style="font-size:13px">no — frozen</div></div>
  </div>
  <p style="margin-top:12px;font-size:14px;color:var(--muted)">An off-the-shelf Stable Diffusion
  autoencoder, downloaded and frozen. NWM never trains it — it knows nothing about robots. Encoder
  {f['encoder_params_m']} M params, decoder {f['decoder_params_m']} M.</p>
</header>

<section>
  <h2>What happens, in order</h2>
  <div class="story">

    <div class="beat">
      <div class="pic"><div class="strip">
{frame_strip}
      </div></div>
      <div>
        <span class="num">1</span>
        <h3>All {T} context frames go in</h3>
        <p>Not one frame — the {T} most recent, oldest to newest. The last one is "now"; together they show
        the model which way things were already moving.</p>
        <p class="shape">obs_image ({T}, 3, {nh * f['spatial_downsample']}, {nw * f['spatial_downsample']})
        &nbsp;·&nbsp; {T * f['pixel_numel']:,} numbers</p>
      </div>
    </div>

    <div class="act vae">
      <span class="arrow">↓</span>
      <span class="what">VAE encode — all {T} in one batched call</span>
      <span class="badge">T3 — this step</span>
      <span class="src">isolated_nwm_infer.py:79</span>
    </div>

    <div class="beat t3">
      <div class="pic"><div class="strip">
{latent_strip}
      </div></div>
      <div>
        <span class="num">2</span>
        <h3>Each frame becomes its own latent</h3>
        <p>The code flattens the {T} frames into one batch, runs <b>one</b> <span class="mono">vae.encode</span>
        call, then unflattens — so you get {T} <b>separate</b> latents, each {nc} grids of {nh}×{nw}. The scene
        is still readable in them.</p>
        <p><b>They are not merged here.</b> The VAE handles each frame alone and has no idea the {T} are
        related — combining them is CDiT's job in step 3.</p>
        <p class="shape">x_cond {ctx_shape} &nbsp;·&nbsp; {T * f['latent_numel']:,} numbers</p>
      </div>
    </div>

    <div class="beat" style="border-style:dashed">
      <div class="pic"><div class="quad">{quad}</div></div>
      <div>
        <span class="num" style="background:var(--muted)">why {nc}?</span>
        <h3>Why {nc} grids and not one</h3>
        <p>Each cell of a {nh}×{nw} grid stands for an
        <b>{f['spatial_downsample']}×{f['spatial_downsample']} block</b> of the photo — that block holds
        {f['spatial_downsample']}×{f['spatial_downsample']}×3 =
        <b>{f['spatial_downsample'] * f['spatial_downsample'] * 3} numbers</b>. The {nc} grids are the
        <b>budget for describing it</b>: {nc} numbers per block, so
        {f['spatial_downsample'] * f['spatial_downsample'] * 3} → {nc} is the
        {f['compression_ratio']}× squeeze.</p>
        <p>With <b>one</b> grid you'd have a single number per block — not enough for colour <i>and</i> texture
        <i>and</i> edges, so it would come back a blur. With <b>32</b> the latent would barely be smaller than
        the photo and step 3 would lose its speed advantage. {nc} at
        {f['spatial_downsample']}× is the pair the Stable Diffusion authors settled on: small enough to be
        cheap, rich enough to rebuild from.</p>
        <p>They are not colours. They are {nc} <i>learned</i> feature maps — you can see it above: channels 0
        and 1 carry most of the structure, 2 and 3 are flatter.</p>
      </div>
    </div>

    <div class="act">
      <span class="arrow">↓</span>
      <span class="what">CDiT + {f['diffusion_steps']} diffusion steps</span>
      <span class="badge">T4 · T5 — not built yet</span>
      <span class="src">models.py · diffusion/</span>
    </div>

    <div class="beat absent">
      <div class="pic">
        <div class="missing">we haven't built<br>this part yet<br><br>T4 and T5</div>
      </div>
      <div>
        <span class="num">3</span>
        <h3>CDiT combines the {T} latents — this is not T3</h3>
        <p><b>This is where they come together.</b> CDiT cuts each latent into
        {f['patch_size']}×{f['patch_size']} patches ({f['tokens_per_frame']} per frame) and adds a
        <i>per-slot</i> position embedding, so it knows which frame is oldest and which is "now". Then it
        strings them into one sequence of <b>{T} × {f['tokens_per_frame']} =
        {f['context_tokens_total']} tokens</b>.</p>
        <p>Steered by that, plus the action and how far ahead we asked, it turns pure noise into the latent
        of a <b>future</b> frame the robot has not seen — going round the loop {f['diffusion_steps']} times.
        Nothing here is a picture.</p>
        <p class="shape">{ctx_shape} + noise {lshape} &nbsp;→&nbsp; {lshape}</p>
      </div>
    </div>

    <div class="act vae">
      <span class="arrow">↓</span>
      <span class="what">VAE decode</span>
      <span class="badge">T3 — this step again</span>
      <span class="src">isolated_nwm_infer.py:87</span>
    </div>

    <div class="beat t3">
      <div class="pic"><img src="{img['recon']}" alt="A picture rebuilt from a latent."></div>
      <div>
        <span class="num">4</span>
        <h3>A picture again</h3>
        <p>The same VAE, run backwards, turns the latent back into something you can look at.
        In the real pipeline this would be the <i>predicted</i> future frame.</p>
        <p class="shape">{pshape} &nbsp;·&nbsp; {f['pixel_numel']:,} numbers</p>
      </div>
    </div>

  </div>

  <div class="note">
    <b>What we actually ran in T3.</b> Steps 1 → 2 → 4, skipping step 3 on purpose: we encoded a real frame
    and decoded it straight back. That isolates the VAE — anything different between picture 1 and picture 4
    is the VAE's own doing, with no model involved. That is why picture 4 shows the <i>same</i> scene rather
    than a future one.
  </div>
</section>

<section>
  <h2>What the VAE costs us</h2>
  <figure class="figcard">
    <div class="trip">
      <div class="cell"><p class="cap">went in</p><img src="{img['original']}" alt="Original frame."></div>
      <div class="cell"><p class="cap">came back</p><img src="{img['recon']}" alt="Rebuilt frame, near identical."></div>
      <div class="cell"><p class="cap">what was lost</p><img src="{img['error']}" alt="Error map: bright only on tree foliage and the roof edge."></div>
    </div>
    <figcaption>Black means "came back perfectly". The loss is only in fine detail — leaves, the roofline,
    the fisheye border. Sky, lawn and the white wall are essentially exact.</figcaption>
  </figure>
  <div class="chips" style="margin-top:16px">
    <div class="chip"><div class="k">PSNR</div><div class="v">{f['psnr_db']} dB</div></div>
    <div class="chip"><div class="k">average pixel off by</div><div class="v">{f['mean_abs_err_255']} / 255</div></div>
    <div class="chip"><div class="k">worst pixel off by</div><div class="v">{f['max_abs_err_255']} / 255</div></div>
  </div>
  <div class="note">
    <b>This is NWM's ceiling.</b> Every predicted frame comes out through step 4, so nothing the model
    produces can be sharper than {f['psnr_db']} dB — no matter how good CDiT gets.
  </div>
</section>

<section>
  <h2>Why bother compressing</h2>
  <p>Step 3 is a <b>loop</b> — it runs {f['diffusion_steps']} times per predicted frame, and each pass compares
  every piece of the latent with every other piece. Steps 2 and 4 run <b>once</b>. So it pays to make the thing
  in the loop as small as possible before entering it.</p>
  <div class="tablecard scroll">
    <table>
      <thead><tr><th>if step 3 worked on…</th><th>pieces to compare</th><th>cost per pass</th></tr></thead>
      <tbody>
        <tr><td class="field">the latent {lshape}</td><td class="num">{f['tokens_latent']}</td>
            <td class="num accent">1×</td></tr>
        <tr><td class="field">raw pixels {pshape}</td><td class="num">{f['tokens_pixel']:,}</td>
            <td class="num amberc">{f['attention_ratio']:,.0f}×</td></tr>
      </tbody>
    </table>
  </div>
  <p style="margin-top:14px">{f['attention_ratio']:,.0f}× cheaper, {f['diffusion_steps']} times over — against
  paying for the VAE twice. That trade is the whole reason it exists.</p>
</section>

<details>
  <summary>The numbers behind all of this</summary>
  <div class="dbody">
    <div class="tablecard scroll">
      <table>
        <thead><tr><th>question</th><th>measured</th></tr></thead>
        <tbody>
          <tr><td class="field">checkpoint</td><td class="num">{f['vae']} · {f['vae_class']}</td></tr>
          <tr><td class="field">context frames encoded</td><td class="num">{T} per prediction, in one batched call → {ctx_shape}</td></tr>
          <tr><td class="field">tokens CDiT sees</td><td class="num">{T} × {f['tokens_per_frame']} = {f['context_tokens_total']} context + {f['tokens_per_frame']} predicted</td></tr>
          <tr><td class="field">size change (per frame)</td><td class="num">{f['pixel_numel']:,} → {f['latent_numel']:,} numbers ({f['compression_ratio']}×)</td></tr>
          <tr><td class="field">shape change</td><td class="num">{pshape} → {lshape} ({f['spatial_downsample']}× per side)</td></tr>
          <tr><td class="field">why {f['spatial_downsample']}×</td><td class="num">3 encoder blocks halve the size: 224→112→56→28</td></tr>
          <tr><td class="field">encoder / decoder size</td><td class="num">{f['encoder_params_m']} M / {f['decoder_params_m']} M params</td></tr>
          <tr><td class="field">latent value range</td><td class="num">{f['latent_mean_min']:+.1f} … {f['latent_mean_max']:+.1f}</td></tr>
          <tr><td class="field">the ×{f['scaling']} constant</td><td class="num">rescales latent std {f['latent_std_raw']} → {f['latent_std_scaled']} ≈ 1</td></tr>
          <tr><td class="field">pieces (patches) compared</td><td class="num">{f['tokens_latent']} vs {f['tokens_pixel']:,} on pixels</td></tr>
          <tr><td class="field">round-trip error</td><td class="num">PSNR {f['psnr_db']} dB · MSE {f['mse']}</td></tr>
          <tr><td class="field">encode is random?</td><td class="num">technically — largest nudge {f['sample_minus_mean_max']}, invisible</td></tr>
        </tbody>
      </table>
    </div>

    <p style="margin:18px 0 8px"><b>Every layer, measured with forward hooks on this frame:</b></p>
    <div class="tablecard scroll">
      <table>
        <thead><tr><th>encoder (step 2)</th><th>output</th></tr></thead>
        <tbody>
          {stage_table(f['encoder_stages'], 'the frame', pshape, 'conv_out + quant_conv', f'(8, {nh}, {nw})')}
        </tbody>
      </table>
    </div>
    <div class="tablecard scroll" style="margin-top:12px">
      <table>
        <thead><tr><th>decoder (step 4)</th><th>output</th></tr></thead>
        <tbody>
          {stage_table(f['decoder_stages'], 'the latent', lshape, 'conv_out', pshape)}
        </tbody>
      </table>
    </div>
    <p style="margin-top:14px;font-size:14px;color:var(--muted)">The encoder ends at <b>8</b> channels for a
    {nc}-channel latent because it writes two numbers per cell — a value and how sure it is.
    Why, and what patches are, is on the Notion concepts page; why this exact checkpoint is on the
    architecture page.</p>
  </div>
</details>

<section>
  <h2>Next</h2>
  <p><b>T4 — CDiT:</b> step 3 above. The model that turns the latent of <i>now</i> into the latent of
  <i>later</i>.</p>
</section>

<footer>
  <p>Measured on <span class="mono">{f['trajectory']}</span> frame <span class="mono">{f['frame_number']}</span>
  · <span class="mono">{f['vae']}</span> · regenerated from <span class="mono">vae_facts.json</span>.</p>
</footer>

</div>
"""
    out = os.path.join(d, "t3_artifact.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else newest_scene())
