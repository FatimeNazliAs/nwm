"""
Build the T3 annotated webpage from the probe's output.

Consumes ONLY debug/out/t3/<scene>/vae_facts.json + the PNG panels the probe
wrote, so every number on the page comes from a real run and cannot drift from
the code. Images are inlined as data URIs (the artifact CSP blocks any external
host).

    cd /app && python debug/t3/build_page.py            # newest scene in out/t3
    cd /app && python debug/t3/build_page.py <scene>    # e.g. sample0

Writes <scene>/t3_artifact.html and debug/out/t3/latest.html. Publishing is a
separate step -- running this script alone does not update the live page.

Visual identity lives in debug/common/page.py -- T1..T6 are one series, so the
tokens (teal accent, panel cards, story beats) stay fixed and only the
highlighted pipeline stage moves.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from debug.common.page import data_uri, newest_scene, render, write

OUT_ROOT = "debug/out/t3"
FACTS = "vae_facts.json"

# Beat bodies are bullets, not prose, and one beat is a full-width "follow this
# one frame" figure -- neither shape exists in the shared shell.
EXTRA_CSS = """
  .beat ul{margin:0 0 6px; padding-left:19px; font-size:14.5px; color:var(--muted)}
  .beat li{margin-bottom:5px}
  .beat li b{color:var(--ink)}

  .solo{background:var(--panel); border:2px solid var(--accent); border-radius:12px;
    box-shadow:var(--shadow); padding:18px 20px}
  .solo .hd{font-size:11px; letter-spacing:.09em; text-transform:uppercase; font-weight:800;
    color:var(--accent); margin:0 0 14px}
  .solo .row{display:grid; grid-template-columns:1fr auto 1fr; gap:18px; align-items:center}
  .solo img{width:100%; height:auto; display:block; border-radius:8px; border:1px solid var(--line)}
  .solo .chan{display:grid; grid-template-columns:1fr 1fr; gap:8px}
  .solo .chan img{image-rendering:pixelated}
  .solo .arrow{font-size:26px; color:var(--accent); text-align:center; line-height:1.2}
  .solo .arrow small{display:block; font-size:10.5px; letter-spacing:.05em; text-transform:uppercase;
    font-family:ui-monospace,Menlo,monospace; color:var(--muted); font-weight:700; margin-top:4px}
  .solo .cap{font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--muted);
    text-align:center; margin-top:9px}
  .solo ul{margin:16px 0 0; padding-left:19px; font-size:14.5px; color:var(--muted)}
  .solo li{margin-bottom:5px}
  .solo li b{color:var(--ink)}
  @media (max-width:640px){ .solo .row{grid-template-columns:1fr} }
"""


def build(scene):
    d = os.path.join(OUT_ROOT, scene)
    f = json.load(open(os.path.join(d, FACTS)))

    nc, nh, nw = f["latent_shape"]
    T = f.get("context_size", 4)

    names = ["original", "recon", "error"] + [f"latent_ch{c}" for c in range(nc)]
    names += [f"ctx_f{t}" for t in range(T)]
    names += [f"ctx_f{t}_ch{c}" for t in range(T) for c in range(nc)]
    img = {n: data_uri(os.path.join(d, f"{n}.png")) for n in names}

    pshape = "(%d, %d, %d)" % tuple(f["pixel_shape"])
    lshape = "(%d, %d, %d)" % tuple(f["latent_shape"])
    ctx_shape = "(%d, %d, %d, %d)" % tuple(f["context_latent_shape"])
    fnums = f.get("context_frame_numbers", [])
    block = f["spatial_downsample"]

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

    solo_chan = "".join(
        f'<img src="{img[f"latent_ch{c}"]}" alt="Latent channel {c} of the now frame.">'
        for c in range(nc))

    body = f"""
<header>
  <p class="eyebrow">NWM debug walkthrough · step T3</p>
  <h1>The VAE: pixels ↔ latents</h1>
  <p class="sub">The VAE is used <b>twice</b> in every prediction — once to turn the picture into
  numbers the model can work on, and once at the end to turn numbers back into a picture.</p>
  <div class="chips" style="margin-top:18px">
    <div class="chip"><div class="k">the model</div><div class="v" style="font-size:12.5px">{f['vae']}</div></div>
    <div class="chip"><div class="k">class</div><div class="v" style="font-size:13px">{f['vae_class']}</div></div>
    <div class="chip"><div class="k">size</div><div class="v">{f['vae_total_params_m']} M params</div></div>
    <div class="chip"><div class="k">trained by NWM?</div><div class="v" style="font-size:13px">no — frozen</div></div>
  </div>
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
        <ul>
          <li>The {T} most recent frames, oldest to newest — the last one is <b>"now"</b>.</li>
          <li>Together they show which way things were already moving.</li>
          <li><b>obs_image ({T}, 3, {nh * block}, {nw * block})</b> — {T * f['pixel_numel']:,} numbers.</li>
        </ul>
      </div>
    </div>

    <div class="act here">
      <span class="arrow">↓</span>
      <span class="what">VAE encode — all {T} in one batched call</span>
      <span class="badge">T3 — this step</span>
      <span class="src">isolated_nwm_infer.py:79</span>
    </div>

    <div class="beat here">
      <div class="pic"><div class="strip">
{latent_strip}
      </div></div>
      <div>
        <span class="num">2</span>
        <h3>Each frame becomes its own latent</h3>
        <ul>
          <li>One <b>batched</b> call in, {T} <b>separate</b> latents out.</li>
          <li>They are <b>not merged here</b> — the VAE has no idea the {T} are related. Combining
              them is CDiT's job in step 3.</li>
          <li><b>x_cond {ctx_shape}</b> — {T * f['latent_numel']:,} numbers.</li>
        </ul>
      </div>
    </div>

  </div>

  <div class="solo" style="margin-top:18px">
    <p class="hd">Follow one frame — the "now" frame, on its own</p>
    <div class="row">
      <div>
        <img src="{img['original']}" alt="The 'now' frame from the robot's fisheye camera.">
        <p class="cap">{pshape} · {f['pixel_numel']:,} numbers</p>
      </div>
      <div class="arrow">→<small>encode</small></div>
      <div>
        <div class="chan">{solo_chan}</div>
        <p class="cap">{lshape} · {f['latent_numel']:,} numbers</p>
      </div>
    </div>
    <ul>
      <li><b>{f['compression_ratio']}× fewer numbers</b>, {block}× smaller on each side.</li>
      <li>One cell of a {nh}×{nw} grid stands for an <b>{block}×{block} block</b> of the photo —
          {block * block * 3} numbers described by {nc}.</li>
      <li>The {nc} grids are <b>not colours</b> — they are learned, and no two look alike.</li>
      <li>The scene is <b>still readable</b> in them: the same bright and dark regions, in the same
          places as in the photo.</li>
    </ul>
  </div>

  <div class="story" style="margin-top:0">

    <div class="act">
      <span class="arrow">↓</span>
      <span class="what">CDiT + {f['diffusion_steps']} diffusion steps</span>
      <span class="badge">T4 · T5 — not this step</span>
      <span class="src">models.py · diffusion/</span>
    </div>

    <div class="beat absent">
      <div class="pic">
        <div class="missing">not built yet<br><br>T4 and T5</div>
      </div>
      <div>
        <span class="num">3</span>
        <h3>The model swaps it for a future latent</h3>
        <ul>
          <li>CDiT combines the {T} latents, steered by the <b>action</b> and <b>how far ahead</b> we asked.</li>
          <li>It produces a <b>different</b> latent — one for a frame the robot has not seen yet.</li>
          <li>Runs <b>{f['diffusion_steps']} times</b> per predicted frame. Nothing here is a picture.</li>
        </ul>
      </div>
    </div>

    <div class="act here">
      <span class="arrow">↓</span>
      <span class="what">VAE decode</span>
      <span class="badge">T3 — this step again</span>
      <span class="src">isolated_nwm_infer.py:87</span>
    </div>

    <div class="beat here">
      <div class="pic"><img src="{img['recon']}" alt="A picture rebuilt from a latent."></div>
      <div>
        <span class="num">4</span>
        <h3>Back to a picture</h3>
        <ul>
          <li>The same VAE run backwards — {lshape} in, {pshape} out.</li>
          <li>In the real pipeline this would be the <b>predicted future frame</b>.</li>
          <li>Here we decoded the <i>same</i> latent straight back, to measure the VAE alone.</li>
        </ul>
      </div>
    </div>

  </div>
</section>

<section>
  <h2>What the VAE costs us</h2>
  <figure class="figcard">
    <div class="trip">
      <div class="cell"><p class="cap">went in</p><img src="{img['original']}" alt="Original frame."></div>
      <div class="cell"><p class="cap">came back</p><img src="{img['recon']}" alt="Rebuilt frame, near identical."></div>
      <div class="cell"><p class="cap">what was lost</p><img src="{img['error']}" alt="Error map: bright where the picture has fine detail, black across flat areas."></div>
    </div>
    <figcaption style="color:var(--muted);font-size:13px;margin-top:12px">Black means it came back
    perfectly. The loss is only in <b>fine detail</b> — foliage and hard edges. Flat areas such as sky,
    road and walls come back essentially exact.</figcaption>
  </figure>
  <div class="chips" style="margin-top:16px">
    <div class="chip"><div class="k">PSNR</div><div class="v">{f['psnr_db']} dB</div></div>
    <div class="chip"><div class="k">average pixel off by</div><div class="v">{f['mean_abs_err_255']} / 255</div></div>
    <div class="chip"><div class="k">worst pixel off by</div><div class="v">{f['max_abs_err_255']} / 255</div></div>
  </div>
  <div class="note" style="margin-top:16px">
    <b>This is NWM's ceiling.</b> Every predicted frame comes out through step 4, so nothing the model
    produces can be sharper than {f['psnr_db']} dB — no matter how good CDiT gets.
  </div>
</section>

<section>
  <h2>Next</h2>
  <p><b>T4 — CDiT:</b> step 3 above. The model that turns the latent of <i>now</i> into the latent of
  <i>later</i>.</p>
</section>

<footer>
  <p>Measured on <span class="mono">{f['trajectory']}</span> frame <span class="mono">{f['frame_number']}</span>
  · regenerated from <span class="mono">vae_facts.json</span>.</p>
</footer>
"""
    html = render("T3 — The VAE: pixels ↔ latents", body, EXTRA_CSS)
    # Two copies on purpose: one per scene so earlier runs are never overwritten,
    # and one at a fixed path -- that is what gets served and published, so the
    # link stays the same however often the scene changes.
    write(os.path.join(d, "t3_artifact.html"), html)
    return write(os.path.join(OUT_ROOT, "latest.html"), html)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else newest_scene(OUT_ROOT, FACTS))
