"""
Build the T4 annotated webpage from the probe's output.

Consumes ONLY debug/out/t4/<scene>/cdit_facts.json + the PNG panels the probe
wrote, so every number on the page comes from a real run and cannot drift from
the code. Images are inlined as data URIs (the artifact CSP blocks any external
host); the page shell comes from debug/common/page.py, shared with T3.

    cd /app && python debug/t4/build_page.py            # newest scene in out/t4
    cd /app && python debug/t4/build_page.py <scene>

Writes <scene>/t4_artifact.html. Publishing that file as the artifact is a
separate step; running this script alone does not update the live page.

Layout rule for this page: the visible story is made of things you can LOOK at
-- the frames, the tile grid, the attention, the filmstrip, the action maps.
Anything that is a statistic rather than a picture (similarity matrices, gate
curves, focus-by-depth, shape traces) lives in the collapsed drawer at the end.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from debug.common import facts as facts_io
from debug.common.page import data_uri, newest_scene, render, write

OUT_ROOT = "debug/out/t4"
FACTS = "cdit_facts.json"

# The handful of facts the page cannot be built without. Deliberately short: the
# probe writes ~85 keys and pinning all of them here would make every probe
# change a page change. These are the ones whose absence means the file is from
# the wrong stage or a much older probe.
REQUIRED = ("model", "context_size", "in_channels", "tokens_per_frame",
            "latent_size", "depth", "hidden", "trajectory", "filmstrip",
            "ablations", "t_sweep")

EXTRA_CSS = """
  .beat .wide{grid-column:1 / -1}
  .beat .plot img{width:100%; height:auto; display:block; border-radius:8px; background:#fff}
  .beat .nums{font-family:ui-monospace,Menlo,monospace; font-size:12.5px; line-height:1.9;
    background:var(--line-soft); border-radius:8px; padding:12px 14px}
  .beat .nums b{color:var(--accent)}

  /* the loop, as a row of real decoded frames */
  .film{display:grid; grid-template-columns:repeat(8,1fr); gap:6px}
  .film .s img{width:100%; height:auto; display:block; border-radius:4px; border:1px solid var(--line)}
  .film .s .lb{font-family:ui-monospace,Menlo,monospace; font-size:9px; color:var(--muted);
    text-align:center; margin-top:4px; white-space:nowrap}
  .film .s.last .lb{color:var(--accent); font-weight:700}
  @media (max-width:560px){ .film{grid-template-columns:repeat(4,1fr)} }

  /* every step states its input, its operation and its output, in that order */
  .iox{display:grid; grid-template-columns:1fr 1.25fr 1fr; border:1px solid var(--line);
    border-radius:10px; overflow:hidden; margin:0 0 16px; background:var(--panel)}
  .iox > div{padding:10px 13px}
  .iox .mid{background:var(--accent-soft);
    border-left:1px solid var(--line); border-right:1px solid var(--line)}
  .iox .lbl{display:block; font-family:ui-monospace,Menlo,monospace; font-size:9.5px;
    letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:800;
    margin:0 0 4px}
  .iox .v{font-family:ui-monospace,Menlo,monospace; font-size:12.5px; font-weight:600;
    line-height:1.5; display:block}
  .iox .mid .v{color:var(--accent)}
  @media (max-width:620px){ .iox{grid-template-columns:1fr}
    .iox .mid{border-left:none; border-right:none;
      border-top:1px solid var(--line); border-bottom:1px solid var(--line)} }

  /* the Figure 2 diagram */
  figure.fig2{background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); padding:18px 16px 14px; margin:0 0 14px}
  figure.fig2 svg{display:block; width:100%; max-width:560px; height:auto; margin:0 auto;
    color:var(--ink)}

  /* every beat explains itself as numbered steps, not prose */
  ol.steps{margin:0 0 12px; padding-left:22px}
  ol.steps li{margin:0 0 6px; font-size:14.5px; color:var(--ink); line-height:1.55}
  ol.steps li::marker{color:var(--accent); font-weight:700; font-size:13px}
  .aside{font-size:13.5px; color:var(--muted); border-left:2px solid var(--line);
    padding-left:12px; margin:0 0 10px}
  .aside b{color:var(--ink)}

  .two{display:grid; grid-template-columns:1fr 1fr; gap:14px}
  .two .cell img{width:100%; height:auto; display:block; border-radius:8px; border:1px solid var(--line)}
  .two .cell .cap{font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin:0 0 6px}
  @media (max-width:620px){ .two{grid-template-columns:1fr} }
"""


def image_names(f):
    """Every PNG this page expects the probe to have written, by bare name.

    Part of the probe/page contract (ADR-0001) alongside the facts keys: the
    probe promises these files exist in the scene folder. Exposed rather than
    inlined so a test can create stubs for exactly this set.
    """
    T, nc = f["context_size"], f["in_channels"]
    return (["tiles_on_frame", "tiles_on_latent", "loop_filmstrip",
             "actiondelta_zero", "actiondelta_mirror", "posembed", "gates",
             "focus", "tsweep", "selfattn", "target"]
            + [f"ctx_f{t}" for t in range(T)]
            + [f"attn_f{t}" for t in range(T)]
            + [f"ctx_f{t}_ch{c}" for t in range(T) for c in range(nc)]
            + [f"noise_ch{c}" for c in range(nc)]
            + [f"eps_ch{c}" for c in range(nc)]
            + [f"loop_s{s['step']}" for s in f["filmstrip"]])


def build(scene, audit=False):
    d = os.path.join(OUT_ROOT, scene)
    f = facts_io.load(os.path.join(d, FACTS), required=REQUIRED)

    T = f["context_size"]
    nc = f["in_channels"]
    n_tok = f["tokens_per_frame"]
    grid = f["grid"]
    lat = f["latent_size"]
    film = f["filmstrip"]

    img = {n: data_uri(os.path.join(d, f"{n}.png")) for n in image_names(f)}

    def tag(t):
        return "now" if t == T - 1 else f"t&minus;{T - 1 - t}"

    frame_strip = "\n".join(
        f"""        <div class="f{' now' if t == T - 1 else ''}">
          <img src="{img[f'ctx_f{t}']}" alt="Context frame {t}.">
          <div class="lb">{tag(t)}</div>
        </div>""" for t in range(T))

    latent_strip = "\n".join(
        f"""        <div class="f{' now' if t == T - 1 else ''}">
          <div class="mini">{''.join(f'<img src="{img[f"ctx_f{t}_ch{c}"]}" alt="">' for c in range(nc))}</div>
          <div class="lb">{tag(t)}</div>
        </div>""" for t in range(T))

    attn_strip = "\n".join(
        f"""        <div class="f{' now' if t == T - 1 else ''}">
          <img src="{img[f'attn_f{t}']}" alt="Context frame {t} with the attention painted on.">
          <div class="lb">{f['attention_per_frame'][t]*100:.0f}%</div>
        </div>""" for t in range(T))

    film_strip = "\n".join(
        f"""      <div class="s{' last' if i == len(film) - 1 else ''}">
        <img src="{img[f'loop_s{s["step"]}']}" alt="The frame at step {s['step']} of the loop.">
        <div class="lb">step {s['step']}</div>
      </div>""" for i, s in enumerate(film))

    noise_quad = "".join(f'<img src="{img[f"noise_ch{c}"]}" alt="">' for c in range(nc))
    eps_quad = "".join(f'<img src="{img[f"eps_ch{c}"]}" alt="">' for c in range(nc))

    ab = {a["how"]: a for a in f["ablations"]}
    sw = f["t_sweep"]

    adaln_rows = "\n            ".join(
        f'<tr><td class="field">{s["name"]}</td><td>{s["what"]}</td>'
        f'<td class="num">{s["mean"]:+.4f}</td><td class="num">{s["std"]:.4f}</td></tr>'
        for s in f["adaln_signals"])
    trace_rows = "\n            ".join(
        f'<tr><td class="field">{r["layer"]}</td>'
        f'<td class="num">{tuple(r["in"]) if r["in"] else "—"}</td>'
        f'<td class="num">{tuple(r["out"]) if r["out"] else "—"}</td></tr>'
        for r in f["block_trace"])
    sweep_rows = "\n            ".join(
        f'<tr><td class="num">{s["step"]}</td><td class="num">{s["t"]}</td>'
        f'<td class="num">{s["echo"]:.3f}</td><td class="num">{s["guess_frac"]}%</td>'
        f'<td class="num">{s["action_pct_of_guess"]:.1f}%</td></tr>' for s in sw)


    # --- the CDiT block, drawn to match Figure 2 of the paper ---
    def box(x, y, w, h, label, sub="", accent=False, dash=False):
        st = "#0f8f8b" if accent else "currentColor"
        sw = "2" if accent else "1.2"
        da = ' stroke-dasharray="4 3"' if dash else ""
        t = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="none" '
             f'stroke="{st}" stroke-width="{sw}"{da}/>')
        if sub:
            t += (f'<text x="{x+w/2}" y="{y+h/2-2}" text-anchor="middle" font-size="12.5" '
                  f'fill="{st}" font-weight="600">{label}</text>'
                  f'<text x="{x+w/2}" y="{y+h/2+13}" text-anchor="middle" font-size="10.5" '
                  f'fill="currentColor" opacity=".65">{sub}</text>')
        else:
            t += (f'<text x="{x+w/2}" y="{y+h/2+4.5}" text-anchor="middle" font-size="12.5" '
                  f'fill="{st}">{label}</text>')
        return t

    MX, MW = 190, 200          # main column left edge, width
    CX = 300                   # main column centre
    rows = [
        (14, 30, "x — 196 squares", "the frame being invented"),
        (66, 26, "LayerNorm", ""),
        (106, 26, "scale, shift", ""),
        (146, 40, "SELF-ATTENTION", "196 read each other"),
        (204, 26, "× gate", ""),
        (290, 26, "LayerNorm", ""),
        (330, 26, "scale, shift", ""),
        (370, 40, "CROSS-ATTENTION", "196 read the 784 past"),
        (428, 26, "× gate", ""),
        (514, 26, "LayerNorm", ""),
        (554, 26, "scale, shift", ""),
        (594, 40, "FEED-FORWARD (MLP)", "each square alone"),
        (652, 26, "× gate", ""),
        (738, 30, "out — 196 squares", "same shape as x"),
    ]
    parts = []
    for y, h, lab, sub in rows:
        parts.append(box(MX, y, MW, h, lab, sub, accent=("CROSS" in lab)))
    # the three residual joins
    for cy in (256, 480, 704):
        parts.append(f'<circle cx="{CX}" cy="{cy}" r="14" fill="none" stroke="currentColor" '
                     f'stroke-width="1.2"/>'
                     f'<text x="{CX}" y="{cy+5}" text-anchor="middle" font-size="15">+</text>')
    # straight arrows down the main column
    downs = [(44,66),(92,106),(132,146),(186,204),(230,242),(270,290),(316,330),(356,370),
             (410,428),(454,466),(494,514),(540,554),(580,594),(634,652),(678,690),(718,738)]
    for a, b in downs:
        parts.append(f'<line x1="{CX}" y1="{a}" x2="{CX}" y2="{b-6}" stroke="currentColor" '
                     f'stroke-width="1.2" marker-end="url(#ar)"/>')
    # residual bypasses
    for y0, y1 in ((46, 256), (272, 480), (496, 704)):
        parts.append(f'<path d="M {CX} {y0} H 168 V {y1} H {CX-20}" fill="none" '
                     f'stroke="currentColor" stroke-width="1.2" opacity=".55" '
                     f'marker-end="url(#ar)"/>')
    parts.append('<text x="160" y="250" text-anchor="end" font-size="10" fill="currentColor" '
                 'opacity=".55">added back</text>')
    # the conditioning column
    parts.append(box(14, 14, 130, 30, "c", "action + horizon + step"))
    parts.append(box(14, 66, 130, 40, "SiLU + Linear", "1152 → 11 × 1152"))
    parts.append(f'<line x1="79" y1="44" x2="79" y2="60" stroke="currentColor" '
                 f'stroke-width="1.2" marker-end="url(#ar)"/>')
    parts.append('<text x="79" y="126" text-anchor="middle" font-size="10.5" '
                 'fill="currentColor" opacity=".75">11 control signals</text>')
    parts.append(f'<path d="M 79 132 V 720" fill="none" stroke="currentColor" '
                 f'stroke-width="1.1" stroke-dasharray="4 3" opacity=".5"/>')
    for y in (119, 217, 343, 441, 567, 665):
        parts.append(f'<line x1="79" y1="{y}" x2="{MX-6}" y2="{y}" stroke="currentColor" '
                     f'stroke-width="1.1" stroke-dasharray="4 3" opacity=".5" '
                     f'marker-end="url(#ar)"/>')
    # the context side
    parts.append(box(410, 290, 176, 30, "x_cond", "784 past squares"))
    parts.append(box(410, 330, 176, 26, "LayerNorm", ""))
    parts.append(box(410, 370, 176, 40, "scale, shift", "the context side too"))
    parts.append('<line x1="498" y1="320" x2="498" y2="324" stroke="currentColor" '
                 'stroke-width="1.2" marker-end="url(#ar)"/>')
    parts.append('<line x1="498" y1="356" x2="498" y2="364" stroke="currentColor" '
                 'stroke-width="1.2" marker-end="url(#ar)"/>')
    parts.append(f'<line x1="410" y1="390" x2="{MX+MW+6}" y2="390" stroke="#0f8f8b" '
                 f'stroke-width="1.6" marker-end="url(#arT)"/>')
    parts.append('<text x="400" y="382" text-anchor="end" font-size="10" fill="#0f8f8b">'
                 'keys, values</text>')
    parts.append('<path d="M 594 300 V 390 H 590" fill="none" stroke="currentColor" '
                 'stroke-width="1.1" stroke-dasharray="4 3" opacity=".5" '
                 'marker-end="url(#ar)"/>')
    parts.append('<text x="594" y="294" text-anchor="middle" font-size="10" '
                 'fill="currentColor" opacity=".6">c</text>')

    fig2 = ('<svg viewBox="0 0 600 790" role="img" aria-label="One CDiT block: x passes '
            'LayerNorm, scale and shift, self-attention, a gate and a residual add; then the '
            'same pattern with cross-attention reading the 784 context squares; then the same '
            'pattern with a feed-forward layer. The vector c supplies 11 control signals to '
            'every scale, shift and gate.">'
            '<defs>'
            '<marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
            'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker>'
            '<marker id="arT" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
            'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#0f8f8b"/></marker>'
            '</defs>' + "".join(parts) + '</svg>')

    body = f"""
<header>
  <p class="eyebrow">NWM debug walkthrough · step T4</p>
  <h1>CDiT: the part that does the predicting</h1>
  <p class="sub">Everything else in the pipeline either moves data around or is a frozen
  autoencoder. This is the model NWM actually trained. It is handed five things and hands back
  one. What it hands back is not a picture.</p>
  <div class="chips" style="margin-top:20px">
    <div class="chip"><div class="k">the model</div><div class="v">{f['model']}</div></div>
    <div class="chip"><div class="k">class</div><div class="v">{f['model_class']}</div></div>
    <div class="chip"><div class="k">size</div><div class="v">{f['params_total_m']:,.0f} M params</div></div>
    <div class="chip"><div class="k">trained by NWM?</div>
      <div class="v" style="font-size:13px">yes — this is the one</div></div>
  </div>
  <p style="margin-top:12px;font-size:14px;color:var(--muted)">Checkpoint
  <span class="mono">{f['checkpoint']}.pth.tar</span>, key <span class="mono">{f['checkpoint_key']}</span>
  at step {f['train_step']:,}. Same shape as DiT-XL/2: {f['depth']} blocks, width {f['hidden']},
  {f['num_heads']} heads, patch {f['patch_size']}. Cross-attention is added so it can read the past.</p>
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
        <h3>Input — the frames before now</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">frames {f['context_frame_numbers'][0]}–{f['context_frame_numbers'][-1]}<br>of the drive</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">this is the starting material.<br>nothing has been done yet</span></div>
          <div><span class="lbl">out</span><span class="v">{T} photos<br>({T}, 3, {f['image_size']}, {f['image_size']})</span></div>
        </div>
        <ol class="steps">
          <li>The camera took these {T} frames, a quarter of a second apart.</li>
          <li>We ask one question: what does the camera see <b>{f['sec']} seconds</b> later?</li>
          <li>That is frame <b>{f['target_frame_number']}</b>. It is held back — the model
              never sees it.</li>
        </ol>
      </div>
    </div>

    <div class="act"><span class="arrow">↓</span><span class="badge">T3 — done</span></div>

    <div class="beat">
      <div class="pic"><div class="strip">
{latent_strip}
      </div></div>
      <div>
        <span class="num">2</span>
        <h3>VAE encode — photo → latent</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">1 photo<br>(3, {f['image_size']}, {f['image_size']})</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">the VAE encoder from T3.<br>frozen, never trained by NWM</span></div>
          <div><span class="lbl">out</span><span class="v">1 latent<br>({nc}, {lat}, {lat}) &nbsp;·&nbsp; done ×{T}</span></div>
        </div>
        <ol class="steps">
          <li>Each photo goes through the VAE.</li>
          <li>Out comes a <b>latent</b>: {nc} small grids of {lat}×{lat} numbers.</li>
          <li>The colours are mine. Those numbers are not colours. I painted them
              blue-to-yellow so you can see them.</li>
          <li>There are still {T} separate things. The VAE handled each photo alone.</li>
        </ol>
      </div>
    </div>

    <div class="act here"><span class="arrow">↓</span><span class="badge">T4 — this step</span>
      <span class="src">models.py:233–235</span></div>

    <div class="beat here">
      <div class="wide">
        <span class="num">3</span>
        <h3>Patchify — latent → squares</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">{T} latents<br>({T}, {nc}, {lat}, {lat})</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">cut into {f['patch_size']}×{f['patch_size']} squares,<br>expand each to {f['hidden']},<br>add a per-frame stamp</span></div>
          <div><span class="lbl">out</span><span class="v">one list of<br>{f['context_tokens']} squares × {f['hidden']}</span></div>
        </div>
        <div class="two" style="margin:12px 0 16px">
          <figure class="cell">
            <p class="cap">the cut, on the latent — where it really happens</p>
            <img src="{img['tiles_on_latent']}"
              alt="One latent grid of 28 by 28 cells, lines every 2 cells, one square outlined.">
          </figure>
          <figure class="cell">
            <p class="cap">the same cut, drawn on the photo</p>
            <img src="{img['tiles_on_frame']}"
              alt="The same context frame with a 14 by 14 grid drawn over it, one square outlined.">
          </figure>
        </div>
        <ol class="steps">
          <li>Take one latent from step 2. It is a grid of {lat}×{lat} cells — the
              <b>left</b> picture.</li>
          <li>Chop it into {f['patch_size']}×{f['patch_size']} squares:
              {grid} × {grid} = <b>{n_tok} squares</b>.</li>
          <li>Each square holds {f['numbers_per_patch']} numbers.</li>
          <li>One small layer turns those {f['numbers_per_patch']} into <b>{f['hidden']}</b>.
              The count goes <b>up</b>, not down.</li>
          <li>Add a learned stamp saying which frame and which position this square came from.</li>
          <li>Repeat for all {T} past frames, then line them up:
              {T} × {n_tok} = <b>{f['context_tokens']} squares in one list</b>.</li>
        </ol>
        <p class="aside">The <b>right</b> picture is the same cut drawn on the photo, to show what
        one square covers — about {f['tile_px']} × {f['tile_px']} pixels. The model never sees
        photos. The proper name for one square is a <b>token</b>.</p>
      </div>
    </div>

    <div class="beat here">
      <div class="pic"><div class="quad">{noise_quad}</div></div>
      <div>
        <span class="num">4</span>
        <h3>The other inputs</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">random noise,<br>and 3 plain numbers</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">the noise is cut into squares<br>just like step 3.<br>the numbers stay numbers</span></div>
          <div><span class="lbl">out</span><span class="v">{n_tok} squares<br>+ t, y, rel_t</span></div>
        </div>
        <ol class="steps">
          <li><b>x</b> — what the model must clean up. On the first pass it is pure random
              numbers. That is the picture on the left.</li>
          <li><b>t</b> = {f['diffusion_t']} — how noisy x is <i>right now</i>.</li>
          <li><b>y</b> = ({f['action'][0]:+.2f}, {f['action'][1]:+.2f}, {f['action'][2]:+.2f})
              — the action. Where the robot ends up.</li>
          <li><b>rel_t</b> = {f['rel_t']} — how far into the <i>future</i> we are asking.</li>
        </ol>
        <p class="aside"><b>t and rel_t are different clocks.</b> Making this one frame takes
        {f['respaced_steps']} passes. <span class="mono">t</span> counts down from
        {f['diffusion_t']} to 0 across them. <span class="mono">rel_t</span> stays
        {f['rel_t']} in every single one.</p>
      </div>
    </div>

    <div class="act here"><span class="arrow">↓</span>
      <span class="src">models.py:236–239</span></div>

    <div class="beat here">
      <div class="wide">
        <span class="num">5</span>
        <h3>Conditioning — 3 numbers → one steering vector</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">t, y, rel_t<br>3 plain numbers</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">each becomes a {f['hidden']}-long vector,<br>then the three are added</span></div>
          <div><span class="lbl">out</span><span class="v">c ({f['hidden']})<br>→ {f['adaln_out_dim']:,} controls per block</span></div>
        </div>
        <ol class="steps">
          <li>Three plain numbers say what is being asked: how noisy the frame is now, where
              the robot goes, and how far ahead to look.</li>
          <li>Each is spread into a {f['hidden']}-long vector, and the three are added into
              one vector called <b>c</b>.</li>
          <li>c <b>never</b> joins the list of squares from step 3.</li>
          <li>Instead all {f['depth']} blocks read c and use it to adjust their own layers.</li>
        </ol>
        <details>
          <summary>Proof that c changes the answer — two pictures</summary>
          <div class="dbody">
            <div class="two" style="margin:0 0 12px">
              <figure class="cell">
                <p class="cap">left · the frame we are predicting</p>
                <img src="{img['target']}" alt="The true future frame, plain.">
              </figure>
              <figure class="cell">
                <p class="cap">right · the same frame, marked</p>
                <img src="{img['actiondelta_mirror']}"
                  alt="The same future frame with a difference map painted on top.">
              </figure>
            </div>
            <ol class="steps">
              <li><b>Left</b> — the true frame, with nothing painted on it. Use it as the
                  reference.</li>
              <li><b>Right</b> — the same frame with <b>bright marks</b> added. Look for where
                  those marks are, and how little of the frame they cover.</li>
              <li>The marks are the <b>only</b> difference between the two pictures.</li>
              <li><b>Bright</b> = changing the action changed the model's answer there.
                  <b>Unmarked</b> = the same answer either way.</li>
            </ol>
          </div>
        </details>
        <div class="nums" style="margin:10px 0">
          real action &nbsp;&nbsp;({f['action'][0]:+.2f}, {f['action'][1]:+.2f}, {f['action'][2]:+.2f})<br>
          mirrored &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;({f['action_mirror'][0]:+.2f},
          {f['action_mirror'][1]:+.2f}, {f['action_mirror'][2]:+.2f})
        </div>
        <p class="aside"><b>If the action did nothing, the right-hand picture would be blank.</b>
        The bright marks are painted on — they show where the two answers disagreed. They are
        patchy rather than spread evenly, so the action changes the answer in particular
        places, not everywhere at once.</p>
        <p>As a number: removing the action entirely moves the answer by
        <b>{ab['y = 0']['pct_pure_noise']:.0f}%</b> at the start of the loop, and only
        <b>{ab['y = 0']['pct_near_end']:.0f}%</b> near the end — the action decides the layout
        early, then has nothing left to say once the frame is settled.</p>
      </div>
    </div>

    <div class="act here"><span class="arrow">↓</span>
      <span class="src">models.py:104–110 · paper Figure 2</span></div>

    <div class="beat here">
      <div class="wide">
        <span class="num">6</span>
        <h3>Inside one block — the paper's Figure 2</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">{n_tok} squares (x)<br>{f['context_tokens']} squares (x_cond)<br>c</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">three sub-layers, each one<br>modulated and gated by c</span></div>
          <div><span class="lbl">out</span><span class="v">{n_tok} squares<br><b>same shape as x</b><br>then ×{f['depth']} blocks</span></div>
        </div>
        <figure class="fig2">
          {fig2}
          <figcaption>One CDiT block. Read it top to bottom. <b>The teal box is the only part a
          standard DiT does not have</b> — it is what lets the model read four past frames without
          the cost growing quadratically.</figcaption>
        </figure>
        <ol class="steps">
          <li>Every sub-layer does the same four things: <b>normalise</b>, then
              <b>scale and shift</b> using c, then <b>the actual operation</b>, then
              <b>× a gate</b> and <b>add back</b> into x.</li>
          <li><b>Self-attention</b> — the {n_tok} squares of the noisy future frame read each
              other.</li>
          <li><b>Cross-attention</b> — those same squares read the {f['context_tokens']} squares of
              the past. One way only; the past is never changed.</li>
          <li><b>Feed-forward</b> — each square is thought about on its own, no mixing.</li>
          <li>Because every part is <i>added</i> back, x leaves the block the same shape it
              entered. The {f['depth']} blocks are {f['depth']} successive edits to the same
              {n_tok} squares.</li>
          <li>The dashed lines are c. It reaches inside every sub-layer instead of joining the
              list — including the context side, before cross-attention reads it.</li>
        </ol>
      </div>
    </div>

    <div class="beat here">
      <div class="pic"><div class="strip">
{attn_strip}
      </div></div>
      <div>
        <span class="num">7</span>
        <h3>Cross-attention, close up</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">1 square's question<br>+ {f['context_tokens']} past squares</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">score every past square,<br>softmax, take a blend</span></div>
          <div><span class="lbl">out</span><span class="v">what that one square<br>takes from the past</span></div>
        </div>
        <ol class="steps">
          <li>Take the outlined square from step 3.</li>
          <li>It reads all {f['context_tokens']} squares of the {T} past frames.</li>
          <li>It decides how much to take from each one.</li>
          <li>The glow above is where it looked, painted on the real frames.</li>
          <li>It leans towards <b>now</b> ({f['attention_per_frame'][T-1]*100:.0f}%), but reads
              all {T} every time.</li>
        </ol>
      </div>
    </div>

    <div class="beat here">
      <div class="pic"><div class="quad">{eps_quad}</div></div>
      <div>
        <span class="num">8</span>
        <h3>Output — a guess at the noise</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">{n_tok} squares × {f['hidden']}</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">one last layer,<br>then reassemble into a grid</span></div>
          <div><span class="lbl">out</span><span class="v">(1, {f['out_channels']}, {lat}, {lat})<br>{nc} noise + {nc} certainty</span></div>
        </div>
        <ol class="steps">
          <li>The model was asked: how much of x was noise?</li>
          <li>So its answer is noise-shaped. That is why it still looks like static.</li>
          <li>The first {nc} channels are the answer. The other {nc} say how sure it is.</li>
          <li>One pass changes almost nothing. This output is {f['echo_cosine']:.3f} identical
              to what went in. Only {f['guess_frac_pct']}% of it is an opinion about this
              scene.</li>
        </ol>
      </div>
    </div>

    <div class="act"><span class="arrow">↓</span><span class="badge">T5 — not built yet</span></div>

    <div class="beat absent">
      <div class="wide">
        <span class="num" style="background:var(--muted)">9</span>
        <h3>The loop — repeat ×{f['respaced_steps']}</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">x, and the guess<br>from step 8</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">take a little of the guess<br>away from x, lower t,<br>go round again</span></div>
          <div><span class="lbl">out</span><span class="v">after {f['respaced_steps']} rounds:<br>a finished latent</span></div>
        </div>
        <div class="film" style="margin:14px 0 10px">
{film_strip}
        </div>
        <p class="aside">Nothing looks like anything until about two-thirds through.
        <b>To be straight about what this is:</b> these are the real noise levels applied to the
        real future frame, so it shows the target the model climbs towards. It is not output we
        generated. Generating it for real is <b>T5</b>.</p>
      </div>
    </div>

    <div class="act"><span class="arrow">↓</span><span class="badge">T3 — done</span></div>

    <div class="beat absent">
      <div class="pic"><div class="missing">the predicted frame<br><br>T6</div></div>
      <div>
        <span class="num">10</span>
        <h3>VAE decode — latent → picture</h3>
        <div class="iox">
          <div><span class="lbl">in</span><span class="v">finished latent<br>({nc}, {lat}, {lat})</span></div>
          <div class="mid"><span class="lbl">what happens</span><span class="v">the VAE decoder from T3.<br>frozen, same one as step 2</span></div>
          <div><span class="lbl">out</span><span class="v">frame {f['target_frame_number']}<br>(3, {f['image_size']}, {f['image_size']})</span></div>
        </div>
        <ol class="steps">
          <li>The finished latent goes back through the VAE.</li>
          <li>Out comes frame {f['target_frame_number']}, as the model imagines it.</li>
          <li>Comparing that with what really happened is <b>T6</b>.</li>
        </ol>
      </div>
    </div>

  </div>
</section>

<section>
  <h2>Next</h2>
  <p><b>T5 — the loop.</b> Step 7 gives back a guess at the noise. Turning
  {f['respaced_steps']} of those guesses into the frame in step 9 is the diffusion sampler.</p>
</section>

<footer>
  <p>Measured on <span class="mono">{f['trajectory']}</span> frame {f['curr_time']}, predicting
  {f['sec']}s ahead · <span class="mono">{f['model']}</span> checkpoint
  <span class="mono">{f['checkpoint']}</span> · regenerated from
  <span class="mono">{FACTS}</span>.</p>
</footer>
"""
    html = render("T4 — CDiT: the part that does the predicting", body, EXTRA_CSS)
    out = write(os.path.join(d, "t4_artifact.html"), html)
    # A second copy at a fixed path. The per-scene folder is what you keep; THIS is
    # what gets published, so the live URL stays the same however often the scene
    # changes. Publishing a new path would otherwise mint a new URL every time.
    write(os.path.join(OUT_ROOT, "latest.html"), html)

    if audit:
        # Which measurements did the probe take that this page never shows? Some
        # are consumed by hand when writing the Notion pages, so a long list is a
        # prompt to decide, not a failure.
        spare = facts_io.unused(f)
        print(f"\n  facts measured but not shown on this page ({len(spare)} of "
              f"{len(f)}):")
        for k in spare:
            print(f"    {k}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--audit"]
    build(args[0] if args else newest_scene(OUT_ROOT, FACTS),
          audit="--audit" in sys.argv)
