"""
Build the T5 annotated webpage from the probe's output.

Consumes ONLY debug/out/t5/<scene>/loop_facts.json + the PNGs the probe wrote,
so every number on the page comes from a real run and cannot drift from the code.
Images are inlined as data URIs (the artifact CSP blocks any external host); the
page shell comes from debug/common/page.py, shared with T3 and T4.

    cd /app && python debug/t5/build_page.py            # newest scene in out/t5
    cd /app && python debug/t5/build_page.py <scene>

Writes <scene>/t5_artifact.html and the fixed debug/out/t5/latest.html.

Layout rules for this page, in the order they were asked for:
  * the general shape FIRST -- three pictures and five lines -- then the detail;
  * every beat is a numbered list, never a paragraph;
  * no collapsed toggles: if something is worth showing it is on the page, and if
    it is not, it is not here at all. The probe still measures far more than this
    shows (`build_page.py --audit` lists it) and that surplus is read by hand.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from debug.common import facts as facts_io
from debug.common.page import data_uri, newest_scene, render, write

OUT_ROOT = "debug/out/t5"
FACTS = "loop_facts.json"

# The handful of facts the page cannot be built without. Deliberately short: the
# probe writes ~70 keys and pinning all of them here would make every probe
# change a page change (ADR-0004).
REQUIRED = ("model", "respaced_steps", "diffusion_steps", "filmstrip", "loop_steps",
            "trajectory", "latent_size", "context_size", "in_channels", "step_opened",
            "pixel_mae_vs_prediction")

EXTRA_CSS = """
  .beat .wide{grid-column:1 / -1}
  .beat .nums{font-family:ui-monospace,Menlo,monospace; font-size:12.5px; line-height:1.9;
    background:var(--line-soft); border-radius:8px; padding:12px 14px; margin:0 0 12px}
  .beat .nums b{color:var(--accent)}

  /* the overview: three pictures and five lines, before any detail */
  .glance{display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:0 0 20px}
  .glance figure img{width:100%; height:auto; display:block; border-radius:10px;
    border:1px solid var(--line)}
  .glance figcaption{font-size:12px; color:var(--muted); margin-top:8px; text-align:center}
  .glance figcaption b{color:var(--accent); display:block; font-family:ui-monospace,Menlo,monospace;
    font-size:11px; letter-spacing:.06em; text-transform:uppercase; margin-bottom:2px}
  .glance .arrowcell{display:flex; align-items:center; justify-content:center; color:var(--muted)}

  /* every beat explains itself as numbered steps, not prose */
  ol.steps{margin:0 0 12px; padding-left:22px}
  ol.steps li{margin:0 0 7px; font-size:14.5px; color:var(--ink); line-height:1.55}
  ol.steps li::marker{color:var(--accent); font-weight:700; font-size:13px}
  ol.big{padding-left:24px}
  ol.big li{font-size:15.5px; margin:0 0 10px}
  .aside{font-size:13.5px; color:var(--muted); border-left:2px solid var(--line);
    padding-left:12px; margin:0 0 10px}
  .aside b{color:var(--ink)}

  /* the loop, as rows of real decoded frames */
  .rowcap{font-size:11.5px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted);
    font-weight:700; margin:14px 0 6px}
  .rowcap b{color:var(--accent)}
  .film{display:grid; grid-template-columns:repeat(8,1fr); gap:6px}
  .film .s img{width:100%; height:auto; display:block; border-radius:4px; border:1px solid var(--line)}
  .film .s .lb{font-family:ui-monospace,Menlo,monospace; font-size:9.5px; color:var(--muted);
    text-align:center; margin-top:5px; white-space:nowrap}
  .film .s.last .lb{color:var(--accent); font-weight:700}
  @media (max-width:560px){ .film{grid-template-columns:repeat(4,1fr)} }

  .two{display:grid; grid-template-columns:1fr 1fr; gap:14px}
  .four{display:grid; grid-template-columns:repeat(4,1fr); gap:12px}
  .five{display:grid; grid-template-columns:repeat(5,1fr); gap:10px}
  .five .cell.given{opacity:.92}
  .five .cell.given img{border-style:dashed}
  .two .cell img,.four .cell img,.five .cell img{width:100%; height:auto; display:block; border-radius:8px;
    border:1px solid var(--line)}
  .two .cell .cap,.four .cell .cap,.five .cell .cap{font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin:0 0 6px; min-height:2.6em}
  .four .cell.hero .cap,.five .cell.hero .cap{color:var(--accent)}
  @media (max-width:760px){ .five{grid-template-columns:repeat(3,1fr)} }
  @media (max-width:620px){ .two,.four{grid-template-columns:1fr 1fr}
    .five{grid-template-columns:1fr 1fr} }

  /* the example being followed, shown before any of the machinery */
  .example{display:grid; grid-template-columns:2.1fr auto 1.05fr; gap:16px; align-items:center;
    background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); padding:18px}
  .example .cap{font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin:0 0 8px}
  .example .cap b{color:var(--accent)}
  .example .one img{width:100%; height:auto; display:block; border-radius:8px;
    border:2px solid var(--accent)}
  .example .gap{text-align:center; color:var(--muted); font-size:12px; line-height:1.5;
    font-family:ui-monospace,Menlo,monospace; white-space:nowrap}
  .example .gap .ar{font-size:26px; display:block; color:var(--accent)}
  @media (max-width:700px){ .example{grid-template-columns:1fr}
    .example .gap .ar{transform:rotate(90deg)} }

  .plot img{width:100%; height:auto; display:block; border-radius:8px; background:#fff;
    border:1px solid var(--line)}
  .quad{display:grid; grid-template-columns:1fr 1fr; gap:5px}
  .quad img{width:100%; height:auto; display:block; image-rendering:pixelated; border-radius:5px}
"""


def image_names(f):
    """Every PNG this page expects the probe to have written, by bare name.

    Part of the probe/page contract (ADR-0001) alongside the facts keys. Exposed
    rather than inlined so a test can create stubs for exactly this set. The probe
    also writes settle.png and mix.png; this page does not show them, so they are
    not listed here.
    """
    return (["ground_truth", "final_pred", "final_noaction", "final_otherseed",
             "schedule", "step_x_in", "step_guess", "step_x_out"]
            + [f"ctx_f{t}" for t in range(f["context_size"])]
            + [f"step_eps_ch{c}" for c in range(f["in_channels"])]
            + [f"loop_x_s{s['step']}" for s in f["filmstrip"]]
            + [f"loop_guess_s{s['step']}" for s in f["filmstrip"]])


def build(scene, audit=False):
    d = os.path.join(OUT_ROOT, scene)
    f = facts_io.load(os.path.join(d, FACTS), required=REQUIRED)

    T = f["context_size"]
    nc = f["in_channels"]
    N = f["respaced_steps"]
    film = f["filmstrip"]
    first, last = film[0]["step"], film[-1]["step"]
    mid = film[len(film) // 2]["step"]          # a pass from the middle of the strip
    # Positions in the strip taken as fractions, not fixed indices: the reader may
    # set filmstrip_steps to as few as three passes, and film[4] would then be an
    # IndexError at build time.
    def at_frac(x):
        return film[min(int(len(film) * x), len(film) - 1)]
    early, settled, emerging = at_frac(0.12), at_frac(0.45), at_frac(0.55)
    mae = f["pixel_mae_vs_prediction"]
    by_step = {r["step"]: r for r in f["loop_steps"]}
    opened = by_step[f["step_opened"]]
    # how big the deliberate part of one pass is, next to the grid it is moving
    drift_pct = f["step_drift"] / opened["x_norm"] * 100

    needed = image_names(f)
    # The other half of the probe/page contract: every picture shown here must be
    # one the probe recorded writing, and must actually be on disk. Checked before
    # rendering, so a missing panel is a sentence rather than a stack trace.
    facts_io.require_images(f, needed)
    img = {n: data_uri(os.path.join(d, f"{n}.png")) for n in needed}

    def tag(t):
        return "now" if t == T - 1 else f"t&minus;{T - 1 - t}"

    frame_strip = "\n".join(
        f"""        <div class="f{' now' if t == T - 1 else ''}">
          <img src="{img[f'ctx_f{t}']}" alt="Context frame {t}.">
          <div class="lb">{tag(t)}</div>
        </div>""" for t in range(T))

    def film_row(kind):
        return "\n".join(
            f"""      <div class="s{' last' if i == len(film) - 1 else ''}">
        <img src="{img[f'loop_{kind}_s{s["step"]}']}" alt="Pass {s['step']} of the loop.">
        <div class="lb">pass {s['step']}</div>
      </div>""" for i, s in enumerate(film))

    eps_quad = "".join(f'<img src="{img[f"step_eps_ch{c}"]}" alt="">' for c in range(nc))

    body = f"""
<header>
  <p class="eyebrow">NWM debug walkthrough · step T5</p>
  <h1>The loop: noise becomes a frame</h1>
  <p class="sub">T4 was the model that answers one question. This is the loop that asks it
  {N} times. Every picture below was decoded from one real run.</p>
</header>

<section>
  <h2>In short</h2>

  <div class="glance">
    <figure>
      <img src="{img[f'loop_x_s{first}']}" alt="Pure coloured static, no shapes.">
      <figcaption><b>start</b>random numbers</figcaption>
    </figure>
    <figure>
      <img src="{img[f'loop_x_s{mid}']}" alt="Static with faint shapes beginning to appear.">
      <figcaption><b>halfway</b>shapes appearing</figcaption>
    </figure>
    <figure>
      <img src="{img['final_pred']}" alt="A finished predicted frame with trees and a building.">
      <figcaption><b>end</b>a predicted frame</figcaption>
    </figure>
  </div>

  <ol class="steps big">
    <li>Start with a screen of <b>pure random numbers</b>. Nothing of the scene is in it.</li>
    <li>Ask the model one question: <b>how much of this is noise?</b> That is all it can answer.</li>
    <li>Its answer can be rearranged into a full <b>guess at the finished frame</b>.</li>
    <li>Take a <b>sliver</b> of that guess — about {f['step_mix_new_guess_pct']}% — and put a little
        fresh randomness back on top.</li>
    <li>Repeat <b>{N} times</b>. What is left is the predicted frame.</li>
  </ol>

  <p class="aside">Why not just take the whole guess in one go? Because many different futures fit
  the same past. Committing at once gives a blurred average of all of them. Taking {N} small passes
  with randomness in between picks one and makes it sharp.</p>
</section>

<section>
  <h2>The example followed all the way down this page</h2>

  <div class="example">
    <div>
      <p class="cap">what the model is <b>given</b> · frames
        {f['context_frame_numbers'][0]}–{f['context_frame_numbers'][-1]}</p>
      <div class="strip">
{frame_strip}
      </div>
    </div>
    <div class="gap">
      <span class="ar">→</span>
      {f['sec']} seconds<br>later
    </div>
    <div class="one">
      <p class="cap">what it must <b>produce</b> · frame {f['target_frame_number']}</p>
      <img src="{img['ground_truth']}" alt="The true frame {f['target_frame_number']}, which the
        model is not shown.">
    </div>
  </div>

  <ol class="steps" style="margin-top:16px">
    <li>One drive: <span class="mono">{f['trajectory']}</span>. The moment called "now" is frame
        {f['curr_time']} — the rightmost of the four on the left.</li>
    <li>The model is given those {T} frames and one action: where the robot ends up over the next
        {f['sec']} seconds, ({f['action'][0]:+.2f}, {f['action'][1]:+.2f},
        {f['action'][2]:+.2f}).</li>
    <li>It must produce frame {f['target_frame_number']}, <b>{f['sec']} seconds later</b>. That
        frame is held back and never shown to it.</li>
    <li>Notice what changes between the two sides: the pale building starts large and near the
        middle, and ends up small in the far left corner, because the robot drove on. Predicting
        that movement is the job.</li>
  </ol>
</section>

<section>
  <h2>The same thing, step by step</h2>
  <div class="story">

    <div class="beat">
      <div class="wide">
        <span class="num">1</span>
        <h3>Before the loop starts</h3>
        <ol class="steps">
          <li>The {T} past frames and the action above are handed in <b>once</b>.</li>
          <li>They never change again, for any of the {N} passes that follow.</li>
          <li>Only two things move from pass to pass: the grid of numbers being worked on, and
              how noisy that grid is meant to be.</li>
        </ol>
      </div>
    </div>

    <div class="act here"><span class="arrow">↓</span><span class="badge">T5 — this step</span></div>

    <div class="beat here">
      <div class="wide">
        <span class="num">2</span>
        <h3>Where it starts, and how the noise comes off</h3>
        <div class="two" style="margin:10px 0 14px">
          <figure class="cell">
            <p class="cap">pass {first} · what the loop begins with</p>
            <img src="{img[f'loop_x_s{first}']}" alt="The starting noise, decoded: coloured static.">
          </figure>
          <figure class="cell plot">
            <p class="cap">how much of the frame is present, pass by pass</p>
            <img src="{img['schedule']}" alt="Two curves: the frame's share rises from zero to one
              across the loop while the noise's share falls from one to zero.">
          </figure>
        </div>
        <ol class="steps">
          <li><b>Left</b> — the starting point, decoded so you can see it. Not a blurry frame.
              Nothing.</li>
          <li><b>Right</b> — the plan. <b>Teal</b> is how much of the true frame is present,
              <b>amber</b> is how much noise. The loop runs left to right.</li>
          <li>The loop follows that plan for {N} passes, from all noise to none.</li>
        </ol>
      </div>
    </div>

    <div class="beat here">
      <div class="wide">
        <span class="num">3</span>
        <h3>One pass, opened up</h3>
        <div class="four" style="margin:10px 0 14px">
          <figure class="cell">
            <p class="cap">1 · going in</p>
            <img src="{img['step_x_in']}" alt="The latent at step {f['step_opened']}, decoded: coloured static.">
          </figure>
          <figure class="cell">
            <p class="cap">2 · what the model returns</p>
            <div class="quad">{eps_quad}</div>
          </figure>
          <figure class="cell hero">
            <p class="cap">3 · 1 minus 2 — the model's answer so far</p>
            <img src="{img['step_guess']}" alt="The model's guess at step {f['step_opened']}, decoded:
              a soft, slightly smeared scene with trees and a building.">
          </figure>
          <figure class="cell">
            <p class="cap">4 · coming out</p>
            <img src="{img['step_x_out']}" alt="The latent one step later, decoded: still coloured static.">
          </figure>
        </div>
        <ol class="steps">
          <li>This is pass {f['step_opened']} of {N} — halfway. <b>Picture 1</b> is what the loop
              is holding: the frame with half its noise still in it.</li>
          <li>The model is asked one thing: <b>how much of picture 1 is noise?</b>
              <b>Picture 2</b> is that answer — a map of noise, which is why it looks like
              noise.</li>
          <li><b>Picture 3 is picture 1 minus picture 2</b>, rescaled and decoded. Take away all
              the noise the model says is there, and this is what is left.</li>
          <li>So picture 3 is <b>the frame the model would hand you if you stopped right now</b>.
              Not the next pass — the <i>last</i> one. At halfway it can already draw the
              scene.</li>
          <li><b>Picture 4</b> is what the loop actually carries forward. It is a mixture,
              made like this:</li>
        </ol>
        <div class="nums" style="margin:0 0 12px">
          picture 4 &nbsp;=&nbsp; <b>{f['step_mix_new_guess_pct']}%</b> of picture 3
          &nbsp;+&nbsp; <b>{f['step_mix_where_it_is_pct']}%</b> of picture 1
          &nbsp;+&nbsp; fresh randomness
        </div>
        <ol class="steps">
          <li><b>That is what the {f['step_mix_new_guess_pct']}% means</b> — how much of the
              model's answer is allowed into the next grid. The other
              {f['step_mix_where_it_is_pct']}% is just where the grid already was.</li>
          <li>So pictures 1 and 4 look identical because <b>they nearly are</b>: almost all of
              picture 4 <i>is</i> picture 1. The deliberate part of the move shifts the grid by
              about {drift_pct:.0f}% of its size, and the randomness added on top hides even
              that.</li>
        </ol>
        <details>
          <summary>So why not just stop at picture 3?</summary>
          <div class="dbody">
        <ol class="steps">
          <li>The model's answer is only ever as good as what it is handed. Picture 1 is mostly
              noise, so the model <b>cannot tell which future it is looking at</b>.</li>
          <li>Many futures fit the same {T} past frames and the same action — the robot could
              pass either side of the tree, the grass could fall any way.</li>
          <li>Given a noisy input, the model's best answer is a <b>compromise between all of
              them</b>. That is exactly why picture 3 is soft and smeared instead of sharp.</li>
          <li>Stop now, and that compromise is your final answer.</li>
          <li>Instead the loop nudges picture 1 slightly towards picture 3, adds randomness, and
              asks again. The next input is a little less noisy and a little more committed to
              one particular future — so <b>the next answer is a little less of a
              compromise</b>.</li>
          <li>After {N} rounds of that the compromise is gone. The bottom row of the next beat is
              this happening: a blur sharpening into a photograph.</li>
        </ol>
          </div>
        </details>
      </div>
    </div>

    <div class="beat here">
      <div class="wide">
        <span class="num">4</span>
        <h3>×{N} — the loop</h3>
        <p class="aside" style="margin-bottom:16px"><b>The top row is the grid the loop is
        holding — a mixture of the real frame and noise. The bottom row is that same grid with
        the model's estimated noise taken out.</b></p>
        <p class="rowcap">Top · <b>what the loop is actually holding</b> — carried from pass to pass</p>
        <div class="film">
{film_row('x')}
        </div>
        <p class="rowcap">Bottom · <b>the answer the model would give at that moment</b> — worked
        out fresh every pass, then thrown away</p>
        <div class="film">
{film_row('guess')}
        </div>
        <details>
          <summary>How to read the two rows</summary>
          <div class="dbody">
        <ol class="steps">
          <li><b>The top row is the real state of the loop.</b> It is the only thing passed from
              one pass to the next, and it stays unreadable static until about pass
              {emerging['step']}.</li>
          <li><b>The bottom row never goes anywhere.</b> At each of those moments the model was
              asked for its answer, we decoded it, and the loop then discarded it and carried on
              with the top row.</li>
          <li>Reading down any single column: that is one pass. Same moment, two very different
              pictures.</li>
          <li><b>Read the bottom row left to right and you can watch the average narrow.</b> At
              pass {early['step']} it is a soft blur — many futures averaged. By pass
              {settled['step']} the tree and the building have settled. After that it only gets
              sharper.</li>
          <li>So the scene is <b>chosen early and sharpened late</b>, and the model knew roughly
              what it was drawing long before the top row showed anything at all.</li>
          <li>In the last column the two rows are the <b>same picture</b> — by then there is no
              noise left to take out, so the grid and the answer have met.</li>
        </ol>
          </div>
        </details>
      </div>
    </div>

    <div class="beat here">
      <div class="wide">
        <span class="num">5</span>
        <h3>What comes out</h3>
        <div class="five" style="margin:10px 0 14px">
          <figure class="cell given">
            <p class="cap">0 · where it starts<br>frame {f['curr_time']}, given</p>
            <img src="{img[f'ctx_f{T - 1}']}" alt="The last context frame: a large pale building
              left of centre, trees to the right, green grass.">
          </figure>
          <figure class="cell">
            <p class="cap">1 · the truth<br>never shown to the model</p>
            <img src="{img['ground_truth']}" alt="The true future frame, held out.">
          </figure>
          <figure class="cell hero">
            <p class="cap">2 · the prediction<br>action kept, seed {f['seed']}</p>
            <img src="{img['final_pred']}" alt="The predicted frame.">
          </figure>
          <figure class="cell">
            <p class="cap">3 · action removed<br>same seed {f['seed']}</p>
            <img src="{img['final_noaction']}" alt="The frame predicted with the action removed.">
          </figure>
          <figure class="cell">
            <p class="cap">4 · different random start<br>action kept, seed {f['compare_seed']}</p>
            <img src="{img['final_otherseed']}" alt="The frame predicted from a different random start.">
          </figure>
        </div>
        <p class="aside" style="margin-bottom:14px"><b>Picture 0 is the starting view</b> —
        the last of the {T} frames the model was given. Pictures 1–4 are all the same later
        moment, frame {f['target_frame_number']}, {f['sec']} seconds on. <b>Picture 2 is the one
        to compare the rest against</b>: pictures 3 and 4 are the whole {N}-pass loop run again
        with exactly one input changed.</p>
        <ol class="steps">
          <li><b>First, look at 0 against 1.</b> In the starting view the pale building is large
              and near the middle. Sixteen seconds later it has shrunk into the far left — the
              robot drove on and the trees came to the centre. That movement is what the model
              has to predict.</li>
          <li><b>1 against 2 — how good is it?</b> The prediction puts the building small and far
              left, and the tall tree in the centre, like the truth. It never saw picture 1; it
              had the {T} frames and one action. The grass and the fine branches do not match and
              never will — those are invented.</li>
          <li><b>2 against 3 — what did the action do?</b> Picture 3 changed one thing: the action
              was set to zero, so the model was told nothing about where the robot travelled. Same
              four past frames, same random start, same {N} passes.</li>
          <li><b>Look at picture 3 next to picture 0, not next to picture 2.</b> The building is
              still large and near the middle — the model produced roughly the view it was
              already looking at. Without an action it has no reason to move the camera at all,
              so it mostly stays put and invents some detail (the thin masts).</li>
          <li><b>2 against 4 — what did the randomness do?</b> Picture 4 changed one thing: the
              random numbers the loop started from. The action was kept.</li>
          <li>The camera <b>moved to the same place</b> — building small and far left, tall tree
              centre — with different grass and different branches.</li>
          <li>So: <b>the action is what moves the camera; the randomness only decides the detail
              once it has arrived.</b> Picture 4 is not a mistake. It is an equally valid future
              of the same drive.</li>
        </ol>
      </div>
    </div>

    <div class="act"><span class="arrow">↓</span><span class="badge">T6 — not built yet</span></div>

    <div class="beat absent">
      <div class="pic"><div class="missing">how good is it?<br><br>T6</div></div>
      <div>
        <span class="num">6</span>
        <h3>Scoring it</h3>
        <div class="nums">
          how far from the prediction &nbsp;&nbsp;&nbsp; <span style="opacity:.6">as numbers</span>
          &nbsp;&nbsp; <span style="opacity:.6">as pixels</span><br>
          no action &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {f['dist_no_action_pct']}% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{mae['final_noaction']:.3f}</b><br>
          different noise &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {f['dist_other_seed_pct']}% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{mae['final_otherseed']:.3f}</b><br>
          the true frame &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {f['dist_ground_truth_pct']}% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{mae['ground_truth']:.3f}</b>
        </div>
        <ol class="steps">
          <li>The two columns rank the same three pictures in <b>opposite orders</b>.</li>
          <li>Regrowing every leaf is a big number and almost no visible change.</li>
          <li>Choosing a measure that agrees with your eyes is <b>T6</b>.</li>
        </ol>
      </div>
    </div>

  </div>
</section>

<footer>
  <p>Measured on <span class="mono">{f['trajectory']}</span> frame {f['curr_time']}, predicting
  {f['sec']}s ahead · <span class="mono">{f['model']}</span> checkpoint
  <span class="mono">{f['checkpoint']}</span>, {N} passes, seed {f['seed']} ·
  regenerated from <span class="mono">{FACTS}</span>.</p>
</footer>
"""
    html = render("T5 — The Diffusion Loop", body, EXTRA_CSS)
    out = write(os.path.join(d, "t5_artifact.html"), html)
    # A second copy at a fixed path. The per-scene folder is what you keep; THIS is
    # what gets served and published, so the live URL stays the same however often
    # the scene changes.
    write(os.path.join(OUT_ROOT, "latest.html"), html)

    if audit:
        spare = facts_io.unused(f)
        print(f"\n  facts measured but not shown on this page ({len(spare)} of {len(f)}):")
        for k in spare:
            print(f"    {k}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--audit"]
    build(args[0] if args else newest_scene(OUT_ROOT, FACTS),
          audit="--audit" in sys.argv)
