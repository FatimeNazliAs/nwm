"""
Build the T2 annotated webpage from the probe's output.

Consumes ONLY debug/out/t2/<scene>/{scene_facts.json, ctx_f*.png, tgt_*s.png,
action_transform.png}, so every number/picture comes from a real run and cannot
drift from the code. Uses the shared shell in debug/common/page.py.

The page shows the data ONE TYPE AT A TIME — the frames, the future frames, the
action (before/after), the time — each with its own big visual and bullets.

    cd /app && python debug/t2/build_page.py            # newest scene in out/t2
    cd /app && python debug/t2/build_page.py sample0    # a specific scene

Writes <scene>/t2_artifact.html and out/t2/latest.html.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from debug.common.page import data_uri, newest_scene, render, write

OUT_ROOT = "debug/out/t2"
FACTS = "scene_facts.json"

EXTRA_CSS = """
  .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}
  .grid5{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}
  .grid4 .cell img,.grid5 .cell img{width:100%;height:auto;display:block;border-radius:7px;border:1px solid var(--line)}
  .cell .lb{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);text-align:center;margin-top:6px;line-height:1.3}
  .cell.now .lb{color:var(--accent);font-weight:700}
  ul.tight{margin:16px 0 0;padding-left:20px}
  ul.tight li{margin:0 0 8px;font-size:14.5px}
  ul.tight li b{color:var(--ink)}
  .datatype{margin-top:40px}
  .datatype h2 .n{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;
    background:var(--accent);color:#fff;border-radius:7px;font-size:13px;font-weight:800;margin-right:4px}
  .datatype h2::before{display:none}
"""


def fmt(x, dec=2):
    return f"{x:+.{dec}f}".replace("-", "−")


def plain(x, dec=2):
    return f"{x:.{dec}f}".replace("-", "−")


def cell(img_src, label, is_now=False):
    return (f'<div class="cell{" now" if is_now else ""}">'
            f'<img src="{img_src}" alt="{label}"><div class="lb">{label}</div></div>')


def build(scene):
    d = os.path.join(OUT_ROOT, scene)
    f = json.load(open(os.path.join(d, FACTS)))

    def uri(name):
        return data_uri(os.path.join(d, name))

    now, n = f["curr_time"], f["n_frames"]
    T, H = f["context_size"], f["len_traj_pred"]
    cf = f["context_frames"]
    pf, pl = f["pred_first"], f["pred_last"]
    pos, yaw, d1 = f["position_now"], f["yaw_now"], f["delta1"]
    tt = f["time_table"]

    # strip of the 4 context frames
    ctx_cells = "".join(
        cell(uri(f"ctx_f{t}.png"),
             f"{'now' if t == T - 1 else f't−{T - 1 - t}'}<br>frame {cf[t]}",
             is_now=(t == T - 1))
        for t in range(T))

    # strip of the 5 scored future frames
    tgt_cells = "".join(
        cell(uri(f"tgt_{r['sec']}s.png"), f"@ {r['sec']}s<br>frame {r['target_frame']}")
        for r in tt)

    time_rows = "\n".join(
        f'<tr><td class="num">{r["sec"]}</td><td class="num">{r["steps"]}</td>'
        f'<td class="num">{r["rel_t"]:.3f}</td><td class="num">'
        f'({fmt(r["curr_delta"][0])}, {fmt(r["curr_delta"][1])}, {fmt(r["curr_delta"][2])})</td></tr>'
        for r in tt)

    body = f"""
<header>
  <p class="eyebrow">NWM debug walkthrough · step T2</p>
  <h1>The data: what one sample is</h1>
  <p class="sub">The four parts of one sample — the frames in, the future held out, the action,
  and the time — each shown with the real pictures from a single RECON drive.</p>
  <div class="chips" style="margin-top:20px">
    <div class="chip"><div class="k">dataset</div><div class="v">RECON · test</div></div>
    <div class="chip"><div class="k">trajectory</div><div class="v" style="font-size:11px;word-break:break-all">{f['trajectory']}</div></div>
    <div class="chip"><div class="k">"now" is</div><div class="v">frame {now}</div></div>
    <div class="chip"><div class="k">camera</div><div class="v">fisheye · 4 fps</div></div>
  </div>
</header>

<section>
  <h2>What goes in — and what's held out</h2>
  <div class="tablecard scroll">
    <table>
      <thead><tr><th>the four parts</th><th>given to the model?</th><th>what it is</th></tr></thead>
      <tbody>
        <tr><td class="field">the frames</td><td class="accent">✓ input</td><td>the 4 recent frames — the past up to "now"</td></tr>
        <tr><td class="field">the action</td><td class="accent">✓ input</td><td>how to move — (dx, dy, dyaw) per step</td></tr>
        <tr><td class="field">the time</td><td class="accent">✓ input</td><td>how far ahead to imagine — the scalar rel_t</td></tr>
        <tr><td class="field">the future frames</td><td class="amberc">✗ held out</td><td>the real answer — only used to score the guess</td></tr>
      </tbody>
    </table>
  </div>
  <p style="margin-top:12px"><b>The difference:</b> the three <b>inputs</b> steer the prediction; the <b>held-out</b> frames are the answer key we grade it against afterwards — the model never sees them.</p>
</section>

<section class="datatype">
  <h2><span class="n">1</span> The frames it sees &nbsp;<span style="color:var(--accent)">· input</span></h2>
  <figure class="figcard">
    <div class="grid4">{ctx_cells}</div>
  </figure>
  <ul class="tight">
    <li><b>The 4 most recent frames</b>, oldest → newest — the only thing the model actually sees.</li>
    <li>The last one is <b>"now"</b> (frame {now}); the others are 0.25 s apart before it.</li>
    <li>Each is RGB, 224×224 px, values rescaled to <span class="mono">[−1, 1]</span>.</li>
    <li><span class="mono">obs_image ({T}, 3, 224, 224)</span></li>
  </ul>
</section>

<section class="datatype">
  <h2><span class="n">2</span> The frames held out &nbsp;<span class="amberc">· the answer key</span></h2>
  <figure class="figcard">
    <div class="grid5">{tgt_cells}</div>
    <figcaption>The real future — <b>hidden</b> from the model, used only to score its guess. (These 5 are the horizons the eval checks; there are {H} in all.)</figcaption>
  </figure>
  <ul class="tight">
    <li><b>{H} real future frames</b> (here {pf}–{pl}) that actually happened next on this drive.</li>
    <li>The model never sees them — they are the ground truth we grade the prediction against.</li>
    <li><span class="mono">pred_image ({H}, 3, 224, 224)</span></li>
  </ul>
  <details>
    <summary>Why every sample carries the full 16 s — even to predict just 1 s</summary>
    <div class="dbody">
      <ul class="tight" style="margin-top:0">
        <li>The dataset <b>always</b> loads all {H} future frames + all {H} action steps, whatever horizon you ask for. That's why a scene needs {H} frames after "now" even for a 1 s query.</li>
        <li><b>Reason:</b> one loaded sample answers <b>all five</b> horizons (1, 2, 4, 8, 16 s) at once. To predict 8 s the eval just sums the first 32 steps and scores against the 8 s frame — the 9–16 s frames stay loaded but unused for that query.</li>
        <li>16 s = {H} frames is simply the largest horizon the eval scores; <span class="mono">len_traj_pred = {H}</span> is fixed by the trained checkpoint.</li>
        <li>Either way the model sees <b>no</b> ground-truth frame — "held out" holds for every horizon.</li>
      </ul>
    </div>
  </details>
</section>

<section class="datatype">
  <h2><span class="n">3</span> The action &nbsp;<span style="color:var(--accent)">· input</span></h2>
  <p>The action tells the model <b>how to move</b>, in the robot's own terms: <b>dx</b> forward,
  <b>dy</b> sideways, <b>dyaw</b> turn — one triple per step.</p>
  <figure class="figcard">
    <img src="{uri('action_transform.png')}" alt="Left: the drive seen on a map. Right: the same drive described from the robot's seat at now, starting at the origin and facing along +x.">
    <figcaption><b>Left:</b> where the robot actually drives, on a map. <b>Right:</b> the same drive described from the robot's seat at "now" — it starts at the origin, facing forward (+x). That right-hand description is the action.</figcaption>
  </figure>
  <ul class="tight">
    <li>Why rewrite it? A camera frame has no map on it, so "forward / sideways / turn" is the only description that matches what the model sees.</li>
    <li>One real step: <span class="mono">delta[1] = (dx&nbsp;{fmt(d1[0])}, dy&nbsp;{fmt(d1[1])}, dyaw&nbsp;{fmt(d1[2])})</span> — forward a bit, barely sideways, slight turn.</li>
  </ul>
  <details>
    <summary>How the action is computed (the mechanics)</summary>
    <div class="dbody">
      <ul class="tight" style="margin-top:0">
        <li>Raw on disk it's world coordinates — <span class="mono">position[{now}] = ({plain(pos[0])}, {plain(pos[1])})</span>, <span class="mono">yaw = {plain(yaw)}</span> — which mean nothing to the model.</li>
        <li><span class="mono">to_local_coords</span> rotates every future point into the robot's frame at "now" (origin = now, +x = heading) → the right-hand plot.</li>
        <li><b>dx, dy</b> are scaled by the 0.25 m waypoint spacing — small, tidy numbers the net trains on.</li>
        <li>Each row is <b>incremental</b> (the change from step i to i+1), not distance from the start.</li>
        <li><span class="mono">delta ({H}, 3)</span> = {H} of these little forward/sideways/turn steps — the whole drive.</li>
      </ul>
    </div>
  </details>
</section>

<section class="datatype">
  <h2><span class="n">4</span> The time &nbsp;<span style="color:var(--accent)">· input · how far ahead</span></h2>
  <p>The action from part 3 is <b>per-step</b> — {H} little moves. "Time" just decides <b>how many of them
  to add up</b>: that sum is <span class="mono">curr_delta</span>, the single move the model actually gets.</p>
  <div class="tablecard scroll">
    <table>
      <thead><tr><th>seconds</th><th>= steps</th><th>rel_t</th><th>curr_delta = sum of those steps (dx, dy, dyaw)</th></tr></thead>
      <tbody>
{time_rows}
      </tbody>
    </table>
  </div>
  <ul class="tight">
    <li>1 s = 4 steps (4 fps): sum the first 4. 8 s = 32 steps; 16 s = all {H}.</li>
    <li><span class="mono">curr_delta</span> = those transformed steps added up = one total (dx, dy, dyaw) for the horizon.</li>
    <li>A scalar <span class="mono">rel_t = steps / 128</span> also tells the model how far ahead.</li>
    <li><b>"One-shot"</b> = a single jump to N s — not chained short steps (that is <i>rollout</i>, deferred).</li>
  </ul>
</section>

<details>
  <summary>The exact tensors, and the T1 vs T2 shape difference</summary>
  <div class="dbody">
    <div class="tablecard scroll">
      <table>
        <thead><tr><th>field</th><th>shape</th><th>what it is</th></tr></thead>
        <tbody>
          <tr><td class="field">idx</td><td class="num">(1,)</td><td>sample index, logging only</td></tr>
          <tr><td class="field">obs_image</td><td class="num">({T}, 3, 224, 224)</td><td>the {T} context frames — input</td></tr>
          <tr><td class="field">pred_image</td><td class="num">({H}, 3, 224, 224)</td><td>{H} future frames — held-out ground truth</td></tr>
          <tr><td class="field">delta</td><td class="num">({H}, 3)</td><td>the action: {H} per-step (dx, dy, dyaw)</td></tr>
        </tbody>
      </table>
    </div>
    <p style="margin:14px 0 0">Shapes look one number shorter than T1 because T1 read them inside the pipeline,
    where a DataLoader adds a leading <b>batch</b> axis. Same data. <span class="mono">curr_delta</span> from T1
    isn't returned here — it's built by summing <span class="mono">delta</span> (section 4).</p>
  </div>
</details>

<details>
  <summary>Two ways to pick which scene this page shows</summary>
  <div class="dbody">
    <ul class="tight" style="margin-top:0">
      <li><b>by index</b> — <span class="mono">sample: N</span> picks row N of the 500 pre-chosen eval samples.</li>
      <li><b>by folder</b> — <span class="mono">trajectory: &lt;name&gt;</span> + <span class="mono">time: &lt;frame&gt;</span> loads any RECON drive at a chosen "now" frame.</li>
    </ul>
    <p style="color:var(--muted);font-size:13.5px;margin-bottom:0">Edit <span class="mono">debug/t2/config.yaml</span>. Full run commands are on the Notion <b>Implementation</b> page.</p>
  </div>
</details>

<footer>
  <p>Measured on <span class="mono">{f['trajectory']}</span> frame <span class="mono">{now}</span>
  · regenerated from <span class="mono">scene_facts.json</span>.</p>
</footer>
"""
    html = render("T2 — The data: what one sample is", body, extra_css=EXTRA_CSS)
    write(os.path.join(d, "t2_artifact.html"), html)
    return write(os.path.join(OUT_ROOT, "latest.html"), html)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else newest_scene(OUT_ROOT, FACTS))
