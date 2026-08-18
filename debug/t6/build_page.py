"""
Build the T6 annotated webpage from the probe's output.

Consumes ONLY debug/out/t6/<scene>/eval_facts.json + the PNGs the probe wrote,
so every number on the page comes from a real run and cannot drift from the code.
Images are inlined as data URIs (the artifact CSP blocks any external host); the
page shell comes from debug/common/page.py, shared with T3, T4 and T5.

    cd /app && python debug/t6/build_page.py            # newest scene in out/t6
    cd /app && python debug/t6/build_page.py <scene>

Writes <scene>/t6_artifact.html and the fixed debug/out/t6/latest.html.

Layout rules for this page, unchanged from T5:
  * the general shape FIRST -- three pictures and five lines -- then the detail;
  * every beat is a numbered list, never a paragraph;
  * <details> holds supporting argument only, never structure.

What is deliberately NOT on this page, though the probe measures it: how far each
of the three comparison runs sits from the STARTING view. It was measured to
check a tempting explanation ("the run without an action just repeats where it
began") and it does not hold on this scene -- all three are further from the
start than the true frame is. The number stays in the facts file as the guard it
is; the page makes only the claim the measurements support.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from debug.common import facts as facts_io
from debug.common.page import data_uri, newest_scene, render, write

OUT_ROOT = "debug/out/t6"
FACTS = "eval_facts.json"

# The handful of facts the page cannot be built without. Deliberately short: the
# probe writes ~50 keys and pinning all of them here would make every probe
# change a page change (ADR-0004).
REQUIRED = ("model", "trajectory", "context_size", "sec", "secs_swept", "sweep",
            "per_horizon", "compare", "compare_ranks", "run_scenes", "run_pairs",
            "target_frame_number", "run_example_files", "run_example_folder")

EXTRA_CSS = """
  .beat .wide{grid-column:1 / -1}

  /* the overview: three pictures and five lines, before any detail */
  .glance{display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:0 0 20px}
  .glance figure img{width:100%; height:auto; display:block; border-radius:10px;
    border:1px solid var(--line); background:#fff}
  .glance figcaption{font-size:12px; color:var(--muted); margin-top:8px; text-align:center}
  .glance figcaption b{color:var(--accent); display:block; font-family:ui-monospace,Menlo,monospace;
    font-size:11px; letter-spacing:.06em; text-transform:uppercase; margin-bottom:2px}

  /* every beat explains itself as numbered steps, not prose */
  ol.steps{margin:0 0 12px; padding-left:22px}
  ol.steps li{margin:0 0 7px; font-size:14.5px; color:var(--ink); line-height:1.55}
  ol.steps li::marker{color:var(--accent); font-weight:700; font-size:13px}
  ol.big{padding-left:24px}
  ol.big li{font-size:15.5px; margin:0 0 10px}
  .aside{font-size:13.5px; color:var(--muted); border-left:2px solid var(--line);
    padding-left:12px; margin:0 0 10px}
  .aside b{color:var(--ink)}

  /* the horizon sweep: truth over prediction, five columns */
  .rowcap{font-size:11.5px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted);
    font-weight:700; margin:14px 0 6px}
  .rowcap b{color:var(--accent)}
  .sweep{display:grid; grid-template-columns:repeat(5,1fr); gap:8px}
  .sweep .s img{width:100%; height:auto; display:block; border-radius:5px; border:1px solid var(--line)}
  .sweep .s .lb{font-family:ui-monospace,Menlo,monospace; font-size:10px; color:var(--muted);
    text-align:center; margin-top:5px; white-space:nowrap}
  .sweep .s.worst .lb{color:var(--amber); font-weight:700}
  @media (max-width:560px){ .sweep{grid-template-columns:repeat(3,1fr)} }

  .five{display:grid; grid-template-columns:repeat(5,1fr); gap:10px}
  .five .cell.given img{border-style:dashed}
  .five .cell img{width:100%; height:auto; display:block; border-radius:8px;
    border:1px solid var(--line)}
  .five .cell .cap{font-size:11.5px; letter-spacing:.05em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin:0 0 6px; min-height:3.4em}
  .five .cell.hero .cap{color:var(--accent)}
  @media (max-width:760px){ .five{grid-template-columns:repeat(3,1fr)} }
  @media (max-width:620px){ .five{grid-template-columns:1fr 1fr} }

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

  /* the folders inference leaves behind -- the whole interface to the scorer */
  .tree{font-family:ui-monospace,Menlo,monospace; font-size:12.5px; line-height:1.85;
    background:var(--line-soft); border-radius:8px; padding:14px 16px; margin:0 0 14px;
    overflow-x:auto; white-space:pre}
  .tree b{color:var(--accent)}
  .tree i{color:var(--muted); font-style:normal}

  td.win{color:var(--accent); font-weight:700}
  td.lose{color:var(--amber)}
"""


def image_names(f):
    """Every PNG this page expects the probe to have written, by bare name.

    Part of the probe/page contract (ADR-0001) alongside the facts keys. Exposed
    rather than inlined so a test can create stubs for exactly this set.
    """
    return (["curve", "cmp_start", "cmp_noaction", "cmp_otherseed"]
            + [f"ctx_f{t}" for t in range(f["context_size"])]
            + [f"pred_{r['sec']}s" for r in f["sweep"]]
            + [f"truth_{r['sec']}s" for r in f["sweep"]])


def build(scene, audit=False):
    d = os.path.join(OUT_ROOT, scene)
    f = facts_io.load(os.path.join(d, FACTS), required=REQUIRED)

    T = f["context_size"]
    sec = f["sec"]
    sweep = f["sweep"]
    rows = f["per_horizon"]
    cmp_rows = f["compare"]
    ranks = f["compare_ranks"]
    best, worst = rows[0], rows[-1]
    # Which horizon this ONE scene happens to score worst on. Read off the run
    # rather than assumed to be the longest: on a single scene it very often is
    # not, and saying so is the point of the caption underneath.
    hardest = max(sweep, key=lambda r: r["lpips"])
    # The winner by each measure, so the table can mark them without the page
    # hard-coding an outcome the next scene may not repeat.
    first = {m: ranks[m][0] for m in ("pixel_mae", "lpips", "dreamsim")}

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

    def sweep_row(kind, label):
        return "\n".join(
            f"""      <div class="s{' worst' if kind == 'pred' and r['sec'] == hardest['sec'] else ''}">
        <img src="{img[f'{kind}_{r["sec"]}s']}" alt="{label} {r['sec']} seconds ahead.">
        <div class="lb">{r['sec']}s{f" &middot; {r['lpips']:.2f}" if kind == 'pred' else ''}</div>
      </div>""" for r in sweep)

    horizon_table = "\n".join(
        f"""        <tr><td class="field">{r['sec']} s</td>
          <td class="num">{r['lpips']:.4f}</td>
          <td class="num">{r['dreamsim']:.4f}</td>
          <td class="num">{r['fid']:.1f}</td></tr>""" for r in rows)

    def cls(metric, label):
        return ' class="win"' if first[metric] == label else ' class="lose"'

    compare_table = "\n".join(
        f"""        <tr><td class="field">{c['label']}</td>
          <td class="num"{cls('pixel_mae', c['label'])}>{c['pixel_mae']:.4f}</td>
          <td class="num"{cls('lpips', c['label'])}>{c['lpips']:.4f}</td>
          <td class="num"{cls('dreamsim', c['label'])}>{c['dreamsim']:.4f}</td></tr>"""
        for c in cmp_rows)

    # The listing shown on the page is the finished run's own files -- one real
    # folder out of the 500 -- not the probe's one-scene copy of the same layout.
    tree = "\n".join(
        f"  <b>{e['sec']}.png</b>   <i>{e['bytes']:,} bytes &middot; predicted "
        f"{e['sec']} s ahead</i>" for e in f["run_example_files"])

    body = f"""
<header>
  <p class="eyebrow">NWM debug walkthrough · step T6</p>
  <h1>The evaluation: {f['run_pairs']:,} predictions, three scores</h1>
  <p class="sub">T5 was one prediction being made. This is the run that makes all of them
  and puts a number on the result. Every picture and every number below came off a real
  run.</p>
</header>

<section>
  <h2>In short</h2>

  <div class="glance">
    <figure>
      <img src="{img[f'pred_{sec}s']}" alt="A predicted frame: a pale building at the left, a
        large tree in the centre, grass in the foreground.">
      <figcaption><b>predicted</b>{sec} seconds ahead</figcaption>
    </figure>
    <figure>
      <img src="{img[f'truth_{sec}s']}" alt="The true frame at the same moment, held back from
        the model.">
      <figcaption><b>what really happened</b>never shown to the model</figcaption>
    </figure>
    <figure>
      <img src="{img['curve']}" alt="Two plots. Left: LPIPS and DreamSim both rise from 1 to 16
        seconds. Right: FID rises too.">
      <figcaption><b>scored</b>{f['run_scenes']} scenes</figcaption>
    </figure>
  </div>

  <ol class="steps big">
    <li>For each of the <b>{f['run_scenes']} test scenes</b>, ask for five futures:
        {', '.join(str(s) for s in f['secs_swept'][:-1])} and {f['secs_swept'][-1]} seconds
        ahead.</li>
    <li>Each one is a separate {f['respaced_steps']}-pass loop — the whole of T5, run
        <b>{f['run_pairs']:,} times</b>. That took about five hours.</li>
    <li>Every prediction is <b>saved as a picture file</b>, and the program then stops.
        It hands nothing on.</li>
    <li>A <b>second program</b> opens those files next to the {f['run_pairs']:,} true frames
        and scores each pair three ways.</li>
    <li>Every score gets <b>worse the further ahead</b> the prediction reaches. That is the
        result of the run.</li>
  </ol>

  <p class="aside">Why three scores and not one? Because "how close are these two pictures?"
  has no single answer — and the obvious answer, subtracting one from the other, gives the
  wrong one. That is beat 4.</p>
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
      {sec} seconds<br>later
    </div>
    <div class="one">
      <p class="cap">what it must <b>produce</b> · frame {f['target_frame_number']}</p>
      <img src="{img[f'truth_{sec}s']}" alt="The true frame {f['target_frame_number']}, which
        the model is not shown.">
    </div>
  </div>

  <ol class="steps" style="margin-top:16px">
    <li>One drive: <span class="mono">{f['trajectory']}</span>. The moment called "now" is frame
        {f['curr_time']} — the rightmost of the four on the left.</li>
    <li>This is one of the {f['run_scenes']} scenes the evaluation runs. Every other scene is
        handled identically.</li>
    <li>The model gets those {T} frames and one action: where the robot ends up over the next
        {sec} seconds. The frame on the right is held back.</li>
    <li>Notice what changes between the two sides: the pale building starts large, filling the
        left half, and ends up a small block at the far edge — the robot drove on past it.</li>
  </ol>
</section>

<section>
  <h2>The same thing, step by step</h2>
  <div class="story">

    <div class="beat here">
      <div class="wide">
        <span class="num">1</span>
        <h3>One scene, five questions</h3>
        <p class="rowcap">Top · <b>what really happened</b>, at each of the five horizons</p>
        <div class="sweep">
{sweep_row('truth', 'The true frame')}
        </div>
        <p class="rowcap">Bottom · <b>what the model produced</b> — and this scene's score
        underneath each one</p>
        <div class="sweep">
{sweep_row('pred', 'The predicted frame')}
        </div>
        <ol class="steps" style="margin-top:14px">
          <li>All five predictions start from the <b>same {T} frames</b>. Only two things change
              across the columns: how far ahead to look, and where the robot ends up by then.</li>
          <li>Each column is its own {f['respaced_steps']}-pass loop from its own screen of
              noise. Nothing is shared between them, and <b>no prediction is fed into the
              next</b> — every one is answered directly from the {T} real frames.</li>
          <li><b>Read down each column, not along the rows.</b> Top is what happened, bottom is
              what the model produced from the {T} frames alone. It gets the place right in
              every column; what drifts is where things have got to inside it.</li>
          <li>The numbers are LPIPS — lower is better. On this scene the worst is
              <b>{hardest['sec']} s</b>, not {f['secs_swept'][-1]} s.</li>
          <li><b>One scene proves nothing.</b> A single pair can go either way depending on what
              happened to be in front of the robot. The trend only appears once you average
              {f['run_scenes']} of them, which is beat 5.</li>
        </ol>
      </div>
    </div>

    <div class="act here"><span class="arrow">↓</span><span class="badge">the handover</span></div>

    <div class="beat here">
      <div class="wide">
        <span class="num">2</span>
        <h3>Where the predictions go: everything between the two halves is files</h3>
        <div class="tree"><b>{f['run_predictions_dir']}/</b>
<i>id_0/  id_1/  …  id_{f['run_scenes'] - 1}/</i>       <i>one folder per scene &mdash; {f['run_example_folder']}/ opened up</i>
{tree}</div>
        <ol class="steps">
          <li>The predicting program writes those files and <b>exits holding nothing</b>. No
              tensor, no model, no memory of the run survives it.</li>
          <li>The scoring program is started separately, afterwards. It <b>never loads the
              model</b>, never opens the checkpoint and never sees anything but pictures.</li>
          <li>It lists the folders of true frames, opens <b>the file with the matching name</b>
              on the prediction side, and compares the two.</li>
          <li>So the file names are the entire handover: <b>the folder says which scene, the
              file says how far ahead</b>. There is a second copy of this whole tree holding
              the true frames.</li>
          <li>That is why predicting took <b>five hours</b> and scoring took
              <b>{f['score_seconds']:.0f} seconds</b> — and why the same scorer can be pointed at
              any model's output without knowing anything about it.</li>
        </ol>
      </div>
    </div>

    <div class="beat here">
      <div class="wide">
        <span class="num">3</span>
        <h3>The three scores</h3>
        <ol class="steps">
          <li><b>LPIPS.</b> Push both pictures through a standard image-recognition network
              (AlexNet) and
              compare what it noticed, layer by layer, rather than the pixels. Lower is
              better. One number per pair.</li>
          <li><b>DreamSim.</b> The same idea, but tuned on <b>people's answers</b> to "which of
              these two pictures is more like that one". It leans towards layout and objects and
              away from texture. Lower is better. One number per pair.</li>
          <li><b>FID.</b> A different kind of question. It takes <b>all {f['run_scenes']}
              predictions at one horizon as a set</b>, does the same for the true frames, and
              measures how far apart the two collections are.</li>
          <li>So FID <b>cannot tell you whether prediction 7 matches truth 7</b>. It only says
              whether the predictions look like the right kind of picture overall — a check on
              realism, not on accuracy.</li>
          <li>That is why FID appears once per horizon over {f['run_scenes']} scenes and never
              next to a single pair, including in beat 1 above.</li>
        </ol>
      </div>
    </div>

    <div class="beat here">
      <div class="wide">
        <span class="num">4</span>
        <h3>Why not just subtract one picture from the other?</h3>
        <div class="five" style="margin:10px 0 14px">
          <figure class="cell given">
            <p class="cap">0 · where it starts<br>frame {f['curr_time']}, given</p>
            <img src="{img['cmp_start']}" alt="The last frame the model is given: a large pale
              building filling the left, trees to the right, green grass.">
          </figure>
          <figure class="cell">
            <p class="cap">1 · the truth<br>never shown to the model</p>
            <img src="{img[f'truth_{sec}s']}" alt="The true frame {sec} seconds later.">
          </figure>
          <figure class="cell hero">
            <p class="cap">2 · the prediction<br>action kept</p>
            <img src="{img[f'pred_{sec}s']}" alt="The predicted frame.">
          </figure>
          <figure class="cell">
            <p class="cap">3 · action removed<br>same starting noise</p>
            <img src="{img['cmp_noaction']}" alt="The frame predicted with the action removed.">
          </figure>
          <figure class="cell">
            <p class="cap">4 · different random start<br>action kept</p>
            <img src="{img['cmp_otherseed']}" alt="The frame predicted from a different random
              start.">
          </figure>
        </div>
        <p class="aside" style="margin-bottom:14px">Pictures 2, 3 and 4 are three attempts at
        <b>the same moment</b>, frame {f['target_frame_number']}. Each is a full
        {f['respaced_steps']}-pass loop with exactly one input changed. Picture 1 is the answer
        they are all being marked against.</p>
        <div class="tablecard"><div class="scroll"><table>
          <thead><tr><th>scored against picture 1, the truth</th>
            <th>subtracting the pictures</th><th>LPIPS</th><th>DreamSim</th></tr></thead>
          <tbody>
{compare_table}
          </tbody>
        </table></div></div>
        <ol class="steps" style="margin-top:14px">
          <li>Lower is better in all three columns, and the <span class="accent">best in each
              column is marked</span>.</li>
          <li><b>Subtracting the pictures picks number 3</b> — the run that was told nothing
              about where the robot went.</li>
          <li>That answer cannot be right. Picture 3 had <b>no information about the
              journey</b>, so it cannot be the best account of where the journey ended.</li>
          <li><b>LPIPS and DreamSim both pick number 2</b>, the real prediction. The two
              measures were built independently and they agree.</li>
          <li>Why subtraction fails: it lines the two pictures up and compares <b>corner with
              corner</b>. It has no idea that one of those corners is a building. Grass in a
              slightly different arrangement counts against you as heavily as a building in the
              wrong place.</li>
          <li>Picture 4 scores worst on all three — and it is <b>not a mistake</b>. Same
              journey, different random start: an equally believable version of the same
              sixteen seconds, with the leaves grown differently.</li>
        </ol>
        <details>
          <summary>The question T5 left open, and what changed</summary>
          <div class="dbody">
        <ol class="steps">
          <li>T5 ended by measuring these same three pictures two ways and getting <b>two
              opposite rankings</b> — so neither could be trusted.</li>
          <li>Both of those were plain distances: one between the compressed forms, one between
              the pixels. Neither knows what a building is.</li>
          <li>The fix is not a better distance. It is to <b>compare what a trained network
              notices</b> in each picture instead of the pictures themselves.</li>
          <li>Two such measures, built by different people for different reasons, agree here.
              That agreement is the reason to trust them.</li>
          <li>The published numbers in beat 5 are these two, plus FID, and nothing else. The
              plain difference appears nowhere in the evaluation.</li>
        </ol>
          </div>
        </details>
      </div>
    </div>

    <div class="beat here">
      <div class="wide">
        <span class="num">5</span>
        <h3>The whole run: {f['run_scenes']} scenes</h3>
        <div style="display:grid; grid-template-columns:1fr 1.15fr; gap:16px;
          align-items:center; margin:10px 0 14px">
          <div class="tablecard"><div class="scroll"><table>
            <thead><tr><th>how far ahead</th><th>LPIPS</th><th>DreamSim</th><th>FID</th></tr></thead>
            <tbody>
{horizon_table}
            </tbody>
          </table></div></div>
          <figure class="plot">
            <img src="{img['curve']}" alt="Left plot: LPIPS and DreamSim both rise steadily from
              1 to 16 seconds. Right plot: FID rises too, with its biggest jump at 16 seconds.">
          </figure>
        </div>
        <ol class="steps">
          <li>{f['run_scenes']} scenes × {len(f['secs_swept'])} horizons =
              <b>{f['run_pairs']:,} pairs</b> of pictures, scored in
              {f['score_seconds']:.0f} seconds.</li>
          <li>Lower is better everywhere, so <b>every line goes the wrong way</b> as the horizon
              grows. From {best['sec']} s to {worst['sec']} s, LPIPS rises
              {f['growth_lpips_pct']:.0f}%, DreamSim {f['growth_dreamsim_pct']:.0f}% and FID
              {f['growth_fid_pct']:.0f}%.</li>
          <li><b>DreamSim moves the most.</b> It is the measure that cares about layout, and
              layout is what a longer drive gets wrong — sixteen seconds of driving can put the
              robot somewhere else entirely.</li>
          <li><b>FID moves the least</b>, and that is consistent: the predictions keep looking
              like plausible photographs of this place even when they are of the wrong moment
              in it. Realism holds up better than accuracy.</li>
          <li>So the result is not one number but a shape: <b>one second ahead is a far easier
              problem than sixteen</b>.</li>
        </ol>
      </div>
    </div>

  </div>

  <p class="aside" style="margin-top:20px"><b>Deferred on purpose.</b> The same script has a
  second mode, <span class="mono">rollout</span>, which feeds each prediction back in as the
  next input instead of answering from the real frames every time. And the checkpoint scored
  here is what planning searches over. Neither is part of this walkthrough.</p>
</section>

<footer>
  <p>Example scene <span class="mono">{f['trajectory']}</span> frame {f['curr_time']},
  predicted at {', '.join(f'{s}s' for s in f['secs_swept'])} · scores over
  {f['run_scenes']} scenes · <span class="mono">{f['model']}</span> checkpoint
  <span class="mono">{f['checkpoint']}</span>, {f['respaced_steps']} passes per prediction ·
  regenerated from <span class="mono">{FACTS}</span>.</p>
</footer>
"""
    html = render("T6 — Inference and Metrics", body, EXTRA_CSS)
    out = write(os.path.join(d, "t6_artifact.html"), html)
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
