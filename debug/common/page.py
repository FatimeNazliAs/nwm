"""The published page's shell, shared by every walkthrough stage.

T1..T6 are one series, so the look is fixed: teal accent, amber for "held out /
cost", panel cards, and the vertical "story" of real images. Only the content
changes per stage. This module owns that shell so a stage's build_page.py is
nothing but the story it has to tell.

  BASE_CSS   layout, type, tables, notes, figures -- used by every page
  STORY_CSS  the numbered beats + labelled arrows + <details> drawer
  render()   wraps a body in <title> + <style>

The stage that owns the current beat marks it `class="beat here"` (and the
arrow into it `class="act here"`); every other beat is a stage owned by a
different T-step, dashed or greyed out.
"""
import base64
import os


BASE_CSS = """
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

  /* three-up comparison */
  .trip{display:grid; grid-template-columns:repeat(3,1fr); gap:14px}
  .trip .cell .cap{font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin:0 0 6px}
  @media (max-width:640px){ .trip{grid-template-columns:1fr} }

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


STORY_CSS = """
  /* ---- the story: a vertical sequence of real images ---- */
  .story{display:flex; flex-direction:column; gap:0}

  .beat{display:grid; grid-template-columns:270px 1fr; gap:22px; align-items:center;
    background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); padding:16px 18px}
  .beat.here{border-color:var(--accent); border-width:2px}
  .beat.absent{background:var(--line-soft); box-shadow:none; border-style:dashed}
  .beat .pic img{width:100%; height:auto; display:block; border-radius:8px; border:1px solid var(--line)}
  .beat .quad{display:grid; grid-template-columns:1fr 1fr; gap:5px}
  .beat .quad img{image-rendering:pixelated; border-radius:5px}
  .beat .num{font-family:ui-monospace,Menlo,monospace; font-size:12px; font-weight:800; color:#fff;
    background:var(--ink); border-radius:6px; padding:2px 8px; display:inline-block; margin-bottom:8px}
  .beat.here .num{background:var(--accent)}
  .beat h3{margin:0 0 6px; font-size:19px; letter-spacing:-.01em}
  .beat p{margin:0 0 8px; font-size:14.5px; color:var(--muted)}
  .beat .shape{font-family:ui-monospace,Menlo,monospace; font-size:13.5px; font-weight:700;
    font-variant-numeric:tabular-nums}
  .beat .missing{height:150px; border:2px dashed var(--line); border-radius:8px; display:flex;
    align-items:center; justify-content:center; text-align:center; color:var(--muted);
    font-family:ui-monospace,Menlo,monospace; font-size:12.5px; padding:12px; line-height:1.5}
  @media (max-width:620px){ .beat{grid-template-columns:1fr} }

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
  .act.here .what{color:var(--accent)}
  .act .src{font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--muted)}
  .act .badge{font-family:ui-monospace,Menlo,monospace; font-size:10px; font-weight:800;
    letter-spacing:.06em; padding:2px 7px; border-radius:5px; background:var(--line-soft); color:var(--muted)}
  .act.here .badge{background:var(--accent); color:#fff}
  .act .arrow{color:var(--muted); font-size:19px; line-height:1}

  details{background:var(--panel); border:1px solid var(--line); border-radius:12px;
    box-shadow:var(--shadow); padding:0; margin-top:14px; overflow:hidden}
  summary{cursor:pointer; padding:13px 18px; font-weight:650; font-size:14.5px; list-style:none;
    display:flex; align-items:center; gap:10px}
  summary::-webkit-details-marker{display:none}
  summary::before{content:"\\25B8"; color:var(--accent); font-size:12px}
  details[open] summary::before{content:"\\25BE"}
  details[open] summary{border-bottom:1px solid var(--line)}
  .dbody{padding:16px 18px}
  .dbody > :first-child{margin-top:0}
  .dbody .tablecard{box-shadow:none}
"""


def data_uri(path):
    """Inline a PNG -- the artifact CSP blocks every external host."""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def newest_scene(out_root, facts_name):
    """The most recently probed scene under debug/out/<stage>/."""
    cands = [d for d in os.listdir(out_root)
             if os.path.isfile(os.path.join(out_root, d, facts_name))]
    if not cands:
        raise SystemExit(f"No scene with {facts_name} under {out_root}/ -- run the stage's probe.py first.")
    return max(cands, key=lambda d: os.path.getmtime(os.path.join(out_root, d, facts_name)))


def render(title, body, extra_css=""):
    """The full page: the artifact wrapper supplies <html>/<head>/<body>."""
    return f"""<title>{title}</title>
<style>{BASE_CSS}{STORY_CSS}{extra_css}</style>
<div class="wrap">
{body}
</div>
"""


def write(path, html):
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path}  ({len(html)/1024:.0f} KB)")
    return path
