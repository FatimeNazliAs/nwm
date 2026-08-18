"""Build the T6 page from a recorded facts file -- no GPU, no checkpoint.

Same shape as test_t5_page.py, against `debug/t6/fixtures/eval_facts.json`: a
real probe run, kept so that page work does not need a model, a five-hour
inference run or the 5000 scored PNGs.

    python debug/tests/test_t6_page.py        # plain, no pytest needed
"""
import base64
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from debug.common import facts as facts_io
from debug.t6 import build_page

FIXTURE = os.path.join(ROOT, "debug", "t6", "fixtures", "eval_facts.json")

# The smallest thing data_uri() will accept: a real 1x1 PNG. The page only
# base64s these, so content is irrelevant -- only that the files exist.
STUB_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg==")


def _scene_dir(tmp, facts=None):
    """A scene folder that looks exactly like one a probe just wrote."""
    scene = "fixture_scene__t0"
    d = os.path.join(tmp, scene)
    os.makedirs(d)
    if facts is None:
        shutil.copy(FIXTURE, os.path.join(d, build_page.FACTS))
        facts = json.load(open(FIXTURE))
    else:
        json.dump(facts, open(os.path.join(d, build_page.FACTS), "w"))
    for name in build_page.image_names(facts):
        with open(os.path.join(d, f"{name}.png"), "wb") as fh:
            fh.write(STUB_PNG)
    return scene


def _build(tmp, scene):
    original = build_page.OUT_ROOT
    build_page.OUT_ROOT = tmp
    try:
        return build_page.build(scene)
    finally:
        build_page.OUT_ROOT = original


def test_page_builds_from_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        out = _build(tmp, _scene_dir(tmp))
        html = open(out).read()
        assert "<title>" in html, "page lost its title"
        # Numbers must come from the facts file, never from the template.
        f = json.load(open(FIXTURE))
        assert f["trajectory"] in html, "the scene is not named on the page"
        assert str(f["run_scenes"]) in html, "the scene count is not on the page"
        assert f"{f['run_pairs']:,}" in html, "the pair count is not on the page"
        assert os.path.exists(os.path.join(tmp, "latest.html")), \
            "the fixed publish path was not written"


def test_page_needs_only_pictures_the_probe_writes():
    """The picture half of the probe/page contract, checked without a GPU."""
    f = json.load(open(FIXTURE))
    manifest = f.get("_images")
    assert manifest, "the fixture predates image manifests -- re-run debug/t6/probe.py"
    unwritten = [n for n in build_page.image_names(f) if n not in manifest]
    assert not unwritten, \
        f"the page asks for pictures the probe never writes: {unwritten}"


def test_every_horizon_is_shown_both_ways():
    """Beat 1 is the argument that all five horizons come off the same frames.

    Drop either row and the page still renders, but stops making that argument
    -- so both are pinned here rather than left to the eye.
    """
    f = json.load(open(FIXTURE))
    names = build_page.image_names(f)
    assert len(f["sweep"]) == len(f["secs_swept"]), "a horizon went missing from the sweep"
    for r in f["sweep"]:
        assert f"truth_{r['sec']}s" in names, f"truth row missing {r['sec']}s"
        assert f"pred_{r['sec']}s" in names, f"prediction row missing {r['sec']}s"


def test_the_winning_measure_is_read_from_the_facts_not_hard_coded():
    """The point of beat 4 is an outcome, so the outcome must not be in the HTML.

    This is the test that matters most on this page. It flips the recorded
    ranking and asserts the marked winner moves with it -- if a future edit ever
    writes "LPIPS picks the prediction" into the template, the page would keep
    saying so on a scene where it is false.
    """
    real = json.load(open(FIXTURE))
    flipped = json.loads(json.dumps(real))
    for metric in ("pixel_mae", "lpips", "dreamsim"):
        flipped["compare_ranks"][metric] = list(reversed(real["compare_ranks"][metric]))

    with tempfile.TemporaryDirectory() as tmp:
        html_real = open(_build(tmp, _scene_dir(tmp))).read()
    with tempfile.TemporaryDirectory() as tmp:
        html_flip = open(_build(tmp, _scene_dir(tmp, flipped))).read()

    def marked(html, label):
        """Does `label`'s row of the comparison table carry a win class?

        Anchored on the row's own <td class="field"> cell, not on the bare
        label: the same words appear in the caption above the table, and
        matching those was this test passing for the wrong reason.
        """
        cell = f'<td class="field">{label}</td>'
        assert cell in html, f"{label!r} is not a row of the comparison table"
        row = html[html.index(cell):]
        row = row[:row.index("</tr>")]
        return 'class="win"' in row

    winner = real["compare_ranks"]["lpips"][0]
    loser = real["compare_ranks"]["lpips"][-1]
    assert marked(html_real, winner), "the recorded winner is not marked"
    assert marked(html_flip, loser), \
        "reversing the recorded ranking did not move the marking -- the page hard-codes it"


def test_missing_required_fact_is_a_clear_error():
    """A page meeting an older probe's output must say which key is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, build_page.FACTS)
        data = json.load(open(FIXTURE))
        del data["per_horizon"]
        json.dump(data, open(path, "w"))
        try:
            facts_io.load(path, required=build_page.REQUIRED)
        except facts_io.FactsError as e:
            assert "per_horizon" in str(e), "the error must name the missing key"
            assert "probe" in str(e).lower(), "the error must say how to fix it"
        else:
            raise AssertionError("a missing required fact was accepted")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
