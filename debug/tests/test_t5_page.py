"""Build the T5 page from a recorded facts file -- no GPU, no checkpoint.

Same shape as test_t4_page.py, against `debug/t5/fixtures/loop_facts.json`: a
real probe run, kept so that page work does not need a model. Building the page
from it takes well under a second, which is the point of the probe/page split
(ADR-0001) and of the facts file being a validated interface (ADR-0004).

    python debug/tests/test_t5_page.py        # plain, no pytest needed
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
from debug.t5 import build_page

FIXTURE = os.path.join(ROOT, "debug", "t5", "fixtures", "loop_facts.json")

# The smallest thing data_uri() will accept: a real 1x1 PNG. The page only
# base64s these, so content is irrelevant -- only that the files exist.
STUB_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg==")


def _scene_dir(tmp):
    """A scene folder that looks exactly like one a probe just wrote."""
    scene = "fixture_scene__t0"
    d = os.path.join(tmp, scene)
    os.makedirs(d)
    shutil.copy(FIXTURE, os.path.join(d, build_page.FACTS))
    f = json.load(open(FIXTURE))
    for name in build_page.image_names(f):
        with open(os.path.join(d, f"{name}.png"), "wb") as fh:
            fh.write(STUB_PNG)
    return scene


def test_page_builds_from_fixture():
    with tempfile.TemporaryDirectory() as tmp:
        scene = _scene_dir(tmp)
        original = build_page.OUT_ROOT
        build_page.OUT_ROOT = tmp
        try:
            out = build_page.build(scene)
        finally:
            build_page.OUT_ROOT = original

        html = open(out).read()
        assert "<title>" in html, "page lost its title"
        # Numbers must come from the facts file, never from the template.
        f = json.load(open(FIXTURE))
        assert f["trajectory"] in html, "the scene is not named on the page"
        assert str(f["respaced_steps"]) in html, "the step count is not on the page"
        assert str(f["step_opened"]) in html, "the opened-up pass is not on the page"
        assert os.path.exists(os.path.join(tmp, "latest.html")), \
            "the fixed publish path was not written"


def test_page_needs_only_pictures_the_probe_writes():
    """The picture half of the probe/page contract, checked without a GPU.

    Before the manifest existed this could not be asserted at all: the test built
    its stubs from image_names() itself, so it was checking the page against its
    own assumption and could never notice a probe that stopped writing a panel.
    """
    f = json.load(open(FIXTURE))
    manifest = f.get("_images")
    assert manifest, "the fixture predates image manifests -- re-run debug/t5/probe.py"
    unwritten = [n for n in build_page.image_names(f) if n not in manifest]
    assert not unwritten, \
        f"the page asks for pictures the probe never writes: {unwritten}"


def test_both_filmstrip_rows_are_shown():
    """The point of T5 is the real loop, so both rows must come off the run.

    T4's filmstrip was built with q_sample and labelled as borrowed. This page
    must show x AND pred_xstart at every filmstrip step -- if a future edit drops
    one of the rows, the page silently stops making its own argument.
    """
    f = json.load(open(FIXTURE))
    names = build_page.image_names(f)
    for s in f["filmstrip"]:
        assert f"loop_x_s{s['step']}" in names, f"x row missing step {s['step']}"
        assert f"loop_guess_s{s['step']}" in names, f"guess row missing step {s['step']}"


def test_missing_required_fact_is_a_clear_error():
    """A page meeting an older probe's output must say which key is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, build_page.FACTS)
        data = json.load(open(FIXTURE))
        del data["filmstrip"]
        json.dump(data, open(path, "w"))
        try:
            facts_io.load(path, required=build_page.REQUIRED)
        except facts_io.FactsError as e:
            assert "filmstrip" in str(e), "the error must name the missing key"
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
