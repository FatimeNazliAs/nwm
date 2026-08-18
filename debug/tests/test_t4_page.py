"""Build the T4 page from a recorded facts file -- no GPU, no checkpoint.

This is the test the probe/page split (ADR-0001) was supposed to buy and did not
until the facts file became a real interface. It exercises build_page.py end to
end in well under a second, against `debug/t4/fixtures/cdit_facts.json`: a real
probe run, kept so that page work does not need a model.

    python debug/tests/test_t4_page.py        # plain, no pytest needed
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
from debug.t4 import build_page

FIXTURE = os.path.join(ROOT, "debug", "t4", "fixtures", "cdit_facts.json")

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
        assert str(f["depth"]) in html, "the block count is not on the page"
        assert "latest.html" != out, "build() should return the per-scene copy"
        assert os.path.exists(os.path.join(tmp, "latest.html")), \
            "the fixed publish path was not written"


def test_page_needs_only_pictures_the_probe_writes():
    """The picture half of the probe/page contract, checked without a GPU."""
    f = json.load(open(FIXTURE))
    manifest = f.get("_images")
    assert manifest, "the fixture predates image manifests -- re-run debug/t4/probe.py"
    unwritten = [n for n in build_page.image_names(f) if n not in manifest]
    assert not unwritten, \
        f"the page asks for pictures the probe never writes: {unwritten}"


def test_missing_required_fact_is_a_clear_error():
    """A page meeting an older probe's output must say which key is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cdit_facts.json")
        data = json.load(open(FIXTURE))
        del data["depth"]
        json.dump(data, open(path, "w"))
        try:
            facts_io.load(path, required=build_page.REQUIRED)
        except facts_io.FactsError as e:
            assert "depth" in str(e), "the error must name the missing key"
            assert "probe" in str(e).lower(), "the error must say how to fix it"
        else:
            raise AssertionError("a missing required fact was accepted")


def test_unused_reports_what_the_page_ignores():
    f = facts_io.load(FIXTURE)
    f["model"]
    spare = facts_io.unused(f)
    assert "depth" in spare, "an unread key should be reported as unused"
    assert "model" not in spare, "a read key must not be reported as unused"
    assert "_schema_version" not in spare, "internals must not be reported"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
