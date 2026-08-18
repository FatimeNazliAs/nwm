"""debug/t5/config.yaml is checked before anything expensive is built.

This is the stage's interface for the person using it, and it used to fail late:
`step` was validated after the first 250-pass loop and `filmstrip_steps` after
all three, so a typo cost a checkpoint load and a minute of GPU before it said
so. These tests pin the fast failure, and the messages that go with it.

    python debug/tests/test_t5_config.py
"""
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from debug.t5 import probe

GOOD = """
trajectory: jackal_2019-12-21-13-21-20_1_r00
time: 171
sec: 16
step: 125
filmstrip_steps: [249, 200, 150, 100, 60, 30, 10, 0]
seed: 0
compare_seed: 7
"""


def _read(text):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        return probe.read_config(path)
    finally:
        os.unlink(path)


def _rejects(text, word):
    try:
        _read(text)
    except SystemExit as e:
        assert word in str(e), f"the message should mention {word!r}, got: {e}"
    else:
        raise AssertionError(f"a config with a bad {word} was accepted")


def test_a_good_config_reads():
    cfg = _read(GOOD)
    assert cfg.sec == 16 and cfg.step == 125
    assert cfg.seed == 0 and cfg.compare_seed == 7
    assert cfg.raw["trajectory"].startswith("jackal"), "the scene keys stay available"


def test_defaults_apply_when_a_knob_is_absent():
    cfg = _read("trajectory: x\ntime: 5\n")
    assert cfg.sec == 1 and cfg.step == 125
    assert len(cfg.filmstrip_steps) == 8, "the default strip is still built"


def test_filmstrip_is_sorted_into_loop_order():
    """The page's captions say left to right; only one order makes them true."""
    cfg = _read(GOOD + "\nfilmstrip_steps: [0, 249, 100]\n")
    assert cfg.filmstrip_steps == (249, 100, 0), cfg.filmstrip_steps


def test_duplicates_in_the_filmstrip_collapse():
    cfg = _read(GOOD + "\nfilmstrip_steps: [249, 249, 100, 0]\n")
    assert cfg.filmstrip_steps == (249, 100, 0)


def test_out_of_range_knobs_are_refused():
    _rejects(GOOD + "\nstep: 250\n", "step")
    _rejects(GOOD + "\nstep: -1\n", "step")
    _rejects(GOOD + "\nsec: 32\n", "sec")
    _rejects(GOOD + "\nsec: 0\n", "sec")
    _rejects(GOOD + "\nfilmstrip_steps: [249, 300]\n", "filmstrip_steps")


def test_a_filmstrip_too_short_to_tell_a_story_is_refused():
    _rejects(GOOD + "\nfilmstrip_steps: [249, 0]\n", "filmstrip_steps")


def test_a_knob_that_is_not_a_number_is_refused():
    _rejects(GOOD + "\nstep: middle\n", "step")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
