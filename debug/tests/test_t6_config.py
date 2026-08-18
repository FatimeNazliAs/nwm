"""debug/t6/config.yaml is checked before anything expensive is built.

T6 is the stage where late validation costs the most: a checkpoint, a VAE, seven
250-pass predictions, three metric networks and 2500 scored pairs, most of it
before a bad `sec` or a mistyped folder would ever be reached. These tests pin
the fast failure and the messages that go with it.

    python debug/tests/test_t6_config.py
"""
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from debug.t6 import probe

GOOD = """
trajectory: jackal_2019-12-21-13-21-20_1_r00
time: 171
sec: 16
seed: 0
compare_seed: 7
score_batch_size: 64
"""


def _read(text, tmp):
    """Read a config whose two folders exist, so only `text` is under test."""
    gt = os.path.join(tmp, "gt")
    exp = os.path.join(tmp, "exp")
    os.makedirs(gt, exist_ok=True)
    os.makedirs(exp, exist_ok=True)
    text += f"\npredictions_dir: {exp}\nground_truth_dir: {gt}\n"
    path = os.path.join(tmp, "config.yaml")
    with open(path, "w") as fh:
        fh.write(text)
    return probe.read_config(path)


def _rejects(text, word):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _read(text, tmp)
        except SystemExit as e:
            assert word in str(e), f"the message should mention {word!r}, got: {e}"
        else:
            raise AssertionError(f"a config with a bad {word} was accepted")


def test_a_good_config_reads():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _read(GOOD, tmp)
        assert cfg.sec == 16 and cfg.seed == 0 and cfg.compare_seed == 7
        assert cfg.score_batch_size == 64
        assert cfg.raw["trajectory"].startswith("jackal"), "the scene keys stay available"


def test_defaults_apply_when_a_knob_is_absent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _read("trajectory: x\ntime: 5\n", tmp)
        assert cfg.sec == 16 and cfg.score_batch_size == 64
        assert cfg.seed == 0 and cfg.compare_seed == 7


def test_only_the_five_horizons_the_evaluation_uses_are_accepted():
    """`sec: 3` would run and produce a picture nothing can be compared against.

    The finished run on disk holds 1, 2, 4, 8 and 16 only, so any other value
    silently leaves the comparison in beat 4 scoring against a frame the
    evaluation never made.
    """
    _rejects(GOOD + "\nsec: 3\n", "sec")
    _rejects(GOOD + "\nsec: 0\n", "sec")
    _rejects(GOOD + "\nsec: 32\n", "sec")
    for good in (1, 2, 4, 8, 16):
        with tempfile.TemporaryDirectory() as tmp:
            assert _read(GOOD + f"\nsec: {good}\n", tmp).sec == good


def test_a_knob_that_is_not_a_number_is_refused():
    _rejects(GOOD + "\nsec: sixteen\n", "sec")
    _rejects(GOOD + "\nscore_batch_size: lots\n", "score_batch_size")


def test_a_batch_size_of_zero_is_refused():
    _rejects(GOOD + "\nscore_batch_size: 0\n", "score_batch_size")


def test_a_folder_that_is_not_there_is_refused_by_name():
    """The finished run is an input to this stage, not something it produces."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        with open(path, "w") as fh:
            fh.write(GOOD + f"\npredictions_dir: {tmp}/nope\nground_truth_dir: {tmp}\n")
        try:
            probe.read_config(path)
        except SystemExit as e:
            assert "predictions_dir" in str(e), "the error must name the knob"
            assert "Phase 4" in str(e), "the error must say where the folder comes from"
        else:
            raise AssertionError("a missing run folder was accepted")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
