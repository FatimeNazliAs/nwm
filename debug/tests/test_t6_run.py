"""T6's two helpers, against fakes and temporary folders. No GPU, no run.

`check_run` and `per_horizon` both stand between the probe and a minute of
scoring, and both fail in ways that are silent rather than loud:

  * an incomplete run scores fewer pairs than it claims to;
  * a mistyped meter name reads a fresh, empty meter instead of the scores
    `evaluate` actually recorded, because MetricLogger.meters is a defaultdict.

The second is the reason per_horizon exists as a function at all, so it is
tested against a fake logger built the same way the real one is.

    python debug/tests/test_t6_run.py
"""
import os
import sys
import tempfile
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from debug.t6 import probe

SECS = (1, 2, 4, 8, 16)


class FakeMeter:
    def __init__(self, value):
        self.global_avg = value


class FakeLogger:
    """The little of dist.MetricLogger that per_horizon touches.

    `meters` is a defaultdict in the real thing (distributed.py:166), which is
    exactly why a wrong key returns an empty meter instead of raising -- so the
    fake is a defaultdict too, or the test would be easier than reality.
    """

    def __init__(self, filled):
        self.meters = defaultdict(lambda: FakeMeter(0.0))
        self.meters.update({k: FakeMeter(v) for k, v in filled.items()})


def _filled(dataset="recon", eval_name="time"):
    out = {}
    for i, sec in enumerate(SECS):
        out[f"{dataset}_{eval_name}_lpips_{sec}s"] = 0.10 + i
        out[f"{dataset}_{eval_name}_dreamsim_{sec}s"] = 0.20 + i
        out[f"{dataset}_{eval_name}_fid_{sec}s"] = 30.0 + i
    return out


def _run_tree(root, scenes=3, secs=SECS, skip=()):
    """A pair of folders shaped like a finished run."""
    gt = os.path.join(root, "gt")
    exp = os.path.join(root, "exp")
    for i in range(scenes):
        for side in (gt, exp):
            d = os.path.join(side, f"id_{i}")
            os.makedirs(d, exist_ok=True)
            for sec in secs:
                if (side, i, sec) in skip:
                    continue
                open(os.path.join(d, f"{sec}.png"), "wb").close()
    return gt, exp


# ---------------------------------------------------------------- per_horizon

def test_scores_are_read_under_the_names_evaluate_writes():
    rows = probe.per_horizon(FakeLogger(_filled()), "recon", "time", SECS)
    assert [r["sec"] for r in rows] == list(SECS), "horizons out of order"
    assert rows[0]["lpips"] == 0.10 and rows[-1]["fid"] == 34.0
    assert rows[2]["dreamsim"] == 2.20, "a row picked up another horizon's score"


def test_a_meter_the_scoring_loop_never_filled_is_an_error_not_a_zero():
    """The failure this function exists to prevent: a silent row of zeros."""
    partial = _filled()
    del partial["recon_time_dreamsim_8s"]
    try:
        probe.per_horizon(FakeLogger(partial), "recon", "time", SECS)
    except SystemExit as e:
        assert "dreamsim" in str(e) and "8" in str(e), \
            f"the error must name the metric and the horizon, got: {e}"
    else:
        raise AssertionError("a missing meter was read as a score")


def test_the_dataset_and_eval_type_are_part_of_the_name():
    """Scoring another dataset's meters would otherwise pass silently."""
    try:
        probe.per_horizon(FakeLogger(_filled("recon", "time")), "recon", "rollout", SECS)
    except SystemExit as e:
        assert "rollout" in str(e) or "lpips" in str(e)
    else:
        raise AssertionError("meters from the wrong eval type were accepted")


# ------------------------------------------------------------------ check_run

def test_a_complete_run_is_counted():
    with tempfile.TemporaryDirectory() as tmp:
        gt, exp = _run_tree(tmp, scenes=4)
        assert probe.check_run(gt, exp, SECS) == (4, 20)


def test_a_missing_prediction_is_named_before_anything_is_scored():
    with tempfile.TemporaryDirectory() as tmp:
        gt, exp = _run_tree(tmp, scenes=3, skip=[(os.path.join(tmp, "exp"), 1, 8)])
        try:
            probe.check_run(gt, exp, SECS)
        except SystemExit as e:
            assert "id_1/8.png" in str(e), f"the error must name the file, got: {e}"
            assert "prediction" in str(e), "the error must say which side is short"
        else:
            raise AssertionError("an incomplete run was accepted")


def test_a_whole_missing_scene_folder_is_named():
    with tempfile.TemporaryDirectory() as tmp:
        gt, exp = _run_tree(tmp, scenes=3)
        for sec in SECS:
            os.remove(os.path.join(exp, "id_2", f"{sec}.png"))
        os.rmdir(os.path.join(exp, "id_2"))
        try:
            probe.check_run(gt, exp, SECS)
        except SystemExit as e:
            assert "id_2" in str(e) and "folder" in str(e)
        else:
            raise AssertionError("a missing scene folder was accepted")


def test_an_empty_run_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        gt, exp = os.path.join(tmp, "gt"), os.path.join(tmp, "exp")
        os.makedirs(gt), os.makedirs(exp)
        try:
            probe.check_run(gt, exp, SECS)
        except SystemExit as e:
            assert "id_0" in str(e), "the error should say what it expected to find"
        else:
            raise AssertionError("an empty run was accepted")


# ------------------------------------------------------------------ pixel_mae

def test_pixel_mae_reads_the_files_and_ignores_any_alpha():
    with tempfile.TemporaryDirectory() as tmp:
        a = np.zeros((4, 4, 3), dtype=np.float32)
        b = np.ones((4, 4, 3), dtype=np.float32)
        pa, pb = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
        plt.imsave(pa, a)
        plt.imsave(pb, b)
        assert probe.pixel_mae(pa, pa) == 0.0, "a file must match itself exactly"
        assert abs(probe.pixel_mae(pa, pb) - 1.0) < 1e-6, "black vs white should be 1.0"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
