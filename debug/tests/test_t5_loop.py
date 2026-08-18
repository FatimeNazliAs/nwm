"""The Run/Pass contract, driven by a fake diffusion. No model, no GPU.

sample_loop's honest raw result is 251 latents and 250 guesses -- two sequences
of different lengths whose alignment used to live only in a docstring, and which
eight call sites re-derived by hand. Run exists to make that off-by-one
impossible, so this is the test that matters most:

    every pass's x_out IS the next pass's x_in, and the run's final IS the last
    x_out.

Get that wrong and every number on the T5 page shifts by one pass, silently.

    python debug/tests/test_t5_loop.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch

from debug.t5 import probe


class FakeDiffusion:
    """The smallest object sample_loop can drive.

    Yields the same two keys p_sample does, with values chosen so that a
    mis-alignment shows up as an obviously wrong number rather than as noise:
    x counts 1, 2, 3... and each guess is ten times its own x.
    """

    def __init__(self, n=5):
        self.num_timesteps = n
        self.timestep_map = [i * 100 for i in range(n)]

    def p_sample_loop_progressive(self, model, shape, noise, **kw):
        x = noise
        for _ in range(self.num_timesteps):
            x = x + 1
            yield {"sample": x, "pred_xstart": x * 10}


class FakeModel:
    """sample_loop hands `model.forward` to the diffusion; the fake never calls it.

    Present so the attribute exists, and raising if it is ever reached, which
    would mean the test had quietly started depending on a real model.
    """

    def forward(self, *args, **kwargs):
        raise AssertionError("the fake diffusion should never call the model")


def _run(n=5):
    return probe.sample_loop(FakeDiffusion(n), model=FakeModel(),
                             z=torch.zeros(1, 1, 1, 1), model_kwargs={}, device=None)


def test_passes_are_numbered_in_loop_order():
    run = _run()
    assert len(run) == 5, "one Pass per timestep"
    assert run.steps == (4, 3, 2, 1, 0), f"the loop counts down, got {run.steps}"


def test_each_pass_output_is_the_next_pass_input():
    """The invariant Run exists to protect."""
    run = _run()
    for a, b in zip(run.passes, run.passes[1:]):
        assert torch.equal(a.x_out, b.x_in), \
            f"pass {a.step}'s output is not pass {b.step}'s input"


def test_first_input_is_the_starting_noise_and_final_is_the_last_output():
    run = _run()
    assert torch.equal(run.at(4).x_in, torch.zeros(1, 1, 1, 1)), \
        "the first pass must be handed the noise the caller passed in"
    assert torch.equal(run.final, run.at(0).x_out), \
        "final must be the last pass's output, not its input"


def test_guess_belongs_to_its_own_pass():
    """A shift of one would make guess = 10 x the WRONG pass's x."""
    run = _run()
    for p in run:
        assert torch.equal(p.guess, p.x_out * 10), f"pass {p.step} has another pass's guess"


def test_t_is_the_noise_level_not_the_step():
    """step and t are different numbers on different scales; Pass carries both."""
    run = _run()
    assert run.at(4).t == 400 and run.at(0).t == 0, "t must come from timestep_map"
    assert run.at(3).step == 3


def test_at_rejects_an_unknown_pass_with_the_valid_range():
    run = _run()
    try:
        run.at(99)
    except SystemExit as e:
        assert "4" in str(e) and "0" in str(e), "the error must name the range it has"
    else:
        raise AssertionError("an out-of-range pass number was accepted")


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
