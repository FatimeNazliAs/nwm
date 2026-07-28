"""Console formatting shared by every walkthrough probe.

Kept separate from the stages so that changing how a probe prints never touches
a stage that is already finished and published.
"""
import torch


def hr(title):
    """A section heading in the probe's console output."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(name, t):
    """One line describing a tensor: shape, dtype and range."""
    if torch.is_tensor(t):
        print(f"  {name:<14} shape={tuple(t.shape)!s:<22} dtype={str(t.dtype):<15} "
              f"min={t.min().item():+.3f} max={t.max().item():+.3f}")
    else:
        print(f"  {name:<14} {t}")
