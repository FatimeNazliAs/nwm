"""Saving a probe's pictures, and remembering which ones it saved.

A stage sends two things across the gap between its probe and its page: the
facts file, and thirty-odd PNGs. `debug/common/facts.py` made the first one a
validated interface (ADR-0004). The pictures travelled outside it -- a page
declared what it wanted, nothing declared what the probe wrote, and the two
could only disagree during a real GPU run.

This module closes that. A probe saves every picture through a Saver, and the
Saver accumulates the names as it goes. The manifest is therefore not a list
anybody maintains: it is built BY the act of writing, so it cannot fall out of
step with what is on disk. It is then stamped into the facts file, and the page
checks its needs against it.

Deliberately narrow: the Saver owns the directory, the `.png` suffix and the
manifest, and nothing else. Every matplotlib option stays the caller's, so
routing an existing probe through it does not change a single pixel of an
already-published picture.

    save = images.Saver(out_dir)
    save.image("ctx_f0", ctx_v[0])
    save.image("noise_ch0", xn[0], cmap="magma")
    save.figure("filmstrip", fig, dpi=120, bbox_inches="tight")
    ...
    facts.write(out_dir, "loop_facts.json", data, stage="t5", images=save.names)
"""
import os

import matplotlib.pyplot as plt


class Saver:
    """Writes a probe's PNGs and records their names in the order written.

    Saving the same name twice (a probe that overwrites a panel) records it
    once: the manifest describes the files that exist, not the calls that were
    made.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._names = []

    def path(self, name):
        """Where `name` lands. The `.png` suffix is this module's business."""
        return os.path.join(self.out_dir, f"{name}.png")

    def image(self, name, arr, **kw):
        """One picture straight from an array. `kw` goes to plt.imsave untouched."""
        plt.imsave(self.path(name), arr, **kw)
        self._record(name)
        return name

    def figure(self, name, fig, close=True, **kw):
        """One matplotlib figure. `kw` goes to fig.savefig untouched.

        Closes the figure afterwards, because every call site in the walkthrough
        does exactly that and a leaked figure is a memory leak in a probe that
        makes forty of them. Pass close=False to keep it.
        """
        fig.savefig(self.path(name), **kw)
        if close:
            plt.close(fig)
        self._record(name)
        return name

    def _record(self, name):
        if name not in self._names:
            self._names.append(name)

    @property
    def names(self):
        """The manifest: every picture written through this Saver, in order."""
        return tuple(self._names)

    def __len__(self):
        return len(self._names)
