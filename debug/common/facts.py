"""The facts file: the interface between a probe and everything downstream.

A probe measures; a page renders. They never share a process, so the only thing
that crosses between them is a JSON file (see docs/adr/0001-probe-page-split.md).
That makes the file a real interface, and this module is the only thing allowed
to read or write it.

What it buys over `json.load`:

  * a version stamp, so a page meeting a facts file written by an older probe
    says so instead of failing on a missing key 300 lines into rendering;
  * required-key validation at load time, with an error that names the key and
    tells the reader to re-run the probe;
  * read tracking, so `unused()` can report facts nobody consumes -- the probe
    measures far more than any one page shows, and without this the surplus is
    invisible.

    from debug.common import facts
    f = facts.load("debug/out/t4/<scene>/cdit_facts.json", required=("model",))
    ...
    print(facts.unused(f))
"""
import json
import os
from collections.abc import Mapping

SCHEMA_VERSION = 1

# Stamped into every file so a reader can tell probe output apart from anything
# else and can refuse a file it is too new or too old to understand.
_VERSION_KEY = "_schema_version"
_STAGE_KEY = "_stage"


class FactsError(RuntimeError):
    """Raised when a facts file is missing, stale, or missing a required key.

    Always phrased so the fix is obvious from the message alone: these errors
    are read by someone who has not looked at this code in months.
    """


class Facts(Mapping):
    """A read-only view over a facts file that remembers what was read.

    Behaves as a plain mapping -- `f["depth"]` -- so call sites do not change.
    The recording exists so `unused()` can answer "what did the probe measure
    that nothing displays?", which is otherwise unanswerable without grepping.
    """

    def __init__(self, data, path):
        self._data = data
        self._read = set()
        self.path = path

    def __getitem__(self, key):
        self._read.add(key)
        try:
            return self._data[key]
        except KeyError:
            raise FactsError(
                f"{self.path} has no fact {key!r}.\n"
                f"  The probe that wrote this file did not measure it. Re-run the "
                f"stage's probe.py, or stop asking the page for {key!r}."
            ) from None

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def read_keys(self):
        """The keys someone actually asked for, in this process."""
        return set(self._read)


def write(out_dir, filename, data, stage):
    """Write a facts file, stamping the schema version and the stage that made it."""
    stamped = {_VERSION_KEY: SCHEMA_VERSION, _STAGE_KEY: stage}
    stamped.update(data)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as fh:
        json.dump(stamped, fh, indent=2)
    return path


def load(path, required=()):
    """Read a facts file and check it is usable before anyone depends on it.

    `required` is the handful of keys the caller cannot proceed without. It is
    deliberately not the full key list -- pinning all 85 would make every probe
    change a page change.
    """
    if not os.path.exists(path):
        raise FactsError(
            f"no facts file at {path}.\n"
            f"  Run the stage's probe.py first -- the page is built from what the "
            f"probe measured, and cannot be built without it."
        )
    with open(path) as fh:
        data = json.load(fh)

    found = data.get(_VERSION_KEY)
    if found is None:
        # Written before versioning existed. Readable, but say so once.
        print(f"  note: {path} predates facts versioning; assuming v{SCHEMA_VERSION}")
    elif found > SCHEMA_VERSION:
        raise FactsError(
            f"{path} was written by a newer probe (schema v{found}, this reader "
            f"understands v{SCHEMA_VERSION}).\n  Update debug/common/facts.py."
        )

    missing = [k for k in required if k not in data]
    if missing:
        raise FactsError(
            f"{path} is missing required fact(s): {', '.join(sorted(missing))}.\n"
            f"  It was probably written by an older probe. Re-run the stage's "
            f"probe.py to regenerate it."
        )
    return Facts(data, path)


def unused(f):
    """Facts present in the file that nothing read. Sorted, internals excluded.

    A non-empty result is not automatically a bug: some facts are consumed by
    hand when writing the Notion pages rather than by the page builder. It is a
    prompt to decide which, not a failure.
    """
    if not isinstance(f, Facts):
        raise TypeError("unused() needs a Facts from load()")
    return sorted(set(f) - f.read_keys() - {_VERSION_KEY, _STAGE_KEY})
