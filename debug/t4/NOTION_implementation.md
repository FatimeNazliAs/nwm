# Implementation — the T4 probe

Source of truth for the Notion Implementation pages. Edit here, paste there.

In Notion this content is split three ways, per the walkthrough convention:

| Notion page | Takes which sections below |
| --- | --- |
| **Implementation — the T4 probe** (parent) | "What you are running", "Where the output goes", "Republishing", "The files" |
| **Run it with one command (T4)** | "How to run it" + "How to see the page" |
| **Run it by hand (T4)** | "The long way, by hand" |
| **Pick a different data example (T4)** | "How to change what the page shows" |

---

## What you are running

Two scripts, in order. Nothing else.

1. **`debug/t4/probe.py`** — builds the real CDiT-XL/2, loads the checkpoint,
   runs real forward passes on the scene named in `config.yaml`.
   Writes `cdit_facts.json` plus a set of PNGs.
2. **`debug/t4/build_page.py`** — reads *only* those files and writes the HTML.
   It never touches the model, so it is instant.

The split matters: if the page looks wrong but the numbers are right, only
step 2 needs rerunning.

---

## How to run it

On the server:

```bash
./debug/t4/update.sh
```

That is the whole procedure. The script:

1. prints the scene it read from `config.yaml`, so you can see your edit landed;
2. enters the `nwm_debug` container, activates the `nwm` conda env;
3. runs `probe.py`, then `build_page.py`;
4. starts a web server on port 8000 if one is not already up.

Roughly ten minutes the first time after a reboot, almost all of it in
`probe.py` loading the checkpoint. Under a minute on repeat runs, once that
checkpoint is sitting in the page cache.

### The long way, by hand

Only if you want to run the halves separately.

```bash
docker exec -it nwm_debug bash
export HOME=/tmp USER=nazli HF_HOME=/data/.cache/huggingface
source /opt/conda/etc/profile.d/conda.sh && conda activate nwm
cd /app && python debug/t4/probe.py && python debug/t4/build_page.py
```

---

## How to see the page

You work in VS Code over Remote-SSH, so the file lives on the server and your
browser lives on your laptop. `file://` cannot cross that gap.

1. Run `./debug/t4/update.sh` once. It starts the server for you.
2. VS Code forwards port 8000 automatically — check the **PORTS** panel.
3. On your laptop, open **`http://localhost:8000/latest.html`**
4. **Leave the tab open.**

From then on, every rebuild is: edit config → `./debug/t4/update.sh` → **refresh
the tab**. No republishing, no waiting on anyone.

---

## How to change what the page shows

Open `debug/t4/config.yaml`. Section 1 is three lines:

```yaml
trajectory: jackal_2019-08-02-16-28-30_5_r01
time: 6
sec: 10
```

- **`trajectory`** — which drive. Names come from
  `data_splits/recon/test/traj_names.txt`.
- **`time`** — the frame you start from.
- **`sec`** — how far ahead to predict, in seconds. Up to 16.

Section 2 of the file holds six ready-made scenes with measured distance and
turn angle. Copy any block over those three lines.

### Two rules for a scene of your own

Both are checked, and you get a message rather than a crash if you break one.

- `time >= 3` — four context frames are needed, ending at `time`.
- `time + 64 <= last frame` — the dataset always builds the full 16 s.

The second does not depend on `sec`, so a scene that works at `sec: 1` also
works at `sec: 16`.

To search for more scenes: `python debug/t4/find_scenes.py`.

### One setting worth knowing about

`diffusion_t` must be one of the 250 steps the real loop actually visits
(0, 4, 8, … 995, 999). The stride is 999/249 = 4.012, not 4, so some gaps are 5
and **996 is never used**. The probe reads `diffusion.timestep_map` and rejects
anything that is not on the list.

---

## Where the output goes

Two copies, on purpose.

- **`debug/out/t4/<trajectory>__t<time>/`** — a folder per scene. This is what
  accumulates; nothing overwrites your earlier runs.
- **`debug/out/t4/latest.html`** — a fixed path, overwritten every run. This is
  what gets served and what gets published, so the live URL stays the same
  however many times you change the scene.

---

## Republishing the shareable page

The web page for advisors lives at a fixed artifact URL:

```
https://claude.ai/code/artifact/b6cd5f71-733b-43a3-bc73-f7fd52739035
```

Only an assistant can publish. In any chat, paste:

> Republish `/home/nazli/projects/nwm/debug/out/t4/latest.html`
> to `https://claude.ai/code/artifact/b6cd5f71-733b-43a3-bc73-f7fd52739035`

Both halves are required. The **path** says what to publish; the **URL** keeps
the link stable. Leave the URL out and a brand-new artifact is minted, and the
link you gave your advisors keeps showing the old run.

---

## The files

| File | What it is |
| --- | --- |
| `debug/t4/config.yaml` | the scene, and every knob |
| `debug/t4/update.sh` | one command: run the model, rebuild, serve |
| `debug/t4/probe.py` | runs the real model, writes facts + PNGs |
| `debug/t4/build_page.py` | facts + PNGs → HTML. Never runs the model |
| `debug/t4/find_scenes.py` | surveys all 500 eval scenes to find good ones |
| `debug/common/page.py` | the shared page shell used by T1–T6 |
