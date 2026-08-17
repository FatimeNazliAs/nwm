# Implementation — how to run T3

Source of truth for three Notion pages. Edit here, paste there.

---

## PAGE: Implementation — how to run T3  (parent)

Two ways to run T3. Both do the same thing — pick one.

### What you are running

1. **`probe.py`** — runs the real VAE on the scene in `config.yaml`, writes
   `vae_facts.json` + PNGs.
2. **`build_page.py`** — reads *only* those files and writes the HTML. Never
   touches the model.

Outputs land in `debug/out/t3/<scene>/` (kept per scene) and
`debug/out/t3/latest.html` (overwritten every run — this is the one that gets
published).

### Republish the shared page

Running the scripts updates the file on disk. It does **not** update the link.
Only an assistant can publish, so in any chat paste both lines:

> Republish `/home/nazli/projects/nwm/debug/out/t3/latest.html`
> to `https://claude.ai/code/artifact/67ff8417-b8ab-47b7-9a95-b27f9c8af427`

Both halves are required. The **path** says what to publish; the **URL** keeps
the link stable. Leave the URL out and a brand-new artifact is minted, and the
link you already shared keeps showing the old run.

Children: *Run it with one command (T3)*, *Run it by hand (T3)*.

---

## PAGE: Run it with one command (T3)

### Step 1 — edit the scene *(optional)*

`debug/t3/config.yaml`. Skip to re-run the current scene. See *Pick a different
data example (T3)*.

### Step 2 — run it

From anywhere on the server. You do **not** need to be inside the container first.

```bash
./debug/t3/update.sh
```

It prints the scene it read, so you can check your edit landed, then runs the
probe and rebuilds the page. A few seconds — T3 has no checkpoint to load.

### Step 3 — republish

The page on disk is now current, but the shared link is not. See *Republish the
shared page* on the parent page.

---

## PAGE: Run it by hand (T3)

Only if you want to run the two halves separately, or watch the probe's output
live. Otherwise use *Run it with one command (T3)*.

### Step 1 — attach to the container

```bash
docker exec -it nwm_debug bash
```

### Step 2 — set up the shell

Needed in every fresh shell, or the HF cache and `$HOME` land in the wrong place.

```bash
export HOME=/tmp USER=nazli HF_HOME=/data/.cache/huggingface
source /opt/conda/etc/profile.d/conda.sh && conda activate nwm
cd /app
```

### Step 3 — run the probe

```bash
python debug/t3/probe.py
```

Prints seven sections in dataflow order: scene → the VAE → pixels in → inside
the encoder → encode → the compression budget → decode + error.

### Step 4 — build the page

```bash
python debug/t3/build_page.py
```

To rebuild an *older* scene without re-running the model, pass its folder name:

```bash
python debug/t3/build_page.py jackal_2019-08-02-16-28-30_5_r01__t6
```

### Step 5 — republish

See *Republish the shared page* on the parent page.

---

## Not on the Notion pages, on purpose

`update.sh` also starts a local preview server on port 8000. The user cannot
reach `localhost:` links, so the pages do not mention it — republishing to the
fixed artifact URL is the documented way to see an updated page. Do not add
localhost instructions back.
