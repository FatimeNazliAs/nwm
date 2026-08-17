#!/usr/bin/env bash
#
#  Rebuild one walkthrough stage's page after you have edited its config.yaml.
#
#      debug/common/update.sh t3
#
#  You normally do not call this directly -- each stage has a one-line wrapper,
#  so `./debug/t3/update.sh` is the documented command. This file is the single
#  copy of the logic: the docker/conda dance, the two scripts, and the server.
#
#  Every stage serves the SHARED debug/out directory on one port, so stages
#  never shadow each other. Each is viewed at /tN/latest.html.
#
set -euo pipefail

STAGE="${1:?usage: update.sh <stage>   e.g. update.sh t3}"

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAGE_DIR="$REPO/debug/$STAGE"
SERVE_DIR="$REPO/debug/out"
PORT=8000
CONTAINER=nwm_debug

if [ ! -d "$STAGE_DIR" ]; then
    echo "  no such stage: debug/$STAGE" >&2
    exit 1
fi

# ---------------------------------------------------------------- the scene
# Echoed back so you can see your edit landed before anything slow happens.
# One pattern covers every stage's knobs; missing ones simply do not match.
echo
echo "  scene from debug/$STAGE/config.yaml"
grep -E '^(sample|trajectory|time|frame|sec|diffusion_t):' "$STAGE_DIR/config.yaml" \
    | sed 's/^/      /' || true
echo

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "  the '$CONTAINER' container is not running. Start it, then run this again." >&2
    exit 1
fi

# ---------------------------------------------------------------- rebuild
# probe.py runs the real code and writes <stage>_facts.json + the PNGs.
# build_page.py reads only those and writes the HTML -- it never touches the
# model, so it is always fast. Stages that load a checkpoint (T4) are slow on
# the first run after a reboot and quick afterwards.
echo "  running debug/$STAGE/probe.py, then build_page.py ..."
docker exec "$CONTAINER" bash -c "
    set -e
    export HOME=/tmp USER=nazli HF_HOME=/data/.cache/huggingface
    source /opt/conda/etc/profile.d/conda.sh
    conda activate nwm
    cd /app
    python debug/$STAGE/probe.py
    python debug/$STAGE/build_page.py
"

# ---------------------------------------------------------------- serve
# Only started if nothing is already listening, so running this repeatedly does
# not leave a pile of servers behind.
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    exec 3<&- 2>/dev/null || true
    SERVER_WAS_UP=yes
else
    # setsid + all three streams detached, or the server inherits this script's
    # stdout and anything piping us (`| tee`, `| tail`) hangs waiting for EOF.
    ( cd "$SERVE_DIR" && setsid nohup python3 -m http.server "$PORT" \
        </dev/null >/dev/null 2>&1 & )
    sleep 1
    SERVER_WAS_UP=no
fi

echo
echo "  ────────────────────────────────────────────────────────────"
if [ "$SERVER_WAS_UP" = yes ]; then
    echo "    Done.  Refresh your browser tab."
else
    echo "    Done.  Serving on port $PORT; view this stage at /$STAGE/latest.html"
    echo "    Leave the tab open; next time you only need to refresh."
fi
echo "  ────────────────────────────────────────────────────────────"
echo
