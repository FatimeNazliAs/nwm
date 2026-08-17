#!/usr/bin/env bash
#
#  Rebuild the T2 page after you have edited config.yaml.
#
#      ./debug/t2/update.sh
#
#  That is the whole thing. Run it on the server, from anywhere (you do NOT need
#  to be inside the container first -- this script does the docker/conda dance
#  for you). It rebuilds the page and makes sure it is being served, so from the
#  second run onwards all you do is refresh the browser tab.
#
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Serve the PARENT debug/out so every stage coexists on one port -- otherwise a
# server already up for another stage (e.g. T4) would shadow this one. View each
# stage at /tN/latest.html.
SERVE_DIR="$REPO/debug/out"
VIEW="t2/latest.html"
PORT=8000
CONTAINER=nwm_debug

# ---------------------------------------------------------------- the scene
# Echoed back so you can see your edit landed.
echo
echo "  scene from debug/t2/config.yaml"
grep -E '^(sample|trajectory|time):' "$REPO/debug/t2/config.yaml" | sed 's/^/      /'
echo

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "  the '$CONTAINER' container is not running. Start it, then run this again." >&2
    exit 1
fi

# ---------------------------------------------------------------- rebuild
# probe.py loads one real sample and writes scene_facts.json + the PNGs.
# build_page.py reads only those and writes the HTML. No model -- a few seconds.
echo "  loading one sample and rebuilding ..."
docker exec "$CONTAINER" bash -c '
    set -e
    export HOME=/tmp USER=nazli HF_HOME=/data/.cache/huggingface
    source /opt/conda/etc/profile.d/conda.sh
    conda activate nwm
    cd /app
    python debug/t2/probe.py
    python debug/t2/build_page.py
'

# ---------------------------------------------------------------- serve
# Only started if nothing is already listening, so running this repeatedly does
# not leave a pile of servers behind.
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    exec 3<&- 2>/dev/null || true
    SERVER_WAS_UP=yes
else
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
    echo "    Done.  Open this on your laptop:"
    echo
    echo "        http://localhost:$PORT/$VIEW"
    echo
    echo "    VS Code forwards the port for you -- see the PORTS panel."
    echo "    Leave the tab open; next time you only need to refresh."
fi
echo "  ────────────────────────────────────────────────────────────"
echo
