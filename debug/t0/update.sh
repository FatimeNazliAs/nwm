#!/usr/bin/env bash
#
#  Serve the T0 overview page:
#      ./debug/t0/update.sh
#
#  T0 is the odd stage: the page is hand-written and static. There is no
#  config.yaml, no probe.py and no build_page.py, so there is nothing for
#  debug/common/update.sh to run -- this script only does its second half,
#  copying the page into the shared serve directory and starting the server.
#
#  Republish after editing debug/t0/latest.html: ask Claude to publish that
#  file to the FIXED artifact URL below with url=, or advisors keep seeing
#  the old page.
#      https://claude.ai/code/artifact/ea04dacd-05a3-44d4-9e91-13e80950f2a0
#
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PAGE="$REPO/debug/t0/latest.html"
SERVE_DIR="$REPO/debug/out"
PORT=8000

# ---------------------------------------------------------------- publish
# Every stage is viewed under the one shared debug/out directory at
# /tN/latest.html. T0's page is written by hand and lives in the stage dir
# (debug/out is not tracked), so it is copied across rather than generated.
mkdir -p "$SERVE_DIR/t0"
cp "$PAGE" "$SERVE_DIR/t0/latest.html"
echo
echo "  copied debug/t0/latest.html  ($(du -k "$PAGE" | cut -f1) KB)"

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
    echo "    Done.  Serving on port $PORT; view this stage at /t0/latest.html"
    echo "    Leave the tab open; next time you only need to refresh."
fi
echo "  ────────────────────────────────────────────────────────────"
echo
