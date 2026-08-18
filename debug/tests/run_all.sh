#!/usr/bin/env bash
#
#  Every walkthrough test, in one command:
#
#      ./debug/tests/run_all.sh
#
#  All of them build pages and check contracts from committed fixtures, so none
#  needs a GPU, a checkpoint or the dataset. The whole suite is a few seconds.
#  Run it inside the nwm_debug container, or anywhere the nwm env is active.
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

failed=0
for t in debug/tests/test_*.py; do
    echo "── $t"
    if python "$t"; then
        :
    else
        echo "   FAILED: $t"
        failed=$((failed + 1))
    fi
    echo
done

if [ "$failed" -ne 0 ]; then
    echo "$failed test file(s) failed."
    exit 1
fi
echo "All test files passed."
