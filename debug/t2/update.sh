#!/usr/bin/env bash
#  Rebuild the T2 page after editing debug/t2/config.yaml:
#      ./debug/t2/update.sh
#  The logic lives once in debug/common/update.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" t2
