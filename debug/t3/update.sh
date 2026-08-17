#!/usr/bin/env bash
#  Rebuild the T3 page after editing debug/t3/config.yaml:
#      ./debug/t3/update.sh
#  The logic lives once in debug/common/update.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" t3
