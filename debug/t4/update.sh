#!/usr/bin/env bash
#  Rebuild the T4 page after editing debug/t4/config.yaml:
#      ./debug/t4/update.sh
#  The logic lives once in debug/common/update.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" t4
