#!/usr/bin/env bash
#  Rebuild the T6 page after editing debug/t6/config.yaml:
#      ./debug/t6/update.sh
#  The logic lives once in debug/common/update.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" t6
