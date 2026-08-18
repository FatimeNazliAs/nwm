#!/usr/bin/env bash
#  Rebuild the T5 page after editing debug/t5/config.yaml:
#      ./debug/t5/update.sh
#  The logic lives once in debug/common/update.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/../common/update.sh" t5
