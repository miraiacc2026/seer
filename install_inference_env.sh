#!/usr/bin/env bash
# Compatibility entry point; configuration and installation live in install_env.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/install_env.sh" "$@"
