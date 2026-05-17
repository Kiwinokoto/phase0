#!/usr/bin/env bash
# Convenience launcher: create/activate the Conda env, install deps only when
# requirements.txt changed, then run the sandbox.
#
# Usage:
#   ./run.sh
#   ./run.sh --seed 123 --width 320 --height 160
#
# Env vars:
#   ALIFE_CONDA_ENV=alife-dev ./run.sh
#   ALIFE_PYTHON_VERSION=3.11 ./run.sh
#   ALIFE_FORCE_SOFTWARE_RENDER=0 ./run.sh   # opt back into hardware rendering

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# shellcheck source=/dev/null
source "$PROJECT_DIR/create_or_activate_env.sh"

export ALIFE_FORCE_SOFTWARE_RENDER="${ALIFE_FORCE_SOFTWARE_RENDER:-1}"

exec python "$PROJECT_DIR/run.py" "$@"
