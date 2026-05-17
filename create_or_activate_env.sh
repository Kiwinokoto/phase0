#!/usr/bin/env bash
# Create or activate the local Conda environment for this project.
#
# IMPORTANT: source this file, do not execute it, otherwise Conda activation
# will disappear when the script exits.
#
# Usage:
#   source ./create_or_activate_env.sh
#
# Optional:
#   ALIFE_CONDA_ENV=alife-dev source ./create_or_activate_env.sh
#   ALIFE_PYTHON_VERSION=3.11 source ./create_or_activate_env.sh
#   source ./create_or_activate_env.sh --no-install
#   source ./create_or_activate_env.sh --force-install

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced so it can activate Conda in your current shell."
  echo "Usage: source ./create_or_activate_env.sh"
  exit 1
fi

ENV_NAME="${ALIFE_CONDA_ENV:-alife_phase2}"
PYTHON_VERSION="${ALIFE_PYTHON_VERSION:-3.11}"
INSTALL_MODE="auto" # auto | never | always

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install)
      INSTALL_MODE="never"
      shift
      ;;
    --force-install)
      INSTALL_MODE="always"
      shift
      ;;
    --name|-n)
      ENV_NAME="${2:?Missing environment name after $1}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:?Missing Python version after --python}"
      shift 2
      ;;
    --help|-h)
      cat <<USAGE
Usage:
  source ./create_or_activate_env.sh [options]

Options:
  -n, --name NAME     Conda environment name. Default: alife_phase2
  --python VERSION    Python version when creating the env. Default: 3.11
  --no-install        Activate only; do not install dependencies
  --force-install     Force pip install -r requirements.txt even if unchanged

Env vars:
  ALIFE_CONDA_ENV
  ALIFE_PYTHON_VERSION
USAGE
      return 0
      ;;
    *)
      echo "Unknown option: $1"
      return 1
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$PROJECT_DIR/requirements.txt"
CACHE_DIR="$PROJECT_DIR/.alife_env"
SAFE_ENV_NAME="${ENV_NAME//[^A-Za-z0-9_.-]/_}"
REQ_HASH_FILE="$CACHE_DIR/${SAFE_ENV_NAME}.requirements.sha256"

hash_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    python - "$file" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
  fi
}

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found in PATH. Open an Anaconda/Miniconda shell or run conda init first."
  return 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating Conda env '$ENV_NAME' with Python $PYTHON_VERSION..."
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "Conda env '$ENV_NAME' already exists."
fi

conda activate "$ENV_NAME"

if [[ "$INSTALL_MODE" != "never" ]]; then
  if [[ ! -f "$REQ_FILE" ]]; then
    echo "Missing requirements.txt at: $REQ_FILE"
    return 1
  fi

  mkdir -p "$CACHE_DIR"
  CURRENT_HASH="$(hash_file "$REQ_FILE")"
  PREVIOUS_HASH=""
  if [[ -f "$REQ_HASH_FILE" ]]; then
    PREVIOUS_HASH="$(cat "$REQ_HASH_FILE")"
  fi

  if [[ "$INSTALL_MODE" == "always" || "$CURRENT_HASH" != "$PREVIOUS_HASH" ]]; then
    if [[ "$INSTALL_MODE" == "always" ]]; then
      echo "Force installing Python dependencies from requirements.txt..."
    else
      echo "requirements.txt changed or not installed yet; installing dependencies..."
    fi
    python -m pip install --upgrade pip
    python -m pip install -r "$REQ_FILE"
    printf '%s\n' "$CURRENT_HASH" > "$REQ_HASH_FILE"
  else
    echo "requirements.txt unchanged; skipping dependency install."
  fi
else
  echo "Skipping dependency install."
fi

echo ""
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Project directory: $PROJECT_DIR"
echo "Run: python run.py"
