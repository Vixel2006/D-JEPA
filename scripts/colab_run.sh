#!/usr/bin/env bash
# Run the current working tree on a fresh Colab VM and pull the outputs back.
#
# Usage:
#   scripts/colab_run.sh --mode synthetic --quick --device cpu
#   scripts/colab_run.sh --mode clevr --seeds 3 --pretrain-steps 3000
#
# Environment overrides:
#   SESSION=name       Colab session name (default: pidjepa-<timestamp>)
#   GPU=T4|L4|A100      Accelerator (default: T4; set GPU="" for CPU)
#   TIMEOUT=seconds    Remote execution timeout (default: 7200)
#   KEEP=1             Leave the VM running after the run (debugging)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-pidjepa-$(date +%s)}"
GPU="${GPU:-T4}"
TIMEOUT="${TIMEOUT:-7200}"
KEEP="${KEEP:-0}"
PIDJEPA_ARGS="$*"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$(mktemp -t pidjepa_src.XXXXXX.tar.gz)"
LOCAL_OUTPUT="$ROOT/outputs/colab_${STAMP}.tar.gz"

cleanup() {
  rm -f "$ARCHIVE"
  if [[ "$KEEP" != "1" ]]; then
    colab stop -s "$SESSION" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[colab-run] packing working tree..."
# Only exclude caches and VCS metadata. Do NOT exclude bare names such as
# "data": tar matches those at every depth, which would silently drop
# src/pidjepa/data.py.
tar -czf "$ARCHIVE" \
  -C "$ROOT" \
  --exclude='*/__pycache__' \
  --exclude='*/__pycache__/*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  --exclude='.venv' \
  --exclude='.git' \
  src pyproject.toml README.md requirements.txt scripts

echo "[colab-run] creating session '$SESSION'..."
if [[ -n "$GPU" ]]; then
  colab new -s "$SESSION" --gpu "$GPU"
else
  colab new -s "$SESSION"
fi

echo "[colab-run] uploading source archive..."
colab upload -s "$SESSION" "$ARCHIVE" /content/pidjepa_src.tar.gz

echo "[colab-run] installing dependencies..."
colab exec -s "$SESSION" --timeout 1200 <<'PY'
import os
import subprocess
import sys
import tarfile

repo = "/content/pidjepa"
os.makedirs(repo, exist_ok=True)
with tarfile.open("/content/pidjepa_src.tar.gz") as archive:
    try:
        archive.extractall(repo, filter="data")
    except TypeError:  # Python < 3.12
        archive.extractall(repo)

required = ["src/pidjepa/cli.py", "src/pidjepa/models.py", "scripts/colab_remote.py"]
missing = [path for path in required if not os.path.exists(os.path.join(repo, path))]
if missing:
    raise SystemExit(f"source archive is incomplete, missing: {missing}")

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q",
     "seaborn", "scikit-learn", "transformers", "datasets"]
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", repo]
)
print("[colab] install complete")
PY

echo "[colab-run] running benchmark: $PIDJEPA_ARGS"
set +e
colab exec -s "$SESSION" --timeout "$TIMEOUT" \
  --env "PIDJEPA_ARGS=$PIDJEPA_ARGS" \
  -f "$ROOT/scripts/colab_remote.py"
RUN_STATUS=$?
set -e

echo "[colab-run] downloading outputs..."
mkdir -p "$(dirname "$LOCAL_OUTPUT")"
DOWNLOAD_STATUS=0
colab download -s "$SESSION" /content/pidjepa_outputs.tar.gz "$LOCAL_OUTPUT" || {
  DOWNLOAD_STATUS=$?
  echo "[colab-run] no output archive was produced"
}

if [[ "$RUN_STATUS" -ne 0 ]]; then
  echo "[colab-run] benchmark failed with exit code $RUN_STATUS" >&2
  exit "$RUN_STATUS"
fi
if [[ "$DOWNLOAD_STATUS" -ne 0 ]]; then
  exit "$DOWNLOAD_STATUS"
fi
echo "[colab-run] outputs: $LOCAL_OUTPUT"
