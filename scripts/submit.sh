#!/bin/bash
# Submit the AdaPBS runs of a list of glaciers to slurm: one array task per
# glacier (run_slurm.sh, which runs run_glacier.py in a container).
#
# Usage: scripts/submit.sh <glacier list> [<config>]
#   <glacier list>: a text file with one RGI id per line
#   <config>: the settings (default: scripts/config.sh, copy it from
#             config.sh.example). One config per experiment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
LIST=$1
CONFIG=${2:-$HERE/config.sh}
source "$CONFIG"
N=$(grep -c . "$LIST")
EXP_DIR="$OUTPUT_ROOT/$EXP_NAME"

# Freeze the code, the settings and the glacier list: the jobs run this copy,
# so that editing the repository while jobs wait in the queue changes nothing
# (and the copy says what was run).
SNAP="$EXP_DIR/_snapshot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SNAP/scripts" "$EXP_DIR/logs"
rsync -a --exclude __pycache__ "$REPO/oggm_adapbs" "$SNAP/"
cp "$HERE/run_glacier.py" "$HERE/run_slurm.sh" "$SNAP/scripts/"
cp "$CONFIG" "$SNAP/scripts/config.sh"
grep . "$LIST" > "$SNAP/glaciers.txt"
git -C "$REPO" rev-parse HEAD > "$SNAP/git_commit.txt" 2>/dev/null || true
git -C "$REPO" status --short > "$SNAP/git_status.txt" 2>/dev/null || true
chmod -R a-w "$SNAP"

SBATCH_ARGS=(--array="0-$((N - 1))"
             --job-name="adapbs_$EXP_NAME"
             --partition="$SLURM_PARTITION"
             --ntasks=1 --cpus-per-task="$SLURM_CPUS"
             --time="$SLURM_TIME" --mem-per-cpu="$SLURM_MEM_PER_CPU"
             --output="$EXP_DIR/logs/%A_%a.out")
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    SBATCH_ARGS+=(--account="$SLURM_ACCOUNT")
fi
if [ -n "${SLURM_MAIL:-}" ]; then
    SBATCH_ARGS+=(--mail-type=FAIL --mail-user="$SLURM_MAIL")
fi

echo "$N glaciers, code frozen in $SNAP"
sbatch "${SBATCH_ARGS[@]}" "$SNAP/scripts/run_slurm.sh" "$SNAP"
