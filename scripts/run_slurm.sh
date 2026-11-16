#!/bin/bash
#
# One slurm array task: the AdaPBS runs of one glacier (run_glacier.py).
# Submitted by submit.sh: do not run it directly.
#
#   $1: the frozen copy of the code made by submit.sh
#
# The work happens in an apptainer container which sees only the frozen code
# (read-only), the download cache (read-only, optional) and the job's own
# scratch. The outputs are copied to $OUTPUT_ROOT/$EXP_NAME at the end,
# outside the container.
set -euo pipefail

SNAP=$1
source "$SNAP/scripts/config.sh"
RGI_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SNAP/glaciers.txt")
JOB_SCRATCH="$SCRATCH_ROOT/$SLURM_JOB_ID"
OUT_DIR="$OUTPUT_ROOT/$EXP_NAME"
mkdir -p "$JOB_SCRATCH/out"
echo "$(date): task $SLURM_ARRAY_TASK_ID, $RGI_ID on $(hostname), $SLURM_CPUS_PER_TASK cpus"

CACHE_BIND=()
CACHE_ENV=""
if [ -n "${DOWNLOAD_CACHE:-}" ]; then
    CACHE_BIND=(--bind "$DOWNLOAD_CACHE:$DOWNLOAD_CACHE:ro")
    CACHE_ENV="export OGGM_DOWNLOAD_CACHE=$DOWNLOAD_CACHE OGGM_DOWNLOAD_CACHE_RO=1"
fi

# --containall: empty home, /tmp and environment (nothing from the host)
# --no-mount bind-paths: ignore the binds of the cluster's apptainer config
# (some clusters bind all of /home read-write by default). This also drops
# the DNS configuration of the node: /etc/resolv.conf is bound back
# (read-only), otherwise no host name resolves (no download, no pip).
STATUS=0
apptainer exec --containall --no-mount bind-paths \
    --bind /etc/resolv.conf:/etc/resolv.conf:ro \
    --bind "$SNAP:$SNAP:ro" ${CACHE_BIND[@]+"${CACHE_BIND[@]}"} \
    --bind "$JOB_SCRATCH" \
    "$CONTAINER" bash -s <<EOF || STATUS=$?
set -e
# Stop if the container can write to the frozen code: not isolated
if [ -w "$SNAP" ]; then echo "NOT ISOLATED: $SNAP is writable"; exit 1; fi

export HOME="$JOB_SCRATCH/home"
# The container's /tmp is small (in memory): pip and OGGM use the scratch
export TMPDIR="$JOB_SCRATCH/tmp"
export OGGM_WORKDIR="$JOB_SCRATCH/wd"
export OGGM_EXTRACT_DIR="$JOB_SCRATCH/extract"
mkdir -p "\$HOME" "\$TMPDIR" "\$OGGM_WORKDIR" "\$OGGM_EXTRACT_DIR"
$CACHE_ENV
# The host environment is gone: OGGM needs this one to count the cpus
export SLURM_JOB_CPUS_PER_NODE="$SLURM_JOB_CPUS_PER_NODE"
export MPLBACKEND=agg

# OGGM at a fixed commit, in a venv on top of the container's packages
python3 -m venv --system-site-packages "$JOB_SCRATCH/venv"
source "$JOB_SCRATCH/venv/bin/activate"
pip install --quiet --no-deps "git+https://github.com/OGGM/oggm.git@$OGGM_COMMIT"
# oggm_adapbs from the frozen copy (pure python: no install needed)
export PYTHONPATH="$SNAP"

python -u "$SNAP/scripts/run_glacier.py" "$RGI_ID" \
    --working-dir "$JOB_SCRATCH/wd" --output-dir "$JOB_SCRATCH/out" \
    --n-workers "$SLURM_CPUS_PER_TASK" $RUN_ARGS
EOF

echo "$(date): container exit status $STATUS, copying the outputs to $OUT_DIR"
mkdir -p "$OUT_DIR"
rsync -a "$JOB_SCRATCH/out/" "$OUT_DIR/"
rm -rf "${JOB_SCRATCH:?}"
exit $STATUS
