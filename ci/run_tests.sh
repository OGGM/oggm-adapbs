#!/bin/bash
# Same as OGGM's ci/run_tests.sh, for one test environment
set -e
set -x

chown -R "$(id -u):$(id -g)" "$HOME"

export MPLBACKEND=agg
PIP=${PIP:-pip3}

$PIP install --upgrade pytest coverage git+https://github.com/fmaussion/salem.git
# The OGGM development version, which brings the test fixtures as well
$PIP install --upgrade git+https://github.com/OGGM/oggm.git
$PIP install -e .

# MuSA's AMIS at the pinned commit, for the comparison test
export MUSA_DIR="$HOME/MuSA"
./ci/get_musa.sh "$MUSA_DIR"

coverage erase
coverage run --source=./oggm_adapbs --module \
    pytest --verbose --durations=0 --run-slow oggm_adapbs
coverage report --skip-covered
