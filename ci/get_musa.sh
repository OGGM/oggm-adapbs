#!/bin/bash
# Download MuSA's modules/filters.py (for the AMIS comparison test) into <dir>.
# Usage: ci/get_musa.sh <dir> [<ref>]
# The default ref is the MuSA commit AdaPBS was ported from.
set -e
MUSA_REF=${2:-eb56ca77044a1d022f4f21659a39ec2c70f63ee8}
mkdir -p "$1/modules"
URL="https://raw.githubusercontent.com/ealonsogzl/MuSA/$MUSA_REF/modules/filters.py"
# Python rather than curl: always available in the test images
python3 -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
    "$URL" "$1/modules/filters.py"
