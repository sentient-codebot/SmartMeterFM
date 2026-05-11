#!/usr/bin/env bash
# Download / verify the Low Carbon London halfhourly dataset + household
# metadata into data/lcl_electricity/raw/.
#
# Required end-state in data/lcl_electricity/raw/:
#   - halfhourly_dataset/block_*.csv.gz  (>= 100 files)
#   - informations_households.csv
#
# Resolution preference order:
#   1. Already on disk → just verify.
#   2. Kaggle CLI present + credentials configured → automatic download from
#      jeanmidev/smart-meters-in-london.
#   3. Manual instructions (London Datastore / Kaggle web UI).
#
# This script is idempotent: re-running with files already in place is a
# fast no-op.

set -euo pipefail

# Resolve repo root via this script's location: scripts/data/download_lcl.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RAW="${REPO_ROOT}/data/lcl_electricity/raw"

mkdir -p "${RAW}"

verify() {
    local hh_dir="${RAW}/halfhourly_dataset"
    local info="${RAW}/informations_households.csv"
    if [[ ! -f "${info}" ]]; then
        return 1
    fi
    if [[ ! -d "${hh_dir}" ]]; then
        return 1
    fi
    # Accept both .csv (Kaggle mirror) and .csv.gz (London Datastore release).
    local n
    n="$(find "${hh_dir}" -maxdepth 1 \
            \( -name 'block_*.csv' -o -name 'block_*.csv.gz' \) | wc -l)"
    if (( n < 100 )); then
        return 1
    fi
    echo "OK: ${n} block files; informations_households.csv present"
    return 0
}

canonicalise_layout() {
    # Some LCL ZIP releases unpack to a double-nested
    # ``halfhourly_dataset/halfhourly_dataset/`` layout.  Flatten to one level.
    local outer="${RAW}/halfhourly_dataset"
    local inner="${outer}/halfhourly_dataset"
    if [[ -d "${inner}" ]]; then
        echo "Flattening double-nested halfhourly_dataset/ ..."
        mv "${inner}"/* "${outer}/"
        rmdir "${inner}"
    fi
}

print_manual_instructions() {
    cat <<'MSG'

==============================================================================
Manual download instructions
==============================================================================

The Low Carbon London halfhourly + metadata files are not on disk and the
Kaggle CLI is not configured.  Please obtain them from one of:

A. London Datastore (~700 MB ZIP, registration may be required):
   https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/
   File:  Power-Networks-LCL-June2015withAcornGps.zip
   After download, unzip into  data/lcl_electricity/raw/  so that you have:
       data/lcl_electricity/raw/halfhourly_dataset/block_*.csv.gz
       data/lcl_electricity/raw/informations_households.csv

B. Kaggle (~600 MB, separate files):
   https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london
   Configure the Kaggle CLI:
       pip install --user kaggle
       mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
       # Place your kaggle.json (from kaggle.com → Account → Create API Token):
       chmod 600 ~/.kaggle/kaggle.json
   Then re-run this script — it will pick up the CLI automatically.

After files are in place, re-run:
   bash scripts/data/download_lcl.sh
==============================================================================
MSG
}

if verify; then
    canonicalise_layout
    verify
    exit 0
fi

# Try Kaggle CLI.
if command -v kaggle >/dev/null 2>&1 \
   && [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
    echo "Kaggle CLI detected — downloading jeanmidev/smart-meters-in-london ..."
    kaggle datasets download \
        -d jeanmidev/smart-meters-in-london \
        -p "${RAW}" \
        --unzip
    canonicalise_layout
    if verify; then
        exit 0
    fi
    echo "Kaggle download did not produce the expected files; falling back to manual."
fi

print_manual_instructions
exit 1
