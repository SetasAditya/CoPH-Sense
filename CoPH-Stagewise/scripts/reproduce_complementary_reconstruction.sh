#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference="$(realpath "${1:-results/complementary_reconstruction_v1}")"
output="$(realpath -m "${2:-results/complementary_reconstruction_reproduction}")"
archive_dir="$(mktemp -d /tmp/coph-complementary-repro.XXXXXX)"
python - "$reference/sources.zip" "$archive_dir" <<'PY'
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:z.extractall(sys.argv[2])
PY
if [[ -f "$reference/reporting_revision.json" ]]; then
  cp "$reference/verification/complementary_reporting.py.txt" "$archive_dir/coph_stagewise/complementary_reporting.py"
fi
cd "$archive_dir"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/coph-mpl \
python -m coph_stagewise.receiver_reconstruction suite --profile complementary --out "$output"
