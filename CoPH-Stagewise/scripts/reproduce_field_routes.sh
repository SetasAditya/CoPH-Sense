#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
archive_root=$(realpath "${1:-results/field_route_v1}")
output_root=$(realpath -m "${2:-results/field_route_reproduction}")
field_checkout=$(mktemp -d /tmp/coph-field-route-XXXXXX)
python - "$archive_root" "$field_checkout" <<'PY'
import sys,zipfile
from pathlib import Path
root,dest=map(Path,sys.argv[1:])
with zipfile.ZipFile(root/'sources.zip') as z:
 for name in z.namelist():
  path=dest/name
  if not path.resolve().is_relative_to(dest.resolve()):raise ValueError('unsafe archive path')
 z.extractall(dest)
PY
reference_root=$(python - "$archive_root" <<'PY'
import json,sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'manifest.json').read_text())['reference'])
PY
)
cd "$field_checkout"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg
python -m coph_stagewise.cli field-route-suite suite --reference "$reference_root" --out "$output_root" --workers 6
python -m coph_stagewise.cli field-route-suite render --out "$output_root"
python -m coph_stagewise.cli field-route-suite verify --out "$output_root"
printf 'Frozen checkout retained at %s\n' "$field_checkout"
