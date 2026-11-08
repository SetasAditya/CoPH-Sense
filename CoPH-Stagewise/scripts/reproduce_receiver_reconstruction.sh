#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
archive_root=$(realpath "${1:-results/receiver_reconstruction_v1}")
output_root=$(realpath -m "${2:-results/receiver_reconstruction_reproduction}")
receiver_checkout=$(mktemp -d /tmp/coph-receiver-XXXXXX)
python - "$archive_root" "$receiver_checkout" <<'PY'
import sys,zipfile
from pathlib import Path
root,dest=map(Path,sys.argv[1:])
with zipfile.ZipFile(root/'sources.zip') as z:
 for name in z.namelist():
  path=dest/name
  if not path.resolve().is_relative_to(dest.resolve()):raise ValueError('unsafe archive path')
 z.extractall(dest)
import json
revision=root/"reporting_revision.json"
if revision.exists():
 r=json.loads(revision.read_text());(dest/r["path"]).write_bytes((root/r["snapshot"]).read_bytes())
PY
reference_root=$(python - "$archive_root" <<'PY'
import json,sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'manifest.json').read_text())['reference'])
PY
)
cd "$receiver_checkout"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg
python -m coph_stagewise.cli receiver-reconstruction suite --reference "$reference_root" --out "$output_root" --workers 4
python -m coph_stagewise.cli receiver-reconstruction render --out "$output_root"
printf 'Frozen checkout retained at %s\n' "$receiver_checkout"
