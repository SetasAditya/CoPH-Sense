#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference=$(realpath "${1:-results/decision_bottleneck_v1}")
output=$(realpath -m "${2:-results/decision_bottleneck_reproduction}")
decision_checkout=$(mktemp -d /tmp/coph-decision-reproduction.XXXXXX)
input_reference=$(python - "$reference" "$decision_checkout" <<'PY'
import hashlib,json,sys,zipfile
from pathlib import Path
reference,checkout=map(Path,sys.argv[1:]);m=json.loads((reference/'manifest.json').read_text())
with zipfile.ZipFile(reference/'sources.zip') as z:
 for name,h in m['source_hashes'].items():
  rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
  data=z.read(name);assert hashlib.sha256(data).hexdigest()==h
  p=checkout/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print(m['reference'])
PY
)
cd "$decision_checkout"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg
python -m coph_stagewise.cli decision-diagnostics suite --reference "$input_reference" --out "$output" --workers "${DECISION_WORKERS:-6}"
python -m coph_stagewise.cli decision-diagnostics render --out "$output"
python - "$reference" "$output" "$decision_checkout" <<'PY'
import json,sys
from pathlib import Path
reference,output,source=map(Path,sys.argv[1:]);(output/'execution_context.json').write_text(json.dumps(dict(reproduced_from=str(reference),source_root=str(source)),indent=2)+'\n')
PY
