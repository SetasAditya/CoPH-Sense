#!/usr/bin/env bash
# Run the frozen sources without touching a concurrently edited working tree.
set -euo pipefail
cd "$(dirname "$0")/.."
reference=$(realpath "${1:-results/attribution_v1}")
output=$(realpath -m "${2:-results/attribution_reproduction}")
attribution_checkout=$(mktemp -d /tmp/coph-attribution-reproduction.XXXXXX)
python - "$reference" "$attribution_checkout" <<'PY'
import hashlib,json,sys,zipfile
from pathlib import Path
reference,checkout=map(Path,sys.argv[1:]);m=json.loads((reference/'manifest.json').read_text())
for filename,inventory,base in [('sources.zip',m['sources'],checkout/'CoPH-Stagewise'),('documents.zip',m['documents'],checkout)]:
    with zipfile.ZipFile(reference/filename) as archive:
        for name,sha in inventory.items():
            relative=Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            data=archive.read(name);assert hashlib.sha256(data).hexdigest()==sha
            destination=base/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
print('Archived execution checkout:',checkout)
PY
cd "$attribution_checkout/CoPH-Stagewise"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl MPLBACKEND=Agg
python -m coph_stagewise.attribution_suite suite --out "$output" --profile all --workers "${ATTRIBUTION_WORKERS:-4}"
python -m coph_stagewise.attribution_suite render --out "$output"
python - "$reference" "$output" "$attribution_checkout/CoPH-Stagewise" <<'PY'
import json,shutil,sys
from pathlib import Path
reference,output,source=map(Path,sys.argv[1:])
shutil.copyfile(reference/'documents.zip',output/'documents.zip')
(output/'execution_context.json').write_text(json.dumps({
    'frozen_source_root':str(source),
    'reproduced_from':str(reference),
    'fingerprint':json.loads((output/'manifest.json').read_text())['fingerprint'],
},indent=2)+'\n')
PY
