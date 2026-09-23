#!/usr/bin/env bash
set -euo pipefail

# End-to-end training -> held-out test -> long-horizon visualization.
# Run from the repository root:
#   bash run_conditional_scout_demo.sh

export PYTHONPATH="${PYTHONPATH:-}:CoPh-sense"
OUT="${OUT:-CoPh-sense/coph_terrain/results/conditional_scout_demo}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-80}"
NTRAIN="${NTRAIN:-10000}"
NVAL="${NVAL:-2000}"
NTEST="${NTEST:-4000}"
MAX_STEPS="${MAX_STEPS:-900}"
FRAME_STRIDE="${FRAME_STRIDE:-8}"

mkdir -p "$OUT"

python -m coph_terrain.conditional_scout_learning train \
  --outdir "$OUT/learning" \
  --n-train "$NTRAIN" --n-val "$NVAL" \
  --epochs "$EPOCHS" --device "$DEVICE"

CKPT="$OUT/learning/dispatch_value_best.pt"

python -m coph_terrain.conditional_scout_learning eval \
  --ckpt "$CKPT" --outdir "$OUT/test" \
  --n-test "$NTEST" --device "$DEVICE"

for CASE in \
  static_hidden_block_useful \
  static_local_known_no_need \
  moving_blocker_prediction \
  moving_cost_prediction \
  moving_blocker_stale_report
do
  for POLICY in never oracle learned; do
    EXTRA=()
    if [[ "$POLICY" == "learned" ]]; then
      EXTRA=(--ckpt "$CKPT")
    fi
    python -m coph_terrain.conditional_scout_long_horizon \
      --case "$CASE" --policy "$POLICY" \
      --outdir "$OUT/rollouts/${CASE}/${POLICY}" \
      --max-steps "$MAX_STEPS" --frame-stride "$FRAME_STRIDE" \
      --device "$DEVICE" "${EXTRA[@]}"
  done
done

# Deterministic dynamic-object demonstrations.  The v4 visualizer generates
# the scout sensing target ahead of the moving carrier at request time, so these
# runs show the complete request -> forward scout -> report -> route update loop.
for MODE in crossing clearing; do
  python -m coph_terrain.conditional_scout_long_horizon \
    --case moving_blocker_prediction --mode-name "$MODE" --policy oracle \
    --outdir "$OUT/rollouts/moving_blocker_prediction/oracle_${MODE}" \
    --max-steps "$MAX_STEPS" --frame-stride "$FRAME_STRIDE" --device "$DEVICE"
done

echo
echo "Done. Key outputs:"
echo "  $OUT/learning/train_history.png"
echo "  $OUT/test/test_metrics.json"
echo "  $OUT/test/value_prediction.png"
echo "  $OUT/rollouts/<case>/<policy>/trajectory.png"
echo "  $OUT/rollouts/<case>/<policy>/trajectory.gif"
echo "  $OUT/rollouts/<case>/<policy>/timeseries.png"
