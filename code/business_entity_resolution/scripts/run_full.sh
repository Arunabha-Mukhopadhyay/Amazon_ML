#!/usr/bin/env bash
# Full production pipeline: train on the complete training data, then predict the test set.
# Usage:
#   bash code/business_entity_resolution/scripts/run_full.sh
#   or from code/business_entity_resolution/:
#   bash scripts/run_full.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$CODE_DIR/../.." && pwd)"

# Auto-detect DATA directory containing dataset/ and utils/
if [[ -n "${DATA:-}" ]]; then
  DATA_DIR="$DATA"
elif [[ -d "$CODE_DIR/../../dataset" ]]; then
  DATA_DIR="$CODE_DIR/../.."
elif [[ -d "$CODE_DIR/dataset" ]]; then
  DATA_DIR="$CODE_DIR"
elif [[ -d "$HOME/student_resource/dataset" ]]; then
  DATA_DIR="$HOME/student_resource"
elif [[ -d "$HOME/data/dataset" ]]; then
  DATA_DIR="$HOME/data"
else
  DATA_DIR="."
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
echo "Using DATA_DIR: $DATA_DIR"
echo "Using CODE_DIR: $CODE_DIR"

cd "$CODE_DIR"
mkdir -p logs "$DATA_DIR/output"

# 1. Train on full dataset
"$SCRIPT_DIR/py" -m er.run train --train-dir "$DATA_DIR/dataset/train" --work work/full "$@" 2>&1 | tee logs/train.log

# 2. Predict on full test set and validate
"$SCRIPT_DIR/py" -m er.run predict --test-dir "$DATA_DIR/dataset/test" --work work/full --out-dir "$DATA_DIR/output" \
  --validator "$DATA_DIR/utils/validate_submission.py" 2>&1 | tee logs/predict.log

echo "=== DONE: short summary to paste back ==="
grep -hE "TRAIN SUMMARY|PREDICT SUMMARY|blocking:|PASS|FAIL|WARNING" logs/train.log logs/predict.log
