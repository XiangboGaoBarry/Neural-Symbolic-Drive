#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-xiangbog/Neural-Symbolic-Drive}"

python -m neural_symbolic_drive.infer \
  --model_path "$MODEL_PATH" \
  --data_json examples/neural_symbolic_drive_examples.json \
  --save_path results/example_predictions.jsonl \
  --max_new_tokens 2048
