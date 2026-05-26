#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/pipeline/raw_to_predictions.yaml}

python3 scripts/01_prepare_data.py --config "$CONFIG"
python3 scripts/02_train.py --config "$CONFIG"
