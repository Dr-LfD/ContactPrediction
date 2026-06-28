#!/usr/bin/env bash
# Train the joint screwdriver+sponge+cup model, then eval all three edges.
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"

LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/train_joint_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "Training: train_screwdriver_sponge_cup_workspace"
echo "Log: $TRAIN_LOG"
echo "============================================================"

conda run -n dr-lfd python train.py \
    --config-name train_screwdriver_sponge_cup_workspace \
    2>&1 | tee "$TRAIN_LOG"

# Locate the latest checkpoint from the most recent output run matching this config.
CKPT=$(find "$WORKSPACE/data/outputs" -path "*/train_screwdriver_sponge_cup*" \
    -name "latest.ckpt" | sort | tail -1)

if [[ -z "$CKPT" ]]; then
    echo "ERROR: could not find latest.ckpt after training." >&2
    exit 1
fi

echo ""
echo "============================================================"
echo "Training complete. Checkpoint: $CKPT"
echo "============================================================"

# --- Eval: left_arm_screwdriver ---
echo ""
bash tools/eval_rollout_all.sh \
    "$CKPT" \
    left_arm_screwdriver \
    "$WORKSPACE/data/rollout/screwdriver/validate" \
    0.5

# --- Eval: right_arm_sponge ---
echo ""
bash tools/eval_rollout_all.sh \
    "$CKPT" \
    right_arm_sponge \
    "$WORKSPACE/data/rollout/sponge/validate" \
    0.5

# --- Eval: right_arm_cup ---
echo ""
bash tools/eval_rollout_all.sh \
    "$CKPT" \
    right_arm_cup \
    "$WORKSPACE/data/rollout/cup" \
    0.5

echo ""
echo "============================================================"
echo "All evals complete."
echo "============================================================"
