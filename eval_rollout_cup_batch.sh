#!/usr/bin/env bash
# Batch eval_rollout_cup (see .vscode/launch.json) for all rollout HDF5 files.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_DIR="${ROLLOUT_DIR:-${WORKSPACE}/data/rollout/sponge/validate}"
CHECKPOINT="${CHECKPOINT:-${WORKSPACE}/data/outputs/2026.05.23/11.45.32_train_clean_cup_with_rollout_clean_cup_right_arm_wristview_with_rollout/checkpoints/latest.ckpt}"
EDGE_TEXT="right_arm_sponge"
THRESHOLD="0.5"

shopt -s nullglob
rollouts=("${ROLLOUT_DIR}"/*.hdf5)
if ((${#rollouts[@]} == 0)); then
  echo "No HDF5 files found in ${ROLLOUT_DIR}" >&2
  exit 1
fi

for rollout in "${rollouts[@]}"; do
  python "${WORKSPACE}/eval_contact_predictor_rollout.py" \
    --checkpoint "${CHECKPOINT}" \
    --rollout "${rollout}" \
    --edge-text "${EDGE_TEXT}" \
    --threshold "${THRESHOLD}" \
    --overwrite > /dev/null 2>&1
done

echo "Processed ${#rollouts[@]} rollout(s). Aggregate metrics:"
python - <<EOF
import h5py, numpy as np, glob

edge = "${EDGE_TEXT}"
threshold = ${THRESHOLD}
files = sorted(glob.glob("${ROLLOUT_DIR}/*.hdf5"))

tp = fp = fn = tn = 0
for path in files:
    with h5py.File(path, "r") as f:
        gt   = f[f"annotations/{edge}/binary"][()].astype(bool)
        pred = f[f"predictions/{edge}/label"][()] > threshold
    tp += int(( gt &  pred).sum())
    fp += int((~gt &  pred).sum())
    fn += int(( gt & ~pred).sum())
    tn += int((~gt & ~pred).sum())

T        = tp + fp + fn + tn
accuracy = (tp + tn) / T
precision = tp / (tp + fp) if (tp + fp) else float("nan")
recall    = tp / (tp + fn) if (tp + fn) else float("nan")
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
fpr       = fp / (fp + tn) if (fp + tn) else float("nan")
fnr       = fn / (fn + tp) if (fn + tp) else float("nan")
specificity = tn / (tn + fp) if (tn + fp) else float("nan")

print(f"  Files     : {len(files)}   Frames: {T}   GT+: {tp+fn}")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  Accuracy    : {accuracy:.4f}")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"  Specificity : {specificity:.4f}")
print(f"  FPR         : {fpr:.4f}")
print(f"  FNR         : {fnr:.4f}")
print(f"  F1          : {f1:.4f}")
EOF
