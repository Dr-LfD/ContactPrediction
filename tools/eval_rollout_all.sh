#!/usr/bin/env bash
# Evaluate a contact predictor checkpoint on all rollout HDF5s and report metrics.
#
# Usage:
#   bash tools/eval_rollout_all.sh <checkpoint> [edge] [rollout-dir] [threshold]
#
# Examples:
#   bash tools/eval_rollout_all.sh data/outputs/.../checkpoints/latest.ckpt
#   bash tools/eval_rollout_all.sh path/to/epoch=0100.ckpt right_arm_cup data/rollout 0.5

set -euo pipefail

CKPT="${1:?Usage: $0 <checkpoint> [edge] [rollout-dir] [threshold]}"
EDGE="${2:-right_arm_cup}"
ROLLOUT_DIR="${3:-data/rollout}"
THRESHOLD="${4:-0.5}"

echo "============================================================"
echo "Checkpoint : $CKPT"
echo "Edge       : $EDGE"
echo "Rollout dir: $ROLLOUT_DIR"
echo "Threshold  : $THRESHOLD"
echo "============================================================"

# Step 1 — write predictions into every rollout HDF5 (suppress verbose JSON stdout)
conda run -n dr-lfd python eval_contact_predictor_rollout.py \
    --checkpoint "$CKPT" \
    --rollout-dir "$ROLLOUT_DIR" \
    --edge-text "$EDGE" \
    --overwrite > /dev/null

echo ""
echo "--- Metrics ---"

# Step 2 — write metrics script to a temp file (conda run doesn't forward stdin).
# Pass config via env vars (quoted heredoc) to avoid shell injection into Python source.
METRICS_SCRIPT="$(mktemp /tmp/rollout_metrics_XXXXXX.py)"
trap 'rm -f "$METRICS_SCRIPT"' EXIT

cat > "$METRICS_SCRIPT" <<'PYEOF'
import h5py, numpy as np, glob, os, sys

rollout_dir = os.environ["ROLLOUT_DIR"]
edge        = os.environ["EDGE"]
threshold   = float(os.environ["THRESHOLD"])

all_gt, all_pred = [], []
skipped = 0
files = sorted(glob.glob(os.path.join(rollout_dir, "*.hdf5")))

for fpath in files:
    with h5py.File(fpath, "r") as f:
        gt_key   = f"annotations/{edge}/binary"
        pred_key = f"predictions/{edge}/label"
        if gt_key not in f or pred_key not in f:
            skipped += 1
            continue
        gt   = f[gt_key][:]
        pred = (f[pred_key][:] >= threshold).astype(int)
        all_gt.append(gt)
        all_pred.append(pred)
        tp = ((gt==1)&(pred==1)).sum(); fp = ((gt==0)&(pred==1)).sum()
        fn = ((gt==1)&(pred==0)).sum(); tn = ((gt==0)&(pred==0)).sum()
        print(f"  {os.path.basename(fpath):35s}  gt_pos={gt.sum():4d}  TP={tp:4d} FP={fp:4d} FN={fn:4d} TN={tn:4d}")

if skipped:
    print(f"  ({skipped} files skipped — missing ann or pred)", file=sys.stderr)

if not all_gt:
    print("ERROR: no files with both annotations and predictions found.", file=sys.stderr)
    sys.exit(1)

gt   = np.concatenate(all_gt)
pred = np.concatenate(all_pred)

TP = ((gt==1)&(pred==1)).sum()
FP = ((gt==0)&(pred==1)).sum()
FN = ((gt==1)&(pred==0)).sum()
TN = ((gt==0)&(pred==0)).sum()

precision = TP / (TP+FP+1e-9)
recall    = TP / (TP+FN+1e-9)
spec      = TN / (TN+FP+1e-9)
f1        = 2*precision*recall / (precision+recall+1e-9)

n_files = len(all_gt)
print()
print(f"=== AGGREGATE  edge={edge}  threshold={threshold}  files={n_files} ===")
print(f"Total frames : {len(gt)}")
print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {spec:.4f}")
print(f"F1           : {f1:.4f}")
PYEOF

ROLLOUT_DIR="$ROLLOUT_DIR" EDGE="$EDGE" THRESHOLD="$THRESHOLD" \
    conda run -n dr-lfd python3 "$METRICS_SCRIPT"
