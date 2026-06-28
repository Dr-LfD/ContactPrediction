"""Debug: why does the model fire on every frame of rollout test files?

Checks:
1. Raw sigmoid output distribution on rollout vs training frames
2. Image pixel statistics (mean/std) of rollout vs training HDF5s
3. Sanity check that the correct camera key is being read
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
import torch
import glob

CKPT = "data/outputs/2026.05.25/23.10.54_train_screwdriver_left_arm_selfplay_screwdriver_left_arm_wristview_with_rollout/checkpoints/epoch=0000-val_recall_edge_left_arm_screwdriver=1.0000.ckpt"
ROLLOUT_TEST_DIR = "data/rollout/screwdriver_test"
TRAINING_CONTACT_DIR = "data/training/screwdriver/contact"
CAMERA_KEY = "cam_left_wrist"
EDGE = "left_arm_screwdriver"

os.chdir(ROOT)

from contact_pred.scripts.common.contact_predictor_runtime import ContactPredictorRuntimeSession
from eval_contact_predictor_rollout import load_runtime_from_checkpoint

rs = ContactPredictorRuntimeSession(device="cuda")
runtime = load_runtime_from_checkpoint(CKPT, runtime_session=rs)
print("selected_camera_key:", runtime["selected_camera_key"])

# Get text embedding for the edge
embedding = rs.get_edge_text_embedding(runtime, EDGE)

def run_frames(frames_hwc, rs, runtime, embedding, label=""):
    """Run the model on a set of HWC uint8 frames, print raw sigmoid distribution."""
    from contact_pred.scripts.common.contact_predictor_runtime import build_edge_text_obs, build_model_obs_from_hwc_image
    shape_meta = (runtime.get("shape_meta") or {}).get("obs") or {}
    edge_attr = shape_meta.get("edge_text", {})
    preds = []
    for frame in frames_hwc:
        obs = {runtime["selected_camera_key"]: build_model_obs_from_hwc_image(frame)}
        if "edge_text" in shape_meta:
            obs["edge_text"] = build_edge_text_obs(embedding, edge_attr)
        p = rs.predict_label(runtime, obs)
        preds.append(p)
    preds = np.array(preds)
    print(f"\n  {label} (n={len(preds)})")
    print(f"    pixel mean/std: {frames_hwc.mean():.1f} / {frames_hwc.std():.1f}")
    print(f"    pred  min={preds.min():.4f}  max={preds.max():.4f}  mean={preds.mean():.4f}")
    print(f"    pred > 0.5: {(preds > 0.5).sum()}/{len(preds)}")
    print(f"    distribution: p<0.1={( preds<0.1).sum()} p<0.3={(preds<0.3).sum()} p<0.5={(preds<0.5).sum()} p>=0.5={(preds>=0.5).sum()} p>=0.9={(preds>=0.9).sum()}")
    return preds

# ── 1. Training HDF5 frames (should have good discrimination) ──────────────────
print("\n=== TRAINING HDF5 FRAMES ===")
train_files = sorted(glob.glob(f"{TRAINING_CONTACT_DIR}/episode_*_contact.hdf5"))[:3]
for fpath in train_files:
    with h5py.File(fpath, "r") as f:
        frames = f[f"observations/images/{CAMERA_KEY}"][:]  # (T,H,W,3)
        labels = f[f"label/{EDGE}"][:]
    pos_frames = frames[labels > 0.5]
    neg_frames = frames[labels < 0.5]
    if len(pos_frames) > 0:
        run_frames(pos_frames[:30], rs, runtime, embedding, f"  train POS {os.path.basename(fpath)}")
    if len(neg_frames) > 0:
        run_frames(neg_frames[:30], rs, runtime, embedding, f"  train NEG {os.path.basename(fpath)}")

# ── 2. Rollout test HDF5 frames ────────────────────────────────────────────────
print("\n=== ROLLOUT TEST HDF5 FRAMES ===")
rollout_files = sorted(glob.glob(f"{ROLLOUT_TEST_DIR}/*.hdf5"))
for fpath in rollout_files:
    with h5py.File(fpath, "r") as f:
        print(f"\n  FILE: {os.path.basename(fpath)}")
        print(f"    root keys: {sorted(f.keys())}")
        # Check which camera key the HDF5 actually has
        cam_key = None
        for k in f.keys():
            if "wrist" in k.lower() or "cam" in k.lower():
                cam_key = k
                break
        if cam_key is None:
            print("    WARNING: no wrist camera key found, skipping")
            continue
        frames = f[cam_key][:]  # (T,H,W,3)
        ann_key = f"annotations/{EDGE}/binary"
        if ann_key in f:
            labels = f[ann_key][:]
            pos_frames = frames[labels > 0.5]
            neg_frames = frames[labels < 0.5]
            if len(pos_frames) > 0:
                run_frames(pos_frames[:30], rs, runtime, embedding, f"  rollout POS {os.path.basename(fpath)}")
            if len(neg_frames) > 0:
                run_frames(neg_frames[:30], rs, runtime, embedding, f"  rollout NEG {os.path.basename(fpath)}")
        else:
            run_frames(frames[:30], rs, runtime, embedding, f"  rollout (no ann) {os.path.basename(fpath)}")

# ── 3. Self-play test HDF5 (all-zero, should all be negative) ─────────────────
print("\n=== SELF-PLAY VALIDATE HDF5 (all-zero ground truth) ===")
selfplay_val_dir = "data/selfplay/rollout/screwdriver/validate"
selfplay_files = sorted(glob.glob(f"{selfplay_val_dir}/*.hdf5"))
for fpath in selfplay_files[:2]:
    with h5py.File(fpath, "r") as f:
        print(f"\n  FILE: {os.path.basename(fpath)}")
        print(f"    root keys: {sorted(f.keys())}")
        # Try to find the camera key
        for k in f.keys():
            if "wrist" in k.lower() or "cam" in k.lower():
                frames = f[k][:]
                run_frames(frames[:30], rs, runtime, embedding, f"  selfplay {os.path.basename(fpath)} via key={k}")
                break
