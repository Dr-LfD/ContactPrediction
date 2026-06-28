#!/usr/bin/env python3
"""Convert annotated rollout HDF5 files to contact-training HDF5 format.

Usage:
    python tools/convert_rollout_to_training.py \\
        --rollout-dir data/rollout \\
        --output-dir data/training/handoff_cup_rollout/contact \\
        --pattern "test_08_*.hdf5"

Output layout (per episode):
    /observations/images/<camera>  (T,128,128,3) uint8
    /label/<edge>  (T,) float32
    /valid/<edge>  (T,) bool
    root attr source_file: source basename
"""

import argparse
import glob
import os

import h5py
import numpy as np


def _assert_uint8_images(name: str, data: np.ndarray, source: str) -> None:
    if data.dtype != np.uint8:
        raise ValueError(f"[{source}] {name}: expected uint8, got {data.dtype}")
    if data.ndim != 4 or data.shape[1:] != (128, 128, 3):
        raise ValueError(f"[{source}] {name}: expected (T,128,128,3), got {data.shape}")


def _assert_binary_1d(name: str, data: np.ndarray, source: str) -> None:
    if data.ndim != 1:
        raise ValueError(f"[{source}] {name}: expected (T,), got {data.shape}")
    if not np.isin(data, (0, 1)).all():
        raise ValueError(f"[{source}] {name}: values must be in {{0, 1}}")


def _convert_one(
    index: int,
    source_path: str,
    output_dir: str,
    edge: str,
    camera: str,
    all_negative_only: bool,
):
    """Return (pos, neg) on success, None if skipped by --all-negative-only."""
    source_name = os.path.basename(source_path)
    annotation_path = f"annotations/{edge}/binary"

    with h5py.File(source_path, "r") as src:
        if annotation_path not in src:
            raise ValueError(f"[{source_name}] missing required dataset '{annotation_path}'")

        cam: np.ndarray = src[camera][()]  # type: ignore[index, assignment]
        annotations: np.ndarray = src[annotation_path][()]  # type: ignore[index, assignment]

        _assert_uint8_images(camera, cam, source_name)
        _assert_binary_1d(annotation_path, annotations, source_name)

        T = cam.shape[0]
        if annotations.shape[0] != T:
            raise ValueError(
                f"[{source_name}] T mismatch: {camera}={T}, {annotation_path}={annotations.shape[0]}"
            )

        pos = int(annotations.sum())
        if all_negative_only and pos > 0:
            print(f"  [{index:02d}] {source_name}: skipped ({pos} positive frames)")
            return None

        labels = annotations.astype(np.float32)
        valid = np.ones(T, dtype=bool)

    output_path = os.path.join(output_dir, f"episode_{index}_contact.hdf5")
    with h5py.File(output_path, "w") as dst:
        dst.create_dataset(f"/observations/images/{camera}", data=cam)
        dst.create_dataset(f"/label/{edge}", data=labels)
        dst.create_dataset(f"/valid/{edge}", data=valid)
        dst.attrs["source_file"] = source_name

    neg = T - pos
    print(f"  [{index:02d}] {source_name}: T={T} pos={pos} neg={neg}")
    return pos, neg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert annotated rollout HDF5 files to training format."
    )
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="test_08_*.hdf5")
    parser.add_argument("--edge", default="right_arm_cup", help="Contact edge name")
    parser.add_argument("--camera", default="cam_right_wrist", help="Source camera key")
    parser.add_argument(
        "--all-negative-only",
        action="store_true",
        help="Only include episodes where every annotated frame is negative (contact=0).",
    )
    args = parser.parse_args()

    source_paths = sorted(glob.glob(os.path.join(args.rollout_dir, args.pattern)))
    if not source_paths:
        raise RuntimeError(f"No files matched {os.path.join(args.rollout_dir, args.pattern)}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Converting {len(source_paths)} files → {args.output_dir}")

    out_index = 0
    total_pos = total_neg = 0
    for path in source_paths:
        result = _convert_one(out_index, path, args.output_dir, args.edge, args.camera, args.all_negative_only)
        if result is not None:
            total_pos += result[0]
            total_neg += result[1]
            out_index += 1

    print(f"\nDone: {out_index} episodes written, total_pos={total_pos} total_neg={total_neg}")


if __name__ == "__main__":
    main()
