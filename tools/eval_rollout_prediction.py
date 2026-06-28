#!/usr/bin/env python
"""Evaluate contact-edge prediction quality against a human annotation.

Requires both predictions/<edge>/binary and annotations/<edge>/binary to exist
in the rollout HDF5 (run annotate_rollout.py first).

Usage::

    python tools/eval_rollout_prediction.py \\
        --hdf5 data/rollout/test_08_18.33.55.hdf5 \\
        --edge right_arm_cup

    # Also show a comparison plot
    python tools/eval_rollout_prediction.py \\
        --hdf5 data/rollout/test_08_18.33.55.hdf5 \\
        --edge right_arm_cup --plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Binary classification metrics from two (T,) uint8 arrays."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(( gt &  pred).sum())
    tn = int((~gt & ~pred).sum())
    fp = int((~gt &  pred).sum())
    fn = int(( gt & ~pred).sum())
    T  = len(gt)

    accuracy  = (tp + tn) / T
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else float("nan"))
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return dict(
        T=T,
        tp=tp, tn=tn, fp=fp, fn=fn,
        gt_positive=int(gt.sum()),
        pred_positive=int(pred.sum()),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
    )


def _print_report(edge: str, m: dict, threshold: float) -> None:
    nan_str = lambda v: f"{v:.4f}" if np.isfinite(v) else "  n/a "  # noqa: E731

    print(f"\n{'─' * 52}")
    print(f"  Edge : {edge}")
    print(f"  Frames : {m['T']}   threshold : {threshold:.3f}")
    print(f"{'─' * 52}")
    print(f"  GT  positives : {m['gt_positive']:>5}  ({100*m['gt_positive']/m['T']:.1f}%)")
    print(f"  Pred positives: {m['pred_positive']:>5}  ({100*m['pred_positive']/m['T']:.1f}%)")
    print(f"{'─' * 52}")
    print(f"  Confusion matrix (rows=GT, cols=pred):")
    print(f"              pred=0   pred=1")
    print(f"    gt=0      {m['tn']:>5}    {m['fp']:>5}")
    print(f"    gt=1      {m['fn']:>5}    {m['tp']:>5}")
    print(f"{'─' * 52}")
    print(f"  Accuracy    : {nan_str(m['accuracy'])}")
    print(f"  Precision   : {nan_str(m['precision'])}")
    print(f"  Recall      : {nan_str(m['recall'])}")
    print(f"  Specificity : {nan_str(m['specificity'])}")
    print(f"  F1          : {nan_str(m['f1'])}")
    print(f"{'─' * 52}\n")


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------


def _show_plot(
    edge: str,
    gt: np.ndarray,
    pred: np.ndarray,
    label: Optional[np.ndarray],
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(gt))
    has_label = label is not None
    nrows = 3 if has_label else 2
    height_ratios = [1.5, 1.5, 1.2] if has_label else [1.5, 1.5]

    fig, axes = plt.subplots(
        nrows, 1, figsize=(13, 5 if not has_label else 7),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.35},
    )
    fig.suptitle(f"Prediction quality — {edge}", fontsize=11)

    # Row 0: GT annotation
    axes[0].fill_between(x, gt.astype(float), step="mid", alpha=0.7,
                         color="#0072B2", label="GT annotation")
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].set_yticks([0, 1])
    axes[0].set_ylabel("GT", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)

    # Row 1: prediction binary + disagreement highlight
    axes[1].fill_between(x, pred.astype(float), step="mid", alpha=0.7,
                         color="#D55E00", label="prediction binary")
    disagree = (gt != pred).astype(float)
    axes[1].fill_between(x, disagree, step="mid", alpha=0.35,
                         color="#CC0000", label="disagreement")
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].set_yticks([0, 1])
    axes[1].set_ylabel("pred", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=8)

    # Row 2 (optional): raw probability + threshold
    if has_label:
        axes[2].plot(x, label, color="#D55E00", linewidth=1.0, label="probability")
        if np.isfinite(threshold):
            axes[2].axhline(threshold, color="#D55E00", linestyle=":", linewidth=1.0,
                            alpha=0.8, label=f"threshold={threshold:.2f}")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_ylabel("prob", fontsize=9)
        axes[2].set_xlabel("frame", fontsize=9)
        axes[2].legend(loc="upper right", fontsize=8)
    else:
        axes[1].set_xlabel("frame", fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--hdf5",
    "hdf5_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the rollout HDF5 file.",
)
@click.option(
    "--edge",
    required=True,
    help="Edge name (must exist under both predictions/ and annotations/).",
)
@click.option(
    "--plot",
    "show_plot",
    is_flag=True,
    default=False,
    help="Show a comparison timeline plot.",
)
def main(hdf5_path: Path, edge: str, show_plot: bool) -> None:
    """Print precision/recall/F1 of predictions vs human annotation."""
    try:
        with h5py.File(hdf5_path, "r") as root:
            pred_path = f"predictions/{edge}"
            ann_path  = f"annotations/{edge}"

            if pred_path not in root:
                raise click.ClickException(
                    f"{pred_path!r} not found — run eval_contact_predictor_rollout.py first."
                )
            if ann_path not in root:
                raise click.ClickException(
                    f"{ann_path!r} not found — run annotate_rollout.py first."
                )

            pred_group = root[pred_path]
            ann_group  = root[ann_path]

            pred_binary = np.asarray(pred_group["binary"][()], dtype=np.uint8)
            ann_binary  = np.asarray(ann_group["binary"][()],  dtype=np.uint8)

            if pred_binary.shape != ann_binary.shape:
                raise click.ClickException(
                    f"Shape mismatch: prediction {pred_binary.shape} vs "
                    f"annotation {ann_binary.shape}."
                )

            threshold = float(pred_group.attrs.get("threshold", float("nan")))
            label = (
                np.asarray(pred_group["label"][()], dtype=np.float32)
                if "label" in pred_group
                else None
            )

    except OSError as exc:
        raise click.ClickException(str(exc)) from exc

    metrics = _compute_metrics(ann_binary, pred_binary)
    _print_report(edge, metrics, threshold)

    if show_plot:
        _show_plot(edge, ann_binary, pred_binary, label, threshold)


if __name__ == "__main__":
    main()
