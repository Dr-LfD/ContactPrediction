#!/usr/bin/env python
"""Matplotlib GUI for annotating contact-edge masks in rollout HDF5 files.

Usage::

    # List available edges
    python tools/annotate_rollout.py --hdf5 data/rollout/test_08_18.33.55.hdf5

    # Annotate one edge (seeds annotation from model prediction)
    python tools/annotate_rollout.py \\
        --hdf5 data/rollout/test_08_18.33.55.hdf5 --edge right_arm_cup

    # Resume a prior annotation session
    python tools/annotate_rollout.py \\
        --hdf5 data/rollout/test_08_18.33.55.hdf5 --edge right_arm_cup \\
        --resume-existing

Keyboard controls::

    ← / →       step ±1 frame
    ↑ / ↓       step ±10 frames
    Home / End  first / last frame
    Space       play / pause
    I           set in-point at current frame
    O           fill [in-point, current] with contact=1, clear in-point
    D           delete the positive interval containing the current frame
    N / P       jump to next / previous disagreement with prediction seed
    C           clear all — set entire annotation to no-contact (all zeros)
    R           reset annotation to prediction seed
    S           save to annotations/<edge>/binary in the HDF5
    Q           quit (press twice if there are unsaved changes)

Written HDF5 layout::

    annotations/<edge>/
        binary  (T,) uint8
        attrs:
            annotator
            annotated_at_utc
            source_predictions_path
            source_predictions_threshold
            edge_text
            edits_count
"""

from __future__ import annotations

import getpass
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


_LEFT_CAM_DEFAULT = "cam_left_wrist"
_RIGHT_CAM_DEFAULT = "cam_right_wrist"

_CONTROLS_HELP = (
    "Controls: ←/→ step ±1 | ↑/↓ step ±10 | Home/End first/last | "
    "Space play/pause | I in-point | O out-point | D delete interval | "
    "N/P jump disagreement | C clear all | R reset to seed | S save | Q quit"
)


# ---------------------------------------------------------------------------
# Pure utilities
# ---------------------------------------------------------------------------


def _validate_edge_name(edge: str) -> None:
    if not edge or "/" in edge or edge.startswith("_"):
        raise ValueError(
            f"Edge name {edge!r} is not safe as an HDF5 group name "
            "(must be non-empty, must not contain '/', must not start with '_')."
        )


def _binary_checksum(values: np.ndarray) -> str:
    data = np.ascontiguousarray(values.astype(np.uint8, copy=False))
    return hashlib.blake2b(data.tobytes(), digest_size=16).hexdigest()


def _annotator_name() -> str:
    for key in ("USER", "USERNAME", "LOGNAME"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return getpass.getuser()
    except OSError:
        return "unknown"


def _read_camera(root: h5py.File, name: str, num_frames: int) -> np.ndarray:
    if name not in root:
        raise ValueError(f"Camera dataset {name!r} not found in HDF5.")
    ds = root[name]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"{name!r} exists but is not a dataset.")
    if ds.ndim != 4 or ds.shape[0] != num_frames or ds.shape[-1] != 3:
        raise ValueError(
            f"{name!r} must have shape (T={num_frames}, H, W, 3); got {ds.shape}."
        )
    if ds.dtype != np.uint8:
        raise ValueError(f"{name!r} must be uint8; got {ds.dtype}.")
    return ds[()]  # type: ignore[return-value]


def _read_binary_dataset(ds: h5py.Dataset, path: str, num_frames: int) -> np.ndarray:
    values = np.asarray(ds[()])
    if values.shape != (num_frames,):
        raise ValueError(
            f"{path} must have shape ({num_frames},); got {values.shape}."
        )
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{path} contains values outside {{0, 1}}.")
    return values.astype(np.uint8, copy=False)


def _read_prediction_label(group: h5py.Group, num_frames: int) -> Optional[np.ndarray]:
    if "label" not in group:
        return None
    values = np.asarray(group["label"][()], dtype=np.float32)
    if values.shape != (num_frames,):
        raise ValueError(
            f"{group.name}/label must have shape ({num_frames},); got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{group.name}/label contains non-finite values.")
    return values


def _read_gripper_state(root: h5py.File, num_frames: int) -> Optional[np.ndarray]:
    if "gripper_state" not in root:
        return None
    ds = root["gripper_state"]
    if not isinstance(ds, h5py.Dataset):
        return None
    values = np.asarray(ds[()], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != num_frames or values.shape[1] < 2:
        raise ValueError(
            f"gripper_state must have shape (T={num_frames}, 2+); got {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("gripper_state contains non-finite values.")
    return values


def _attr_float(attrs: h5py.AttributeManager, key: str) -> float:
    if key not in attrs:
        return float("nan")
    try:
        return float(attrs[key])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def list_prediction_edges(path: Path) -> list[str]:
    """Return sorted edge names that have a predictions/<edge>/binary dataset."""
    with h5py.File(path, "r") as root:
        if "predictions" not in root:
            return []
        predictions = root["predictions"]
        if not isinstance(predictions, h5py.Group):
            return []
        return sorted(
            name
            for name, item in predictions.items()
            if isinstance(item, h5py.Group) and "binary" in item
        )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutData:
    path: Path
    edge: str
    fps: float
    num_frames: int
    left_camera: str
    right_camera: str
    left_frames: np.ndarray           # (T, H, W, 3) uint8
    right_frames: np.ndarray          # (T, H, W, 3) uint8
    seed_binary: np.ndarray           # (T,) uint8
    initial_binary: np.ndarray        # (T,) uint8 — starting point for annotation
    prediction_label: Optional[np.ndarray]   # (T,) float32 | None
    gripper_state: Optional[np.ndarray]      # (T, 2+) float32 | None
    source_threshold: float
    existing_annotation_present: bool
    existing_annotation_checksum: Optional[str]
    base_edits_count: int


def load_rollout_data(
    path: Path,
    edge: str,
    left_camera: str,
    right_camera: str,
    resume_existing: bool,
    overwrite: bool,
) -> RolloutData:
    _validate_edge_name(edge)

    with h5py.File(path, "r") as root:
        if "num_frames" not in root.attrs:
            raise ValueError("HDF5 is missing root attribute 'num_frames'.")
        num_frames = int(root.attrs["num_frames"])
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive; got {num_frames}.")

        try:
            fps = float(root.attrs.get("fps", 10.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            fps = 10.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 10.0

        left_frames = _read_camera(root, left_camera, num_frames)
        right_frames = _read_camera(root, right_camera, num_frames)

        pred_path = f"predictions/{edge}"
        if pred_path in root:
            pred_group = root[pred_path]
            if not isinstance(pred_group, h5py.Group) or "binary" not in pred_group:
                raise ValueError(f"{pred_path} is missing the 'binary' dataset.")
            seed_binary = _read_binary_dataset(
                pred_group["binary"], f"{pred_path}/binary", num_frames  # type: ignore[arg-type]
            )
            prediction_label = _read_prediction_label(pred_group, num_frames)
            source_threshold = _attr_float(pred_group.attrs, "threshold")
        else:
            seed_binary = np.zeros(num_frames, dtype=np.uint8)
            prediction_label = None
            source_threshold = float("nan")
        gripper_state = _read_gripper_state(root, num_frames)

        ann_path = f"annotations/{edge}"
        existing_present = ann_path in root
        existing_checksum: Optional[str] = None
        base_edits_count = 0

        if existing_present:
            ann_group = root[ann_path]
            if not isinstance(ann_group, h5py.Group) or "binary" not in ann_group:
                raise ValueError(f"{ann_path} exists but has no 'binary' dataset.")
            existing_binary = _read_binary_dataset(
                ann_group["binary"], f"{ann_path}/binary", num_frames  # type: ignore[arg-type]
            )
            existing_checksum = _binary_checksum(existing_binary)
            base_edits_count = int(ann_group.attrs.get("edits_count", 0))

            if resume_existing:
                initial_binary = existing_binary.copy()
            elif not overwrite:
                raise RuntimeError(
                    f"{ann_path} already exists. Use --resume-existing to continue "
                    "editing it, or --overwrite to start fresh from all-zero."
                )
            else:
                initial_binary = np.zeros(num_frames, dtype=np.uint8)
                base_edits_count = 0
        else:
            if resume_existing:
                raise RuntimeError(
                    f"{ann_path} does not exist yet — cannot resume. "
                    "Remove --resume-existing to start from the prediction seed."
                )
            initial_binary = np.zeros(num_frames, dtype=np.uint8)

    return RolloutData(
        path=path,
        edge=edge,
        fps=fps,
        num_frames=num_frames,
        left_camera=left_camera,
        right_camera=right_camera,
        left_frames=left_frames,
        right_frames=right_frames,
        seed_binary=seed_binary,
        initial_binary=initial_binary,
        prediction_label=prediction_label,
        gripper_state=gripper_state,
        source_threshold=source_threshold,
        existing_annotation_present=existing_present,
        existing_annotation_checksum=existing_checksum,
        base_edits_count=base_edits_count,
    )


# ---------------------------------------------------------------------------
# HDF5 writeback
# ---------------------------------------------------------------------------


def write_annotation_atomic(
    path: Path,
    edge: str,
    binary_values: np.ndarray,
    source_threshold: float,
    edits_count: int,
    allow_replace: bool,
    expected_existing_present: bool,
    expected_existing_checksum: Optional[str],
) -> None:
    """Atomically write annotations/<edge>/binary via a staged rename.

    Mirrors write_edge_predictions() in eval_contact_predictor_rollout.py.
    Guards against concurrent modifications by comparing a checksum of the
    on-disk annotation against the one recorded at session load time.
    """
    _validate_edge_name(edge)
    binary_values = np.ascontiguousarray(binary_values, dtype=np.uint8)
    pending_name = f"_pending_{edge}"
    backup_name = f"_backup_{edge}"
    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    with h5py.File(path, "r+") as root:
        annotations = root.require_group("annotations")
        target_exists = edge in annotations

        # Concurrent-edit guard.
        if target_exists:
            if not allow_replace:
                raise RuntimeError(
                    f"annotations/{edge} already exists and replacement is disabled."
                )
            if not expected_existing_present:
                raise RuntimeError(
                    f"annotations/{edge} appeared after this session started. "
                    "Reload the file before saving."
                )
            on_disk = _read_binary_dataset(
                annotations[edge]["binary"],  # type: ignore[index]
                f"annotations/{edge}/binary",
                binary_values.shape[0],
            )
            if _binary_checksum(on_disk) != expected_existing_checksum:
                raise RuntimeError(
                    f"annotations/{edge} was modified by another process since this "
                    "session started. Reload the file before saving."
                )
        elif expected_existing_present:
            raise RuntimeError(
                f"annotations/{edge} disappeared after this session started. "
                "Reload the file before saving."
            )

        # Clean stale scratch groups from a prior aborted run.
        for scratch in (pending_name, backup_name):
            if scratch in annotations:
                del annotations[scratch]

        # Stage new content before mutating the committed group.
        pending = annotations.create_group(pending_name)
        try:
            pending.create_dataset("binary", data=binary_values, dtype=np.uint8)
            pending.attrs["annotator"] = _annotator_name()
            pending.attrs["annotated_at_utc"] = timestamp_utc
            pending.attrs["source_predictions_path"] = f"predictions/{edge}"
            pending.attrs["source_predictions_threshold"] = float(source_threshold)
            pending.attrs["edge_text"] = edge
            pending.attrs["edits_count"] = int(edits_count)
        except Exception:
            if pending_name in annotations:
                del annotations[pending_name]
            raise

        # Atomic-ish replace via backup → rename → drop backup.
        if target_exists:
            annotations.move(edge, backup_name)
        try:
            annotations.move(pending_name, edge)
        except Exception:
            if pending_name in annotations:
                del annotations[pending_name]
            if backup_name in annotations:
                annotations.move(backup_name, edge)
            raise
        if backup_name in annotations:
            del annotations[backup_name]

        root.flush()


# ---------------------------------------------------------------------------
# Annotation state
# ---------------------------------------------------------------------------


class AnnotationState:
    def __init__(self, data: RolloutData) -> None:
        self.data = data
        self.annotation = data.initial_binary.copy()
        self.current_frame: int = 0
        self.in_point: Optional[int] = None
        self.playing: bool = False
        self.dirty: bool = False
        self.quit_armed: bool = False
        self.edits_count: int = data.base_edits_count

    def set_frame(self, frame: int) -> None:
        self.current_frame = int(np.clip(frame, 0, self.data.num_frames - 1))
        self.quit_armed = False

    def set_in_point(self) -> str:
        self.in_point = self.current_frame
        self.quit_armed = False
        return f"In-point set at frame {self.current_frame + 1}."

    def fill_to_current(self) -> str:
        if self.in_point is None:
            return "Set an in-point with I before using O."
        start, end = sorted((self.in_point, self.current_frame))
        before = self.annotation[start : end + 1].copy()
        self.annotation[start : end + 1] = 1
        self.in_point = None
        if np.array_equal(before, self.annotation[start : end + 1]):
            return f"Interval [{start + 1}, {end + 1}] was already all positive."
        self._record_edit()
        return f"Marked contact for frames {start + 1}–{end + 1}."

    def delete_current_interval(self) -> str:
        frame = self.current_frame
        if self.annotation[frame] == 0:
            return f"Frame {frame + 1} is not inside a positive interval."
        start = frame
        while start > 0 and self.annotation[start - 1] == 1:
            start -= 1
        end = frame
        while end + 1 < self.data.num_frames and self.annotation[end + 1] == 1:
            end += 1
        self.annotation[start : end + 1] = 0
        self._record_edit()
        return f"Deleted contact interval {start + 1}–{end + 1}."

    def clear_all(self) -> str:
        if not self.annotation.any():
            return "Annotation is already all-zero (no contact)."
        self.annotation[:] = 0
        self.in_point = None
        self._record_edit()
        return "Cleared all contact — annotation is now all-zero."

    def reset_to_seed(self) -> str:
        if np.array_equal(self.annotation, self.data.seed_binary):
            return "Annotation already matches prediction seed."
        self.annotation[:] = self.data.seed_binary
        self.in_point = None
        self._record_edit()
        return "Reset annotation to prediction seed."

    def jump_disagreement(self, direction: int) -> str:
        disagreements = np.flatnonzero(self.annotation != self.data.seed_binary)
        if disagreements.size == 0:
            return "No disagreements between annotation and prediction seed."
        if direction >= 0:
            ahead = disagreements[disagreements > self.current_frame]
            target = int(ahead[0] if ahead.size else disagreements[0])
        else:
            behind = disagreements[disagreements < self.current_frame]
            target = int(behind[-1] if behind.size else disagreements[-1])
        self.set_frame(target)
        return f"Jumped to disagreement at frame {target + 1}."

    def mark_saved(self) -> None:
        self.dirty = False
        self.quit_armed = False

    def _record_edit(self) -> None:
        self.edits_count += 1
        self.dirty = True
        self.quit_armed = False


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------


class RolloutView:
    def __init__(self, state: AnnotationState) -> None:
        self.state = state
        data = state.data
        T = data.num_frames
        x = np.arange(T)

        has_traces = data.prediction_label is not None or data.gripper_state is not None
        nrows = 4 if has_traces else 3
        height_ratios = [5.0, 1.2, 1.2, 0.35] if has_traces else [5.0, 1.5, 0.35]

        self.figure = plt.figure(figsize=(13, 8 if has_traces else 7))
        fig_manager = getattr(self.figure.canvas, "manager", None)
        if fig_manager is not None:
            try:
                fig_manager.set_window_title(f"Annotate rollout — {data.edge}")
            except Exception:
                pass

        grid = self.figure.add_gridspec(
            nrows, 2,
            height_ratios=height_ratios,
            hspace=0.45,
            wspace=0.06,
        )

        self.left_ax = self.figure.add_subplot(grid[0, 0])
        self.right_ax = self.figure.add_subplot(grid[0, 1])
        self.timeline_ax = self.figure.add_subplot(grid[1, :])
        self.trace_ax: Optional[plt.Axes] = (  # type: ignore[name-defined]
            self.figure.add_subplot(grid[2, :], sharex=self.timeline_ax)
            if has_traces
            else None
        )
        self.slider_ax = self.figure.add_subplot(grid[nrows - 1, :])

        # Camera images — both wrist cams side-by-side.
        self.left_image = self.left_ax.imshow(data.left_frames[0])
        self.right_image = self.right_ax.imshow(data.right_frames[0])
        for ax in (self.left_ax, self.right_ax):
            ax.set_xticks([])
            ax.set_yticks([])

        # Timeline: seed binary (gray, lower band) + annotation (blue, upper band).
        # seed ∈ [0, 0.8], annotation ∈ [1.2, 2.0] — visually separated.
        self._seed_ydata = data.seed_binary * 0.8
        self._ann_ydata = state.annotation * 0.8 + 1.2

        (self._seed_line,) = self.timeline_ax.plot(
            x, self._seed_ydata,
            drawstyle="steps-mid", color="0.50", linewidth=1.2, label="prediction seed",
        )
        (self._ann_line,) = self.timeline_ax.plot(
            x, self._ann_ydata,
            drawstyle="steps-mid", color="#0072B2", linewidth=2.0, label="annotation",
        )
        self.timeline_ax.set_ylim(-0.15, 2.25)
        self.timeline_ax.set_yticks([0.0, 0.8, 1.2, 2.0])
        self.timeline_ax.set_yticklabels(
            ["seed=0", "seed=1", "ann=0", "ann=1"], fontsize=8
        )
        self.timeline_ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.6)
        self.timeline_ax.set_xlabel("frame", fontsize=8)

        # Probability + gripper traces.
        if has_traces:
            assert self.trace_ax is not None
            if data.prediction_label is not None:
                self.trace_ax.plot(
                    x, data.prediction_label,
                    color="#D55E00", linewidth=1.2, label="contact prob",
                )
                if np.isfinite(data.source_threshold):
                    self.trace_ax.axhline(
                        data.source_threshold,
                        color="#D55E00", linestyle=":", linewidth=1.0, alpha=0.7,
                    )
                self.trace_ax.set_ylim(-0.05, 1.05)
                self.trace_ax.set_ylabel("prob", fontsize=8)
                self.trace_ax.legend(loc="upper right", fontsize=8, framealpha=0.6)

            if data.gripper_state is not None:
                gripper_ax = self.trace_ax.twinx()
                gripper_ax.plot(
                    x, data.gripper_state[:, 0],
                    color="#009E73", alpha=0.65, linewidth=1.0, label="gripper L",
                )
                gripper_ax.plot(
                    x, data.gripper_state[:, 1],
                    color="#CC79A7", alpha=0.65, linewidth=1.0, label="gripper R",
                )
                gripper_ax.set_ylabel("grip", fontsize=8)
                gripper_ax.legend(loc="upper left", fontsize=8, framealpha=0.6)

        # Red vertical line marking the current frame.
        _vline_axes = [self.timeline_ax]
        if self.trace_ax is not None:
            _vline_axes.append(self.trace_ax)
        self._current_lines = [
            ax.axvline(0, color="#D62728", linewidth=1.0, zorder=5)
            for ax in _vline_axes
        ]
        # Yellow dashed line marking the in-point (hidden until I is pressed).
        self._in_line = self.timeline_ax.axvline(
            0, color="#E69F00", linewidth=1.2, linestyle="--", visible=False, zorder=4,
        )

        # Frame slider — _syncing_slider guards against the on_changed feedback loop.
        self._syncing_slider = False
        self.slider = Slider(
            self.slider_ax, "Frame", 0, max(T - 1, 1),
            valinit=0, valstep=1, valfmt="%0.0f",
        )

        self._status_text = self.figure.text(0.01, 0.005, "", fontsize=9, va="bottom")
        self.refresh(_CONTROLS_HELP)

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def refresh(self, message: Optional[str] = None) -> None:
        """Full redraw: frame images, annotation trace, status, then draw_idle."""
        self.refresh_frame()
        self.refresh_annotation()
        self._refresh_status(message)
        self.figure.canvas.draw_idle()

    def refresh_frame(self) -> None:
        """Update camera images and vertical frame markers only (O(1))."""
        frame = self.state.current_frame
        data = self.state.data
        self.left_image.set_data(data.left_frames[frame])
        self.right_image.set_data(data.right_frames[frame])
        self.left_ax.set_title(
            f"{data.left_camera}  [{frame + 1}/{data.num_frames}]", fontsize=9
        )
        self.right_ax.set_title(
            f"{data.right_camera}  [{frame + 1}/{data.num_frames}]", fontsize=9
        )
        for line in self._current_lines:
            line.set_xdata([frame, frame])
        if self.state.in_point is not None:
            self._in_line.set_xdata([self.state.in_point, self.state.in_point])
            self._in_line.set_visible(True)
        else:
            self._in_line.set_visible(False)
        self._syncing_slider = True
        try:
            self.slider.set_val(frame)
        finally:
            self._syncing_slider = False

    def refresh_annotation(self) -> None:
        """Update the annotation timeline trace after an edit."""
        self._ann_ydata = self.state.annotation * 0.8 + 1.2
        self._ann_line.set_ydata(self._ann_ydata)

    def _refresh_status(self, message: Optional[str] = None) -> None:
        dirty_marker = "* " if self.state.dirty else ""
        in_str = (
            f"  I={self.state.in_point + 1}"
            if self.state.in_point is not None
            else ""
        )
        msg_str = f"   {message}" if message else ""
        self._status_text.set_text(
            f"{dirty_marker}edge={self.state.data.edge}  "
            f"frame={self.state.current_frame + 1}/{self.state.data.num_frames}"
            f"{in_str}  edits={self.state.edits_count}{msg_str}"
        )

    @property
    def syncing_slider(self) -> bool:
        return self._syncing_slider


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class RolloutAnnotator:
    def __init__(self, data: RolloutData, replace_existing: bool) -> None:
        self.state = AnnotationState(data)
        self.view = RolloutView(self.state)
        self._replace_existing = replace_existing
        self._expected_existing_present = data.existing_annotation_present
        self._expected_existing_checksum = data.existing_annotation_checksum

        # Disable all matplotlib default keybindings so they don't shadow ours.
        # e.g. 'o' → zoom, 's' → save dialog, 'r'/'home' → reset view,
        #      'p' → pan, 'c' → back, 'left'/'right' → history navigation.
        for rckey in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[rckey] = []

        interval_ms = max(1, int(round(1000.0 / data.fps)))
        self._timer = self.view.figure.canvas.new_timer(interval=interval_ms)
        self._timer.add_callback(self._on_timer)

        canvas = self.view.figure.canvas
        canvas.mpl_connect("key_press_event", self._on_key)
        self.view.slider.on_changed(self._on_slider)

    def show(self) -> None:
        plt.show()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_slider(self, value: float) -> None:
        if self.view.syncing_slider:
            return
        self.state.set_frame(int(round(value)))
        self.view.refresh_frame()
        self.view._refresh_status()
        self.view.figure.canvas.draw_idle()

    def _on_timer(self) -> bool:
        if not self.state.playing:
            return True
        if self.state.current_frame >= self.state.data.num_frames - 1:
            self.state.playing = False
            self._timer.stop()
            self.view.refresh("Reached last frame.")
            return True
        self.state.set_frame(self.state.current_frame + 1)
        self.view.refresh_frame()
        self.view._refresh_status()
        self.view.figure.canvas.draw_idle()
        return True

    def _toggle_play(self) -> str:
        self.state.playing = not self.state.playing
        if self.state.playing:
            self._timer.start()
            return "Playing..."
        self._timer.stop()
        return "Paused."

    def _save(self) -> str:
        write_annotation_atomic(
            path=self.state.data.path,
            edge=self.state.data.edge,
            binary_values=self.state.annotation,
            source_threshold=self.state.data.source_threshold,
            edits_count=self.state.edits_count,
            allow_replace=self._replace_existing or self._expected_existing_present,
            expected_existing_present=self._expected_existing_present,
            expected_existing_checksum=self._expected_existing_checksum,
        )
        # Update session-local state so subsequent saves use the fresh checksum.
        self._expected_existing_present = True
        self._expected_existing_checksum = _binary_checksum(self.state.annotation)
        self._replace_existing = True
        self.state.mark_saved()
        return f"Saved -> annotations/{self.state.data.edge}/binary"

    def _on_key(self, event) -> None:  # type: ignore[override]
        if event.key is None:
            return
        key = event.key.lower()
        message: Optional[str] = None
        annotation_changed = False

        try:
            if key == "left":
                self.state.set_frame(self.state.current_frame - 1)
            elif key == "right":
                self.state.set_frame(self.state.current_frame + 1)
            elif key == "up":
                self.state.set_frame(self.state.current_frame + 10)
            elif key == "down":
                self.state.set_frame(self.state.current_frame - 10)
            elif key == "home":
                self.state.set_frame(0)
            elif key == "end":
                self.state.set_frame(self.state.data.num_frames - 1)
            elif key == "i":
                message = self.state.set_in_point()
            elif key == "o":
                message = self.state.fill_to_current()
                annotation_changed = True
            elif key == "d":
                message = self.state.delete_current_interval()
                annotation_changed = True
            elif key == "c":
                message = self.state.clear_all()
                annotation_changed = True
            elif key == "r":
                message = self.state.reset_to_seed()
                annotation_changed = True
            elif key == "n":
                message = self.state.jump_disagreement(direction=1)
            elif key == "p":
                message = self.state.jump_disagreement(direction=-1)
            elif key in (" ", "space"):
                message = self._toggle_play()
            elif key == "s":
                message = self._save()
            elif key == "q":
                if self.state.dirty and not self.state.quit_armed:
                    self.state.quit_armed = True
                    message = "Unsaved changes -- press Q again to quit without saving."
                else:
                    self._timer.stop()
                    plt.close(self.view.figure)
                    return
            else:
                return
        except Exception as exc:
            message = f"ERROR: {exc}"

        # For edit operations (O/D/R), do a full refresh that includes the
        # annotation trace. All other keys use the lightweight path (same as
        # the timer and slider) to keep frame navigation snappy.
        if annotation_changed:
            self.view.refresh(message)
        else:
            self.view.refresh_frame()
            self.view._refresh_status(message)
            self.view.figure.canvas.draw_idle()


# ---------------------------------------------------------------------------
# CLI entry point
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
    default=None,
    help=(
        "Edge name to annotate (from predictions/<edge>). "
        "Omit to list available edges and exit."
    ),
)
@click.option(
    "--left-camera",
    default=_LEFT_CAM_DEFAULT,
    show_default=True,
    help="HDF5 dataset name for the left wrist camera.",
)
@click.option(
    "--right-camera",
    default=_RIGHT_CAM_DEFAULT,
    show_default=True,
    help="HDF5 dataset name for the right wrist camera.",
)
@click.option(
    "--resume-existing/--seed-from-prediction",
    default=False,
    show_default=True,
    help=(
        "If annotations/<edge> already exists: --resume-existing loads it as the "
        "starting point; --seed-from-prediction (default) seeds from the model "
        "prediction instead (requires --overwrite to replace existing)."
    ),
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Allow replacing an existing annotations/<edge> group from scratch.",
)
def main(
    hdf5_path: Path,
    edge: Optional[str],
    left_camera: str,
    right_camera: str,
    resume_existing: bool,
    overwrite: bool,
) -> None:
    """Annotate ground-truth contact masks on rollout HDF5 recordings."""
    if edge is None:
        try:
            edges = list_prediction_edges(hdf5_path)
        except OSError as exc:
            raise click.ClickException(str(exc)) from exc
        if not edges:
            raise click.ClickException(
                f"No prediction edges (predictions/<edge>/binary) found in {hdf5_path}."
            )
        click.echo("Available prediction edges:")
        for name in edges:
            click.echo(f"  {name}")
        return

    try:
        data = load_rollout_data(
            path=hdf5_path,
            edge=edge,
            left_camera=left_camera,
            right_camera=right_camera,
            resume_existing=resume_existing,
            overwrite=overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_CONTROLS_HELP)
    annotator = RolloutAnnotator(
        data=data,
        replace_existing=overwrite or resume_existing,
    )
    annotator.show()


if __name__ == "__main__":
    main()
