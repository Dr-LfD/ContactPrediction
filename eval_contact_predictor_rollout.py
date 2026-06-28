"""Evaluate a contact-edge predictor on deployed-policy rollout HDF5 files.

The script loads a checkpoint, runs the contact predictor frame-by-frame on a
rollout recording, and writes the per-frame results back into the same HDF5 under
``predictions/<edge_name>``.

Expected rollout HDF5 layout (produced by the deployed policy):
    root attrs : {fps, max_timesteps, num_frames}
    root datasets named by the checkpoint task config camera metadata, e.g.
        cam_left_wrist   (T, H, W, 3) uint8
        cam_right_wrist  (T, H, W, 3) uint8

Writeback layout::

    predictions/<edge_name>/
        label   (T,) float32   raw value returned by predictor
        binary  (T,) uint8     (label > threshold)
        attrs:  checkpoint, checkpoint_stem, threshold, source_camera,
                root_dataset, edge_text, horizon, output_kind, timestamp_utc
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Iterable

import click
import h5py
import numpy as np

from contact_pred.scripts.common.contact_predictor_runtime import (
    ContactPredictorRuntimeSession,
    RolloutHDF5FrameProvider,
    build_edge_text_obs,
    build_model_obs_from_hwc_image,
    resolve_source_camera_name,
)
from contact_pred.scripts.common.text_embedding_util import (
    load_or_create_text_embeddings as _load_or_create_text_embeddings,
)

# Test seam: tests/test_single_wrist_edge_selection.py monkeypatches this name to inject
# fake embeddings without loading a real BERT model.
load_or_create_text_embeddings = _load_or_create_text_embeddings


def load_runtime_from_checkpoint(
    checkpoint: str,
    device: str | None = None,
    runtime_session: ContactPredictorRuntimeSession | None = None,
) -> dict[str, Any]:
    runtime_session = runtime_session or ContactPredictorRuntimeSession(device=device)
    runtime = runtime_session.load_runtime(checkpoint)
    shape_obs_meta = (runtime.get("shape_meta") or {}).get("obs") or {}
    model_rgb_keys = [
        key for key, attr in shape_obs_meta.items() if attr.get("type") == "rgb"
    ]

    single_camera_per_edge = bool(runtime.get("single_camera_per_edge", False))
    selected_camera_key = runtime.get("selected_camera_key")
    if not single_camera_per_edge:
        raise ValueError(
            "Rollout evaluation requires the checkpoint dataset config to set "
            "single_camera_per_edge=True."
        )
    if selected_camera_key is None or selected_camera_key not in model_rgb_keys:
        raise ValueError(
            f"selected_camera_key '{selected_camera_key}' must be one of model_rgb_keys "
            f"{model_rgb_keys}."
        )
    if model_rgb_keys != [selected_camera_key]:
        raise ValueError(
            "Rollout evaluation only supports single-camera models; got "
            f"model_rgb_keys={model_rgb_keys}, selected_camera_key='{selected_camera_key}'."
        )

    runtime["model_rgb_keys"] = model_rgb_keys
    return runtime


def expected_rollout_camera_names(runtime: dict[str, Any]) -> list[str]:
    """Return rollout camera dataset names from required robot_camera_map config."""
    if "robot_camera_map" not in runtime:
        raise ValueError(
            "Checkpoint dataset config must define robot_camera_map for rollout "
            "camera discovery."
        )
    robot_camera_map = runtime["robot_camera_map"]
    if not isinstance(robot_camera_map, dict) or not robot_camera_map:
        raise ValueError(
            "Checkpoint dataset config robot_camera_map must be a non-empty mapping."
        )

    camera_names = list(dict.fromkeys(robot_camera_map.values()))
    if any(not isinstance(name, str) or not name for name in camera_names):
        raise ValueError(
            f"robot_camera_map values must be non-empty camera dataset names; got "
            f"{robot_camera_map}."
        )
    return camera_names


def discover_rollout_camera_datasets(
    root: h5py.File, runtime: dict[str, Any]
) -> dict[str, str]:
    """Map configured rollout camera name -> root dataset name."""
    expected_camera_names = expected_rollout_camera_names(runtime)
    mapping: dict[str, str] = {}
    missing_camera_names: list[str] = []
    for name in expected_camera_names:
        if name not in root:
            missing_camera_names.append(name)
            continue
        value = root[name]
        if not isinstance(value, h5py.Dataset):
            raise ValueError(
                f"Configured rollout camera '{name}' exists at the HDF5 root but is "
                f"not a dataset. Available root keys: {sorted(root.keys())}."
            )
        mapping[name] = name

    if missing_camera_names:
        raise ValueError(
            f"Missing rollout camera dataset(s) {missing_camera_names} at the HDF5 "
            f"root. Available root keys: {sorted(root.keys())}."
        )
    return mapping


def parse_edge_names(values: Iterable[str]) -> list[str]:
    """Normalise repeatable / comma-separated --edge-text values."""
    edge_names: list[str] = []
    for raw in values:
        for token in raw.split(","):
            edge_name = token.strip()
            if not edge_name:
                continue
            if edge_name not in edge_names:
                edge_names.append(edge_name)
    if not edge_names:
        raise ValueError("At least one non-empty edge name is required.")
    return edge_names


def validate_edges_against_runtime(
    edge_names: list[str],
    runtime: dict[str, Any],
    dataset_camera_to_root_name: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Resolve every requested edge to its source camera and root dataset name."""
    available_cameras = sorted(dataset_camera_to_root_name.keys())
    if "robot_camera_map" not in runtime:
        raise ValueError(
            "Checkpoint dataset config must define robot_camera_map for edge camera "
            "resolution."
        )
    robot_camera_map = runtime["robot_camera_map"]
    if not isinstance(robot_camera_map, dict) or not robot_camera_map:
        raise ValueError(
            "Checkpoint dataset config robot_camera_map must be a non-empty mapping."
        )
    resolution: dict[str, dict[str, str]] = {}
    for edge_name in edge_names:
        try:
            source_camera = resolve_source_camera_name(
                edge_name, available_cameras, robot_camera_map
            )
        except KeyError as exc:
            raise ValueError(
                f"Edge '{edge_name}' could not be mapped to a rollout camera: {exc}"
            ) from exc
        resolution[edge_name] = {
            "source_camera": source_camera,
            "root_dataset": dataset_camera_to_root_name[source_camera],
        }
    return resolution


def precompute_edge_text_embeddings(
    runtime: dict[str, object],
    edge_names: list[str],
    runtime_session: ContactPredictorRuntimeSession | None = None,
) -> dict[str, np.ndarray]:
    if load_or_create_text_embeddings is not _load_or_create_text_embeddings:
        text_cfg = runtime.get("text_cfg") or {}
        embedding_map = load_or_create_text_embeddings(
            edge_names,
            text_cfg.get("embedding_cache_dir"),
            text_cfg.get("text_encoder_model_name", "bert-base-cased"),
            text_cfg.get("text_encoder_max_length", 25),
        )
        return {
            edge_name: np.asarray(embedding_map[edge_name], dtype=np.float32)
            for edge_name in edge_names
        }
    runtime_session = runtime_session or ContactPredictorRuntimeSession(
        device=runtime.get("device")
    )
    return {
        edge_name: runtime_session.get_edge_text_embedding(runtime, edge_name)
        for edge_name in edge_names
    }


def resolve_rollout_horizon(
    root: h5py.File, root_dataset_name: str, max_frames: int | None = None
) -> int:
    if "num_frames" not in root.attrs:
        raise ValueError("Rollout HDF5 is missing required root attribute 'num_frames'.")
    declared = int(root.attrs["num_frames"])
    actual = int(root[root_dataset_name].shape[0])
    if declared != actual:
        raise ValueError(
            f"num_frames mismatch for dataset '{root_dataset_name}': "
            f"attr={declared}, dataset={actual}"
        )
    horizon = declared
    if max_frames is not None:
        horizon = min(horizon, int(max_frames))
    return horizon


def build_rollout_frame_obs(
    root: Any,
    runtime: dict[str, object],
    root_dataset_name: str,
    edge_embedding: np.ndarray,
    edge_name: str,
    frame_index: int,
    runtime_session: ContactPredictorRuntimeSession | None = None,
) -> dict[str, np.ndarray]:
    runtime_session = runtime_session or ContactPredictorRuntimeSession(
        device=runtime.get("device")
    )
    provider = RolloutHDF5FrameProvider(root, frame_index)
    det = {
        "contact_edge": edge_name,
        "obs_key_map": {
            runtime["selected_camera_key"]: root_dataset_name,
        },
    }
    obs_dict: dict[str, np.ndarray] = {}
    for obs_name, attr in ((runtime.get("shape_meta") or {}).get("obs") or {}).items():
        obs_type = attr.get("type")
        if obs_type == "rgb":
            obs_dict[obs_name] = runtime_session.build_image_obs(
                runtime,
                det,
                obs_name,
                provider,
            )
        elif obs_name == "edge_text":
            obs_dict["edge_text"] = build_edge_text_obs(edge_embedding, attr)
        else:
            raise KeyError(
                f"Contact predictor checkpoint requires unsupported low-dim observation "
                f"'{obs_name}'. Add runtime construction for this key before inference."
            )
    return obs_dict


def predict_contact(
    image: np.ndarray,
    edge_text: str,
    checkpoint: str,
    device: str | None = None,
    runtime: "dict[str, Any] | None" = None,
    runtime_session: ContactPredictorRuntimeSession | None = None,
) -> float:
    """Return the contact probability for a single image + edge text pair.

    Args:
        image: uint8 array, shape (H, W, 3) or (3, H, W).
        edge_text: Edge name, e.g. ``"robot1_needle_obj"``.
        checkpoint: Path to the ``.ckpt`` file.
        device: Torch device string. Defaults to ``cuda:0`` when available.
        runtime: Pre-loaded runtime dict from :func:`load_runtime_from_checkpoint`.
                 Pass this to avoid reloading the checkpoint on every call.

    Returns:
        Sigmoid contact probability in [0, 1].
    """
    runtime_session = runtime_session or ContactPredictorRuntimeSession(device=device)
    if runtime is None:
        runtime = load_runtime_from_checkpoint(
            checkpoint, device=device, runtime_session=runtime_session
        )

    # Normalise image to (H, W, 3) uint8 before conversion.
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    obs_dict: dict[str, np.ndarray] = {
        runtime["selected_camera_key"]: build_model_obs_from_hwc_image(img)
    }
    if "edge_text" in (runtime.get("shape_meta", {}).get("obs") or {}):
        embedding = runtime_session.get_edge_text_embedding(runtime, edge_text)
        obs_dict["edge_text"] = np.expand_dims(embedding.astype(np.float32), axis=0)

    return runtime_session.predict_label(runtime, obs_dict)


def evaluate_edge_on_episode(
    root: h5py.File,
    runtime: dict[str, object],
    edge_name: str,
    edge_embedding: np.ndarray,
    source_camera: str,
    root_dataset_name: str,
    max_frames: int | None = None,
    runtime_session: ContactPredictorRuntimeSession | None = None,
) -> dict[str, Any]:
    horizon = resolve_rollout_horizon(root, root_dataset_name, max_frames=max_frames)
    # Bulk-load all frames into memory once; per-frame HDF5 reads are expensive at T=300+.
    all_frames = root[root_dataset_name][:horizon]
    if "process_name" in root:
        raw_pnames = root["process_name"][:horizon]
        valid_mask = np.array([v != b"lfd" for v in raw_pnames], dtype=bool)
    else:
        valid_mask = np.ones(horizon, dtype=bool)

    label_values = np.full(horizon, np.nan, dtype=np.float32)
    runtime_session = runtime_session or ContactPredictorRuntimeSession(
        device=runtime.get("device")
    )
    for frame_index in range(horizon):
        if not valid_mask[frame_index]:
            continue
        obs_dict = build_rollout_frame_obs(
            root={root_dataset_name: all_frames},
            runtime=runtime,
            root_dataset_name=root_dataset_name,
            edge_embedding=edge_embedding,
            edge_name=edge_name,
            frame_index=frame_index,
            runtime_session=runtime_session,
        )
        label_values[frame_index] = runtime_session.predict_label(runtime, obs_dict)

    if not np.isfinite(label_values[valid_mask]).all():
        raise ValueError(
            f"Predictor produced non-finite values for edge '{edge_name}'."
        )

    return {
        "edge_name": edge_name,
        "source_camera": source_camera,
        "root_dataset": root_dataset_name,
        "horizon": horizon,
        "input_source": "raw",
        "label": label_values,
        "valid_mask": valid_mask,
    }


def write_edge_predictions(
    episode_path: str,
    edge_name: str,
    label_values: np.ndarray,
    threshold: float,
    checkpoint: str,
    checkpoint_stem: str,
    source_camera: str,
    root_dataset_name: str,
    overwrite: bool = False,
    valid_mask: "np.ndarray | None" = None,
) -> None:
    """Atomically write a single edge's predictions back into the rollout HDF5."""
    if "/" in edge_name or edge_name.startswith("_"):
        raise ValueError(
            f"Edge name '{edge_name}' is not a safe HDF5 group name."
        )
    label_values = np.asarray(label_values, dtype=np.float32)
    binary_values = (label_values > float(threshold)).astype(np.uint8)
    pending_name = f"_pending_{edge_name}"
    backup_name = f"_backup_{edge_name}"
    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    with h5py.File(episode_path, "a") as root:
        predictions = root.require_group("predictions")

        # Clean stale scratch groups from a prior aborted run.
        for scratch in (pending_name, backup_name):
            if scratch in predictions:
                del predictions[scratch]

        if edge_name in predictions and not overwrite:
            raise RuntimeError(
                f"Prediction group 'predictions/{edge_name}' already exists in "
                f"'{episode_path}'. Pass overwrite=True (--overwrite) to replace it."
            )

        # Stage new content before mutating the committed group, so an existing
        # group can be restored if any write step fails.
        pending = predictions.create_group(pending_name)
        try:
            pending.create_dataset("label", data=label_values, dtype=np.float32)
            pending.create_dataset("binary", data=binary_values, dtype=np.uint8)
            if valid_mask is not None:
                pending.create_dataset("valid_mask", data=valid_mask.astype(np.uint8), dtype=np.uint8)
            pending.attrs["checkpoint"] = os.path.abspath(checkpoint)
            pending.attrs["checkpoint_stem"] = checkpoint_stem
            pending.attrs["threshold"] = float(threshold)
            pending.attrs["source_camera"] = source_camera
            pending.attrs["root_dataset"] = root_dataset_name
            pending.attrs["edge_text"] = edge_name
            pending.attrs["horizon"] = int(label_values.shape[0])
            pending.attrs["valid_frame_count"] = int(valid_mask.sum()) if valid_mask is not None else int(label_values.shape[0])
            pending.attrs["output_kind"] = "label"
            pending.attrs["timestamp_utc"] = timestamp_utc
        except Exception:
            if pending_name in predictions:
                del predictions[pending_name]
            raise

        # Atomic-ish replace via backup-rename-delete.
        if edge_name in predictions:
            predictions.move(edge_name, backup_name)
        try:
            predictions.move(pending_name, edge_name)
        except Exception:
            if pending_name in predictions:
                del predictions[pending_name]
            if backup_name in predictions:
                predictions.move(backup_name, edge_name)
            raise
        if backup_name in predictions:
            del predictions[backup_name]


def evaluate_rollout_episode(
    episode_path: str,
    runtime: dict[str, object],
    edge_names: list[str],
    edge_embeddings: dict[str, np.ndarray] | None = None,
    threshold: float = 0.5,
    max_frames: int | None = None,
    overwrite: bool = False,
    output_dir: str | None = None,
    runtime_session: ContactPredictorRuntimeSession | None = None,
) -> dict[str, Any]:
    runtime_session = runtime_session or ContactPredictorRuntimeSession(
        device=runtime.get("device")
    )
    if edge_embeddings is None:
        edge_embeddings = precompute_edge_text_embeddings(
            runtime, edge_names, runtime_session=runtime_session
        )

    per_edge_labels: dict[str, np.ndarray] = {}
    per_edge_valid_masks: dict[str, np.ndarray] = {}
    edge_summaries: dict[str, dict[str, Any]] = {}
    episode_horizon: int | None = None

    with h5py.File(episode_path, "r") as root:
        dataset_camera_to_root_name = discover_rollout_camera_datasets(root, runtime)
        edge_resolution = validate_edges_against_runtime(
            edge_names, runtime, dataset_camera_to_root_name
        )

        for edge_name in edge_names:
            source_camera = edge_resolution[edge_name]["source_camera"]
            root_dataset_name = edge_resolution[edge_name]["root_dataset"]
            edge_result = evaluate_edge_on_episode(
                root=root,
                runtime=runtime,
                edge_name=edge_name,
                edge_embedding=edge_embeddings[edge_name],
                source_camera=source_camera,
                root_dataset_name=root_dataset_name,
                max_frames=max_frames,
                runtime_session=runtime_session,
            )
            if episode_horizon is None:
                episode_horizon = edge_result["horizon"]
            elif edge_result["horizon"] != episode_horizon:
                raise ValueError(
                    f"Inconsistent horizons within '{episode_path}': edge "
                    f"'{edge_name}' resolved to {edge_result['horizon']} frames, "
                    f"expected {episode_horizon}."
                )
            per_edge_labels[edge_name] = edge_result["label"]
            valid_mask = edge_result.get("valid_mask")
            per_edge_valid_masks[edge_name] = valid_mask
            valid = valid_mask if valid_mask is not None else np.ones(edge_result["horizon"], dtype=bool)
            valid_labels = edge_result["label"][valid]
            edge_summaries[edge_name] = {
                "source_camera": source_camera,
                "root_dataset": root_dataset_name,
                "horizon": edge_result["horizon"],
                "valid_frame_count": int(valid.sum()),
                "input_source": edge_result["input_source"],
                "mean_label": float(valid_labels.mean()) if valid_labels.size else 0.0,
                "binary_mean": float((valid_labels > threshold).mean()) if valid_labels.size else 0.0,
            }

    for edge_name, edge_summary in edge_summaries.items():
        write_edge_predictions(
            episode_path=episode_path,
            edge_name=edge_name,
            label_values=per_edge_labels[edge_name],
            threshold=threshold,
            checkpoint=runtime["checkpoint_path"],
            checkpoint_stem=runtime["checkpoint_stem"],
            source_camera=edge_summary["source_camera"],
            root_dataset_name=edge_summary["root_dataset"],
            overwrite=overwrite,
            valid_mask=per_edge_valid_masks.get(edge_name),
        )

    episode_name = pathlib.Path(episode_path).stem

    summary = {
        "episode_path": os.path.abspath(episode_path),
        "episode_name": episode_name,
        "skipped": False,
        "checkpoint": runtime["checkpoint_path"],
        "checkpoint_stem": runtime["checkpoint_stem"],
        "threshold": float(threshold),
        "overwrite": bool(overwrite),
        "edge_names": list(edge_names),
        "horizon": int(episode_horizon or 0),
        "edges": edge_summaries,
    }
    return summary


def iter_rollout_paths(
    rollout: str | None = None, rollout_dir: str | None = None
) -> list[str]:
    if bool(rollout) == bool(rollout_dir):
        raise ValueError("Exactly one of `rollout` or `rollout_dir` must be provided.")
    if rollout is not None:
        return [os.path.abspath(rollout)]
    return sorted(
        str(path.resolve()) for path in pathlib.Path(rollout_dir).glob("*.hdf5")
    )


def evaluate_rollout_paths(
    checkpoint: str,
    rollout_paths: list[str],
    edge_names: list[str],
    device: str | None = None,
    max_frames: int | None = None,
    threshold: float = 0.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    runtime_session = ContactPredictorRuntimeSession(device=device)
    runtime = load_runtime_from_checkpoint(
        checkpoint, device=device, runtime_session=runtime_session
    )
    edge_embeddings = precompute_edge_text_embeddings(
        runtime, edge_names, runtime_session=runtime_session
    )

    episode_summaries: list[dict[str, Any]] = []
    for episode_path in rollout_paths:
        summary = evaluate_rollout_episode(
            episode_path=episode_path,
            runtime=runtime,
            edge_names=edge_names,
            edge_embeddings=edge_embeddings,
            threshold=threshold,
            max_frames=max_frames,
            overwrite=overwrite,
            runtime_session=runtime_session,
        )
        episode_summaries.append(summary)

    episodes_by_edge: dict[str, list[str]] = {edge_name: [] for edge_name in edge_names}
    for summary in episode_summaries:
        for edge_name in summary["edges"]:
            episodes_by_edge[edge_name].append(summary["episode_path"])

    aggregate = {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_stem": pathlib.Path(checkpoint).stem,
        "rollout_paths": list(rollout_paths),
        "edge_names": list(edge_names),
        "threshold": float(threshold),
        "overwrite": bool(overwrite),
        "episode_count": len(episode_summaries),
        "episodes_by_edge": episodes_by_edge,
        "episodes": episode_summaries,
    }

    return aggregate


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--rollout", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--rollout-dir", default=None, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--edge-text",
    "edge_text",
    multiple=True,
    required=True,
    help="Edge name(s); repeat the flag or pass comma-separated values.",
)
@click.option("--device", default=None, help="Torch device, e.g. cuda:0 or cpu")
@click.option("--max-frames", default=None, type=int)
@click.option("--threshold", default=0.5, type=float, show_default=True)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def main(
    checkpoint: str,
    rollout: str | None,
    rollout_dir: str | None,
    edge_text: tuple[str, ...],
    device: str | None,
    max_frames: int | None,
    threshold: float,
    overwrite: bool,
) -> None:
    try:
        edge_names = parse_edge_names(edge_text)
        rollout_paths = iter_rollout_paths(rollout=rollout, rollout_dir=rollout_dir)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    summary = evaluate_rollout_paths(
        checkpoint=checkpoint,
        rollout_paths=rollout_paths,
        edge_names=edge_names,
        device=device,
        max_frames=max_frames,
        threshold=threshold,
        overwrite=overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
