from __future__ import annotations

import os
import pathlib
from typing import Any

import dill
import hydra
import numpy as np
import torch

from contact_pred.scripts.common.pytorch_util import dict_apply
from contact_pred.scripts.common.text_embedding_util import load_or_create_text_embeddings


def _default_embedding_cache_dir() -> str:
    return os.path.join(pathlib.Path(__file__).resolve().parents[3], "data", "bert")


def to_plain_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, dict):
        return value
    return {}


def resolve_source_camera_name(
    edge_name: str, available_camera_names: list[str], robot_camera_map: dict[str, str]
) -> str:
    matched_robot = next(
        (
            robot_name
            for robot_name in sorted(robot_camera_map, key=len, reverse=True)
            if edge_name == robot_name or edge_name.startswith(f"{robot_name}_")
        ),
        None,
    )
    if matched_robot is None:
        raise KeyError(
            f"Cannot infer source camera for edge '{edge_name}'. "
            "Configure robot_camera_map with the robot entity prefix used by the edge."
        )
    source_camera = robot_camera_map[matched_robot]
    if source_camera not in available_camera_names:
        raise KeyError(
            f"Resolved source camera '{source_camera}' for edge '{edge_name}' is not in "
            f"available cameras {available_camera_names}."
        )
    return source_camera


def build_model_obs_from_hwc_image(
    image: np.ndarray, expected_chw_shape: tuple[int, int, int] | None = None
) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {array.shape}.")
    if expected_chw_shape is not None:
        if len(expected_chw_shape) != 3:
            raise ValueError(
                f"expected_chw_shape must be (C,H,W), got {expected_chw_shape}."
            )
        expected_c, expected_h, expected_w = tuple(int(v) for v in expected_chw_shape)
        if array.shape[-1] != expected_c:
            raise ValueError(
                f"Input image has {array.shape[-1]} channels, expected {expected_c}."
            )
    if np.issubdtype(array.dtype, np.integer) or np.max(array) > 1.0:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32, copy=False)
    array = np.expand_dims(np.moveaxis(array, -1, 0), axis=0)
    if expected_chw_shape is not None and (
        array.shape[2] != expected_h or array.shape[3] != expected_w
    ):
        tensor = torch.from_numpy(array)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(expected_h, expected_w),
            mode="bilinear",
            align_corners=False,
        )
        array = tensor.numpy()
    return array


def build_edge_text_obs(embedding: np.ndarray, attr: dict[str, Any]) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32)
    expected_shape = tuple(attr.get("shape") or array.shape)
    if array.shape != expected_shape:
        array = array.reshape(expected_shape)
    return np.expand_dims(array, axis=0)


def _lookup_obs(container: Any, key: str) -> Any:
    if container is None:
        return None
    value = None
    if isinstance(container, dict):
        value = container.get(key)
        if value is None and isinstance(container.get("images"), dict):
            value = container["images"].get(key)
        return value
    try:
        value = container[key]
    except Exception:
        value = None
    if value is not None:
        return value
    images = getattr(container, "get", lambda _key, default=None: default)("images")
    if isinstance(images, dict):
        return images.get(key)
    return None


class LFDObservationProvider:
    def __init__(self, lfd):
        self.lfd = lfd

    def get_image(self, obs_key: str) -> np.ndarray:
        ts = getattr(self.lfd, "ts", None)
        if ts is not None:
            value = _lookup_obs(getattr(ts, "observation", None), obs_key)
            if value is not None:
                return self._processed_to_hwc(value, obs_key)

        obs = getattr(self.lfd, "obs", None)
        value = _lookup_obs(obs, obs_key)
        if value is not None:
            return self._processed_to_hwc(value, obs_key)

        raw_obs = getattr(self.lfd, "raw_obs", None)
        value = _lookup_obs(raw_obs, obs_key)
        if value is not None:
            return self._raw_to_hwc(value, obs_key)

        raise KeyError(
            f"Observation key '{obs_key}' is missing from lfd.ts.observation, lfd.obs, and lfd.raw_obs."
        )

    def _processed_to_hwc(self, image: Any, obs_key: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(
                f"Processed observation '{obs_key}' must be 3D, got shape {array.shape}."
            )
        # robomimic-processed RGB observations are channels-first float images.
        if array.shape[0] in (1, 3):
            return np.moveaxis(array, 0, -1)
        if array.shape[-1] in (1, 3):
            return array
        raise ValueError(
            f"Processed observation '{obs_key}' must be CHW or HWC, got shape {array.shape}."
        )

    def _raw_to_hwc(self, image: Any, obs_key: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(
                f"Raw observation '{obs_key}' must be an HxWxC image, got shape {array.shape}."
            )
        # robosuite raw camera output follows OpenGL convention; flip vertically to
        # match the processed observation path used by deployment and HDF5 rollouts.
        return array[::-1]


class RolloutHDF5FrameProvider:
    def __init__(self, root, frame_index: int):
        self.root = root
        self.frame_index = int(frame_index)

    def get_image(self, obs_key: str) -> np.ndarray:
        image = np.asarray(self.root[obs_key][self.frame_index])
        if image.ndim != 3:
            raise ValueError(
                f"Rollout frame dataset '{obs_key}' must yield HxWxC images, got shape {image.shape}."
            )
        return image


class ContactPredictorRuntimeSession:
    def __init__(
        self,
        *,
        device: str | None = None,
        output_dir: str | None = None,
        embedding_cache_dir: str | None = None,
    ):
        self.device = device
        self.output_dir = output_dir
        self.embedding_cache_dir = embedding_cache_dir or _default_embedding_cache_dir()
        self._runtime_cache: dict[str, dict[str, Any]] = {}
        self._edge_text_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def load_runtime(self, checkpoint: str) -> dict[str, Any]:
        checkpoint_path = os.path.abspath(checkpoint)
        runtime = self._runtime_cache.get(checkpoint_path)
        if runtime is not None:
            return runtime

        with open(checkpoint_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace = self._instantiate_workspace(workspace_cls, cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        predictor = workspace.model
        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        predictor = predictor.to(torch.device(device))
        predictor.eval()

        shape_meta = to_plain_config(getattr(cfg, "shape_meta", {}))
        task_cfg = getattr(cfg, "task", None)
        dataset_cfg = getattr(task_cfg, "dataset", None)
        camera_names = list(getattr(dataset_cfg, "camera_names", []))
        if not camera_names:
            camera_names = [
                key
                for key, attr in (shape_meta.get("obs") or {}).items()
                if attr.get("type") == "rgb"
            ]

        runtime = {
            "predictor": predictor,
            "device": device,
            "torch": torch,
            "dict_apply": dict_apply,
            "shape_meta": shape_meta,
            "camera_names": camera_names,
            "contact_edge": getattr(dataset_cfg, "contact_edge", None),
            "single_camera_per_edge": bool(
                getattr(dataset_cfg, "single_camera_per_edge", False)
            ),
            "selected_camera_key": getattr(dataset_cfg, "selected_camera_key", None),
            "robot_camera_map": to_plain_config(
                getattr(dataset_cfg, "robot_camera_map", {})
            ),
            "contact_label_threshold": float(
                getattr(dataset_cfg, "contact_label_threshold", 0.5)
            )
            if dataset_cfg is not None
            else 0.5,
            "checkpoint_path": checkpoint_path,
            "checkpoint_stem": pathlib.Path(checkpoint_path).stem,
            "text_cfg": {
                "embedding_cache_dir": getattr(
                    dataset_cfg,
                    "embedding_cache_dir",
                    self.embedding_cache_dir,
                ),
                "text_encoder_model_name": getattr(
                    dataset_cfg, "text_encoder_model_name", "bert-base-cased"
                ),
                "text_encoder_max_length": getattr(
                    dataset_cfg, "text_encoder_max_length", 25
                ),
            },
        }
        rcmap = dict(runtime.get("robot_camera_map") or {})
        if rcmap:
            cn = list(runtime.get("camera_names") or [])
            runtime["camera_names"] = list(dict.fromkeys(cn + list(rcmap.values())))
        self._runtime_cache[checkpoint_path] = runtime
        return runtime

    def _instantiate_workspace(self, workspace_cls, cfg):
        if self.output_dir is not None:
            try:
                return workspace_cls(cfg, output_dir=self.output_dir)
            except TypeError:
                pass
        return workspace_cls(cfg)

    def get_edge_text_embedding(
        self, runtime: dict[str, Any], contact_edge: str
    ) -> np.ndarray:
        text_cfg = runtime.get("text_cfg") or {}
        cache_key = (
            text_cfg.get("embedding_cache_dir"),
            text_cfg.get("text_encoder_model_name"),
            text_cfg.get("text_encoder_max_length"),
            contact_edge,
        )
        if cache_key not in self._edge_text_cache:
            embeddings = load_or_create_text_embeddings(
                [contact_edge],
                text_cfg.get("embedding_cache_dir", self.embedding_cache_dir),
                text_cfg.get("text_encoder_model_name", "bert-base-cased"),
                text_cfg.get("text_encoder_max_length", 25),
            )
            self._edge_text_cache[cache_key] = np.asarray(
                embeddings[contact_edge], dtype=np.float32
            )
        return self._edge_text_cache[cache_key]

    def build_image_obs(
        self,
        runtime: dict[str, Any],
        det: dict[str, Any],
        camera_name: str,
        provider,
        default_obs_key_map: dict[str, str] | None = None,
    ) -> np.ndarray:
        """Build CHW model input tensor for one RGB observation key."""
        obs_key_map = dict(default_obs_key_map or {})
        obs_key_map.update(det.get("obs_key_map") or {})

        lookup_camera_name = camera_name
        if runtime.get("single_camera_per_edge") and camera_name not in obs_key_map:
            selected_camera_key = runtime.get("selected_camera_key")
            if selected_camera_key is not None and camera_name != selected_camera_key:
                raise KeyError(
                    f"Checkpoint requested image key '{camera_name}', but single_camera_per_edge "
                    f"expects '{selected_camera_key}'."
                )
            lookup_camera_name = resolve_source_camera_name(
                det["contact_edge"],
                list(runtime.get("camera_names") or []),
                dict(runtime.get("robot_camera_map") or {}),
            )

        obs_key = obs_key_map.get(lookup_camera_name, lookup_camera_name)
        image = provider.get_image(obs_key)
        obs_shape_meta = (runtime.get("shape_meta") or {}).get("obs") or {}
        obs_attr = obs_shape_meta.get(camera_name)
        if obs_attr is None:
            raise ValueError(
                f"Missing shape_meta for camera key '{camera_name}' in runtime checkpoint."
            )
        expected_chw_shape = tuple(obs_attr.get("shape") or ())

        return build_model_obs_from_hwc_image(
            image, expected_chw_shape=expected_chw_shape
        )

    def build_obs_dict(
        self,
        runtime: dict[str, Any],
        det: dict[str, Any],
        provider,
        default_obs_key_map: dict[str, str] | None = None,
    ) -> dict[str, np.ndarray]:
        shape_meta = runtime.get("shape_meta")
        obs_shape_meta = shape_meta.get("obs")

        obs_dict: dict[str, np.ndarray] = {}
        for obs_name, attr in obs_shape_meta.items():
            obs_type = attr.get("type")
            if obs_type == "rgb":
                obs_dict[obs_name] = self.build_image_obs(
                    runtime,
                    det,
                    obs_name,
                    provider,
                    default_obs_key_map=default_obs_key_map,
                )
            elif obs_name == "edge_text":
                embedding = self.get_edge_text_embedding(runtime, det["contact_edge"])
                obs_dict[obs_name] = build_edge_text_obs(embedding, attr)
            else:
                raise KeyError(
                    f"Contact predictor checkpoint requires unsupported low-dim observation "
                    f"'{obs_name}'. Add runtime construction for this key before inference."
                )
        return obs_dict

    def predict_label(
        self, runtime: dict[str, Any], obs_dict: dict[str, np.ndarray]
    ) -> float:
        torch_obs = runtime["dict_apply"](
            obs_dict,
            lambda x: runtime["torch"].from_numpy(x).to(runtime["device"]),
        )
        with runtime["torch"].no_grad():
            result = runtime["predictor"].predict_keypose_and_mode(torch_obs)
        pred_label = result["label"]
        if hasattr(pred_label, "detach"):
            pred_label = pred_label.detach().float().cpu().numpy()
        pred_label = np.asarray(pred_label).reshape(-1)
        return float(pred_label[0])
