if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import copy
import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from contact_pred.scripts.common.normalize_util import get_image_range_normalizer
from contact_pred.scripts.common.pytorch_util import dict_apply
from contact_pred.scripts.common.sampler import downsample_mask, get_val_mask
from contact_pred.scripts.common.text_embedding_util import load_or_create_text_embeddings
from contact_pred.scripts.dataset.contact_base_dataset import ContactBaseDataset
from contact_pred.scripts.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _get_project_root() -> str:
    """Locate the repo root via the committed ``.repo_root`` marker.

    Depth-independent walk (not a fixed ``..`` count), so the file can move or
    the repo be renamed/vendored without breaking path resolution.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for k in range(16):
        candidate = os.path.abspath(os.path.join(here, *([os.pardir] * k)))
        if os.path.isfile(os.path.join(candidate, ".repo_root")):
            return candidate
    # Fallback to the historical fixed depth (contact_pred/scripts/dataset -> root).
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _resolve_repo_path(path: str) -> str:
    """Anchor a relative dataset path to the repo root; leave absolute paths as-is."""
    return path if os.path.isabs(path) else os.path.join(_get_project_root(), path)


def _resolve_source_camera_name(
    edge_name: str, available_camera_names: List[str], robot_camera_map: dict
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


class ContactImageDataset(ContactBaseDataset):
    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        shape_meta: Optional[dict] = None,
        num_episodes=50,
        camera_names=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        n_obs_steps=None,
        max_train_episodes=None,
        contact_edge="robot0_moka_pot_1",
        except_idx=None,
        sources=None,
        embedding_cache_dir: Optional[str] = None,
        text_encoder_model_name: str = "bert-base-cased",
        text_encoder_max_length: int = 25,
        single_camera_per_edge: bool = False,
        selected_camera_key: Optional[str] = None,
        robot_camera_map: Optional[dict] = None,
    ):
        super().__init__()
        if shape_meta is None:
            raise ValueError("shape_meta is required")

        if camera_names is None:
            camera_names = ["cam_robot0"]
        if except_idx is None:
            except_idx = []

        self.shape_meta = shape_meta
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.n_obs_steps = n_obs_steps
        self.max_train_episodes = max_train_episodes
        self.embedding_cache_dir = embedding_cache_dir or os.path.join(_get_project_root(), "data", "bert")
        self.text_encoder_model_name = text_encoder_model_name
        self.text_encoder_max_length = text_encoder_max_length
        self.single_camera_per_edge = single_camera_per_edge
        self.selected_camera_key = selected_camera_key
        self.robot_camera_map = dict(robot_camera_map or {})

        self.rgb_keys = []
        self.lowdim_keys = []
        self.proprio_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
            elif obs_type == "proprio":
                self.proprio_keys.append(key)
        self.rgb_keys = sorted(self.rgb_keys)
        self.lowdim_keys = sorted(self.lowdim_keys)
        self.proprio_keys = sorted(self.proprio_keys)
        self.has_edge_text = "edge_text" in self.lowdim_keys

        self.sources = self._normalize_sources(
            sources=sources,
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            camera_names=camera_names,
            contact_edge=contact_edge,
            except_idx=except_idx,
        )

        self.episodes = []
        self.edge_names = []
        self.edge_name_to_id = {}
        self.edge_id_to_name = {}
        self.split_episode_indices_by_source = {}
        self._source_episode_global_indices = {}

        self._load_episodes()
        self._init_text_embeddings()
        self._build_split_indices()
        self._build_edge_loss_weights()
        self._set_active_split("train")

    def _resolve_source_camera_name(self, edge_name: str, available_camera_names: List[str]) -> str:
        return _resolve_source_camera_name(edge_name, available_camera_names, self.robot_camera_map)

    def _get_episode_image_sequence(self, episode: dict, obs_key: str, edge_name: str) -> np.ndarray:
        if obs_key in episode["images"]:
            return episode["images"][obs_key]
        if not self.single_camera_per_edge:
            raise KeyError(f"Image key '{obs_key}' not found in episode images.")
        if self.selected_camera_key is not None and obs_key != self.selected_camera_key:
            raise KeyError(
                f"Image key '{obs_key}' not found in episode images and does not match "
                f"selected_camera_key '{self.selected_camera_key}'."
            )

        source_camera = self._resolve_source_camera_name(edge_name, episode["camera_names"])
        return episode["images"][source_camera]

    def _normalize_sources(
        self,
        sources,
        dataset_dir,
        num_episodes,
        camera_names,
        contact_edge,
        except_idx,
    ) -> List[dict]:
        if sources is not None:
            normalized = []
            for source_idx, source in enumerate(sources):
                normalized.append(
                    {
                        "task_name": source.get("task_name", f"source_{source_idx}"),
                        "dataset_dir": _resolve_repo_path(source["dataset_dir"]),
                        "num_episodes": source.get("num_episodes", num_episodes),
                        "camera_names": source.get("camera_names", camera_names),
                        "contact_edges": list(source["contact_edges"]),
                        "negative_edges": list(source.get("negative_edges", [])),
                        "except_idx": list(source.get("except_idx", [])),
                    }
                )
            return normalized

        if dataset_dir is None:
            raise ValueError("dataset_dir is required when sources is not provided")

        return [
            {
                "task_name": "default",
                "dataset_dir": _resolve_repo_path(dataset_dir),
                "num_episodes": num_episodes,
                "camera_names": list(camera_names),
                "contact_edges": [contact_edge],
                "negative_edges": [],
                "except_idx": list(except_idx),
            }
        ]

    def _load_episodes(self):
        edge_names = []
        for source in self.sources:
            source_name = source["task_name"]
            global_indices = []
            for episode_id in tqdm(range(source["num_episodes"]), desc=f"load:{source_name}", leave=False):
                if episode_id in source["except_idx"]:
                    continue

                dataset_path = os.path.join(source["dataset_dir"], f"episode_{episode_id}_contact.hdf5")
                with h5py.File(dataset_path, "r") as root:
                    images = {}
                    for camera_name in source["camera_names"]:
                        images[camera_name] = root[f"/observations/images/{camera_name}"][()]

                    labels = {}
                    valid_masks = {}
                    for edge_name in source["contact_edges"]:
                        labels[edge_name] = root[f"/label/{edge_name}"][()].astype(np.float32)
                        valid_path = f"/valid/{edge_name}"
                        if valid_path in root:
                            valid_masks[edge_name] = root[valid_path][()].astype(bool)
                        else:
                            valid_masks[edge_name] = np.ones_like(labels[edge_name], dtype=bool)

                    # Synthesize all-zero labels for same-arm cross-edge negatives.
                    # These frames show the arm doing a different object, so contact = 0.
                    for neg_edge in source["negative_edges"]:
                        if neg_edge in labels:
                            raise ValueError(
                                f"negative_edge '{neg_edge}' overlaps with contact_edges "
                                f"for source '{source_name}'"
                            )
                        T = next(iter(labels.values())).shape[0]
                        labels[neg_edge] = np.zeros(T, dtype=np.float32)
                        valid_masks[neg_edge] = np.ones(T, dtype=bool)

                episode_length = next(iter(labels.values())).shape[0]
                for camera_name, image in images.items():
                    if image.shape[0] != episode_length:
                        raise ValueError(f"Camera {camera_name} length mismatch for {dataset_path}")
                for edge_name, label in labels.items():
                    if label.shape[0] != episode_length:
                        raise ValueError(f"Label {edge_name} length mismatch for {dataset_path}")
                    if valid_masks[edge_name].shape[0] != episode_length:
                        raise ValueError(f"Valid mask {edge_name} length mismatch for {dataset_path}")

                episode = {
                    "source_name": source_name,
                    "episode_id": episode_id,
                    "camera_names": list(source["camera_names"]),
                    "images": images,
                    "labels": labels,
                    "valid_masks": valid_masks,
                    "edge_names": list(source["contact_edges"]) + list(source["negative_edges"]),
                    "length": episode_length,
                }
                global_indices.append(len(self.episodes))
                self.episodes.append(episode)
                edge_names.extend(source["contact_edges"])
                edge_names.extend(source["negative_edges"])

            self._source_episode_global_indices[source_name] = global_indices

        self.edge_names = list(dict.fromkeys(edge_names))
        self.edge_name_to_id = {
            edge_name: edge_id for edge_id, edge_name in enumerate(self.edge_names)
        }
        self.edge_id_to_name = {
            edge_id: edge_name for edge_name, edge_id in self.edge_name_to_id.items()
        }

    def _init_text_embeddings(self):
        self.edge_text_embeddings = {}
        if not self.has_edge_text:
            return

        self.edge_text_embeddings = load_or_create_text_embeddings(
            self.edge_names,
            self.embedding_cache_dir,
            self.text_encoder_model_name,
            self.text_encoder_max_length,
        )

    def _build_split_indices(self):
        self._train_episode_global_indices = []
        self._val_episode_global_indices = []
        for source_name, global_indices in self._source_episode_global_indices.items():
            episode_ids = [self.episodes[idx]["episode_id"] for idx in global_indices]
            val_mask = get_val_mask(
                n_episodes=len(global_indices),
                val_ratio=self.val_ratio,
                seed=self.seed,
            )
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=self.max_train_episodes,
                seed=self.seed,
            )

            train_local_ids = [episode_ids[idx] for idx, keep in enumerate(train_mask) if keep]
            val_local_ids = [episode_ids[idx] for idx, keep in enumerate(val_mask) if keep]
            self.split_episode_indices_by_source[source_name] = {
                "train": train_local_ids,
                "val": val_local_ids,
            }

            self._train_episode_global_indices.extend(
                [global_indices[idx] for idx, keep in enumerate(train_mask) if keep]
            )
            self._val_episode_global_indices.extend(
                [global_indices[idx] for idx, keep in enumerate(val_mask) if keep]
            )

        self._train_sample_index = self._build_sample_index(self._train_episode_global_indices)
        self._val_sample_index = self._build_sample_index(self._val_episode_global_indices)

    def _build_sample_index(self, episode_global_indices: List[int]) -> List[dict]:
        sample_index = []
        for episode_idx in episode_global_indices:
            episode = self.episodes[episode_idx]
            for edge_name in episode["edge_names"]:
                edge_id = self.edge_name_to_id[edge_name]
                sample_index.extend(
                    self._build_sequence_index_for_episode(
                        episode_idx=episode_idx,
                        edge_name=edge_name,
                        edge_id=edge_id,
                        episode_length=episode["length"],
                        valid_mask=episode["valid_masks"][edge_name],
                        label_array=episode["labels"][edge_name],
                    )
                )
        return sample_index

    def _build_sequence_index_for_episode(
        self,
        episode_idx: int,
        edge_name: str,
        edge_id: int,
        episode_length: int,
        valid_mask: np.ndarray,
        label_array: np.ndarray,
    ) -> List[dict]:
        indices = []
        min_start = -self.pad_before
        max_start = episode_length - self.horizon + self.pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0)
            if not valid_mask[buffer_start_idx]:
                continue
            buffer_end_idx = min(idx + self.horizon, episode_length)
            start_offset = buffer_start_idx - idx
            end_offset = (idx + self.horizon) - buffer_end_idx
            sample_start_idx = start_offset
            sample_end_idx = self.horizon - end_offset
            indices.append(
                {
                    "episode_idx": episode_idx,
                    "edge_name": edge_name,
                    "edge_id": edge_id,
                    "buffer_start_idx": buffer_start_idx,
                    "buffer_end_idx": buffer_end_idx,
                    "sample_start_idx": sample_start_idx,
                    "sample_end_idx": sample_end_idx,
                    "label_value": float(np.clip(label_array[buffer_start_idx], 0.0, 1.0)),
                }
            )
        return indices

    def _set_active_split(self, split: str):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split {split}")
        self.active_split = split
        self.sample_index = self._train_sample_index if split == "train" else self._val_sample_index

    def _build_edge_loss_weights(self):
        self.edge_positive_weight = {}
        positive_counts = {edge_name: 0 for edge_name in self.edge_names}
        total_counts = {edge_name: 0 for edge_name in self.edge_names}

        for spec in self._train_sample_index:
            positive_counts[spec["edge_name"]] += spec["label_value"]
            total_counts[spec["edge_name"]] += 1

        for edge_name in self.edge_names:
            total = total_counts[edge_name]
            positive = positive_counts[edge_name]
            if total == 0 or positive == 0:
                self.edge_positive_weight[edge_name] = 1.0
                continue
            positive_rate = positive / total
            self.edge_positive_weight[edge_name] = float((1.0 - positive_rate) / positive_rate)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set._set_active_split("val")
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        for rgb_key in self.rgb_keys:
            normalizer[rgb_key] = get_image_range_normalizer()

        if self.has_edge_text and self.edge_text_embeddings:
            edge_text_data = np.stack(
                [self.edge_text_embeddings[edge_name] for edge_name in self.edge_names],
                axis=0,
            )
            normalizer["edge_text"] = SingleFieldLinearNormalizer.create_fit(
                data=edge_text_data,
                last_n_dims=1,
                mode=mode,
                **kwargs,
            )

        return normalizer

    def __len__(self) -> int:
        return len(self.sample_index)

    def _slice_sequence(self, array: np.ndarray, spec: dict) -> np.ndarray:
        sample = array[spec["buffer_start_idx"] : spec["buffer_end_idx"]]
        if spec["sample_start_idx"] == 0 and spec["sample_end_idx"] == self.horizon:
            return sample

        data = np.zeros((self.horizon,) + array.shape[1:], dtype=array.dtype)
        if spec["sample_start_idx"] > 0:
            data[: spec["sample_start_idx"]] = sample[0]
        if spec["sample_end_idx"] < self.horizon:
            data[spec["sample_end_idx"] :] = sample[-1]
        data[spec["sample_start_idx"] : spec["sample_end_idx"]] = sample
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        spec = self.sample_index[idx]
        episode = self.episodes[spec["episode_idx"]]
        time_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            image_sequence = self._get_episode_image_sequence(episode, key, spec["edge_name"])
            image = self._slice_sequence(image_sequence, spec)[time_slice]
            image = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0
            if image.shape[0] == 1:
                image = image[0]
            obs_dict[key] = image

        if self.has_edge_text:
            edge_text = self.edge_text_embeddings[spec["edge_name"]].astype(np.float32)
            n_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
            if n_steps == 1:
                obs_dict["edge_text"] = edge_text
            else:
                obs_dict["edge_text"] = np.repeat(edge_text[None, :], n_steps, axis=0)

        label = self._slice_sequence(episode["labels"][spec["edge_name"]], spec)[time_slice]
        if label.shape[0] == 1:
            label = label.reshape(1)
        label_value = float(np.clip(label[0], 0.0, 1.0))
        positive_weight = self.edge_positive_weight[spec["edge_name"]]
        loss_weight = label_value * positive_weight + (1.0 - label_value)

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "label": torch.from_numpy(label.astype(np.float32)),
            "edge_id": torch.tensor(spec["edge_id"], dtype=torch.long),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
        }


if __name__ == "__main__":
    pass
