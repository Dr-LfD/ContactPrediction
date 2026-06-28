'''
python eval_sim_kp_dataset.py \
    -k data/outputs/2024.03.02/21.10.08_train_keypose_transformer_keypose_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -d ${WS_ROOT}/dataset_sim/ \
    -t sim_transfer_cube_scripted \ 
    -i 0
'''

import click
import os
import pathlib
import os
import h5py
import numpy as np
import torch
from einops import rearrange
import hydra
import dill
import matplotlib.pyplot as plt
import time
import cv2


from contact_pred.scripts.common.pytorch_util import dict_apply
from contact_pred.scripts.common.text_embedding_util import load_or_create_text_embeddings
from contact_pred.scripts.dataset.contact_image_dataset import _resolve_source_camera_name
from contact_pred.scripts.workspace.base_workspace import BaseWorkspace


# camera_names = ["cam_robot0", "cam_agent"]
# rgb_keys = ['cam_robot0', 'cam_agent']
# camera_names = ["cam_robot0", "cam_robot1", "cam_agent"]
# rgb_keys = ["cam_robot0", "cam_robot1", 'cam_agent']
# camera_names = ["cam_robot0", "cam_robot1"]
# rgb_keys = ["cam_robot0", "cam_robot1"]
lowdim_keys = []


def _shape_meta_has_edge_text(cfg):
    obs_meta = cfg["shape_meta"]["obs"]
    return "edge_text" in obs_meta


def _resolve_eval_edge_name(cfg, edge_name):
    if edge_name is not None:
        return edge_name

    dataset_cfg = cfg["task"]["dataset"]
    if "contact_edge" in dataset_cfg:
        return dataset_cfg["contact_edge"]

    raise ValueError("edge_name is required for mixed multi-edge checkpoints.")


def _get_eval_edge_text_embedding(cfg, edge_name):
    dataset_cfg = cfg["task"]["dataset"]
    cache_dir = dataset_cfg.get(
        "embedding_cache_dir",
        os.path.join(pathlib.Path(__file__).resolve().parent, "data", "bert"),
    )
    model_name = dataset_cfg.get("text_encoder_model_name", "bert-base-cased")
    max_length = dataset_cfg.get("text_encoder_max_length", 25)
    embedding_map = load_or_create_text_embeddings(
        [edge_name],
        cache_dir,
        model_name,
        max_length,
    )
    return embedding_map[edge_name]


def _resolve_camera_names(cfg):
    names = globals().get("camera_names")
    if names:
        return names

    dataset_cfg = cfg["task"]["dataset"]
    if "camera_names" in dataset_cfg:
        return list(dataset_cfg["camera_names"])
    if "sources" in dataset_cfg and len(dataset_cfg["sources"]) > 0:
        return list(dataset_cfg["sources"][0]["camera_names"])
    raise ValueError("camera_names could not be resolved from config or CLI.")


def _resolve_model_rgb_keys(cfg):
    shape_meta = cfg["shape_meta"]["obs"]
    return [key for key, attr in shape_meta.items() if attr.get("type") == "rgb"]


def _resolve_episode_rgb_keys(data):
    configured = globals().get("rgb_keys") or []
    existing = [key for key in configured if key in data]
    if existing:
        return existing
    return [
        key
        for key, value in data.items()
        if isinstance(value, np.ndarray) and value.ndim == 4
    ]


def load_episode_to_data(dataset_path, cfg, edge_name=None):
    edge_name = _resolve_eval_edge_name(cfg, edge_name)
    dataset_cfg = cfg["task"]["dataset"]
    available_camera_names = _resolve_camera_names(cfg)
    single_camera_per_edge = bool(dataset_cfg.get("single_camera_per_edge", False))
    selected_camera_key = dataset_cfg.get("selected_camera_key")
    robot_camera_map = dict(dataset_cfg.get("robot_camera_map", {}))
    model_rgb_keys = _resolve_model_rgb_keys(cfg)

    with h5py.File(dataset_path, "r") as root: 

        all_cam_images = dict()
        for cam_name in model_rgb_keys:
            dataset_camera_name = cam_name
            if cam_name not in root["observations/images"]:
                if not single_camera_per_edge:
                    raise KeyError(f"Camera '{cam_name}' not found in {dataset_path}")
                if selected_camera_key is not None and cam_name != selected_camera_key:
                    raise KeyError(
                        f"Camera '{cam_name}' missing from {dataset_path} and does not match "
                        f"selected_camera_key '{selected_camera_key}'."
                    )
                dataset_camera_name = _resolve_source_camera_name(
                    edge_name,
                    available_camera_names,
                    robot_camera_map,
                )
            all_cam_images[cam_name] = root[f"/observations/images/{dataset_camera_name}"][()]

        label = root[f"/label/{edge_name}"][()]

    episode = {
        "label": label,
        "edge_name": edge_name,
    }
    episode.update(all_cam_images)
    if _shape_meta_has_edge_text(cfg):
        episode["edge_text"] = _get_eval_edge_text_embedding(cfg, edge_name)

    return episode

def get_data(data, idx, rgb_keys=None):

    obs_dict = dict()
    for key in (rgb_keys if rgb_keys is not None else _resolve_episode_rgb_keys(data)):
        obs_dict[key] = (
            np.moveaxis(data[key][idx], -1, 0).astype(np.float32) / 255.0
        )
        obs_dict[key] = np.expand_dims(obs_dict[key], axis=0) #[1, C, H, W]
     
    # for key in lowdim_keys:
    #     obs_dict[key] = data[key][idx].astype(np.float32)
    #     obs_dict[key] = np.expand_dims(obs_dict[key], axis=0)  # [1, D]
    if "edge_text" in data:
        obs_dict["edge_text"] = np.expand_dims(data["edge_text"].astype(np.float32), axis=0)

    label = data["label"][idx].astype(np.float32)  # [1]
    label = np.expand_dims(label, axis=0)  # [1, 1]

    torch_data = {
        "obs": dict_apply(obs_dict, torch.from_numpy),
        "label": torch.from_numpy(label)
    }
    return torch_data


def cv2_show_img(img, ts, index, save_dir):

    img = cv2.resize(img, (128*5, 128*5), interpolation=cv2.INTER_NEAREST)

    # cv2.imshow('img', img)
    file_name = f'ts_{ts}_contact_True.png'
    cv2.imwrite(f'{save_dir}/{file_name}', img)
    print(f'save image to {file_name}')
    ## sleep for 0.05 seconds
    cv2.waitKey(50)


def visualize_contact_performance(pred_contact_list, tgt_contact_list, output_dir, epi_idx):
    """
    可视化模式预测的性能
    1. 模式预测的散点图对比
    2. 模式预测的直方图对比
    """
    pred_contact_all = np.concatenate(pred_contact_list, axis=0)
    tgt_contact_all = np.expand_dims(
        np.concatenate(tgt_contact_list, axis=0),
        axis=-1,
    )  # [T, 1]

    ## proportion of correct predictions
    correct = (pred_contact_all == tgt_contact_all).sum()
    total = tgt_contact_all.shape[0]
    accuracy = correct / total
    print(f"    Contact Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

    ## export to txt
    with open(os.path.join(output_dir, f'result_Epi{epi_idx}.txt'), 'w') as f:
        f.write(f"Episode {epi_idx} Contact Prediction Results\n\n")
        f.write(f"Contact Accuracy: {accuracy*100:.2f}% ({correct}/{total})\n\n")
        f.write(f"pred_contact_all: {pred_contact_all.flatten().tolist()}\n")
        f.write(f"tgt_contact_all: {tgt_contact_all.flatten().tolist()}\n ")
        f.close()

    # 直方图对比
    plt.figure(figsize=(10, 5))
    plt.hist(tgt_contact_all, bins=20, alpha=0.7, label='Target', color='blue', density=True)
    plt.hist(pred_contact_all, bins=20, alpha=0.7, label='Predicted', color='red', density=True)
    plt.xlabel('Contact Value')
    plt.ylabel('Density')
    plt.title('Contact Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'contact_histogram_{epi_idx}.png'), dpi=150)
    plt.close()

    # 时间序列对比
    plt.figure(figsize=(10, 5))
    plt.plot(tgt_contact_all, label='Ground-truth Contact', linewidth=2)
    plt.plot(pred_contact_all, label='Predicted Contact', linewidth=2, alpha=0.8)
    plt.xlabel('Time Step')
    plt.ylabel('Contact Boolean')
    plt.title('Contact Time Series Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'contact_timeseries_{epi_idx}.png'), dpi=150)
    plt.close()


def visualize_contact_images(episode_data, pred_contact_list, output_dir, epi_idx):
    """
    可视化接触预测的图像结果
    """
    save_dir = os.path.join(output_dir, 'contact_images')
    if not os.path.exists(save_dir):
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    pred_contact_all = np.concatenate(pred_contact_list, axis=0)  # [T, 1]
    
    ## get index where pred_contact_all == 1
    contact_indices = np.where(pred_contact_all[:, 0] == 1)[0]

    episode_rgb_keys = _resolve_episode_rgb_keys(episode_data)
    agent_images = episode_data[episode_rgb_keys[-1]]  # [T, H, W, C]
        
    for idx in contact_indices:
        img = agent_images[idx]
        cv2_show_img(img, idx, epi_idx, save_dir)


def evaluate_episode(ct_predictor, episode_path, output_dir, cfg, epi_idx, device, edge_name=None):
    """
    评估单个 episode 的数据
    """
    
    episode_data = load_episode_to_data(episode_path, cfg, edge_name=edge_name)
    horizon = episode_data["label"].shape[0]
    episode_rgb_keys = _resolve_episode_rgb_keys(episode_data)

    pred_label_list = []
    gt_label_list = []
    for idx in range(horizon):
        data = get_data(episode_data, idx, rgb_keys=episode_rgb_keys)
        data = dict_apply(data, lambda x: x.to(device))

        with torch.no_grad():
            t_start = time.time()
            result = ct_predictor.predict_keypose_and_mode(data["obs"])
            t_end = time.time()

        ## prediction
        pred_label = result["label"]

        ## target
        tgt_label = data["label"]

        ## collect results
        pred_label_list.append(pred_label.cpu().numpy())
        gt_label_list.append(tgt_label.cpu().numpy())

    # visualize_keypose_performance(pred_keypose_all, tgt_keypose_all, output_dir, epi_idx)
    visualize_contact_performance(pred_label_list, gt_label_list, output_dir, epi_idx)
    visualize_contact_images(episode_data, pred_label_list, output_dir, epi_idx)

@click.command()
@click.option('-p', '--predictor_ckpt', required=True)
@click.option('-d', '--dataset_path', required=True)
@click.option('-i', '--epi_idx', type=int, default=49, help="Episode index to evaluate")
@click.option('-dv', '--device', default='cuda:0', help="Device to run the evaluation on")
@click.option('--camera-names', 'camera_names_arg', default='cam_robot0,cam_robot1', help='Comma-separated camera names to load')
@click.option('--rgb-keys', '--rbt-keys', 'rgb_keys_arg', default='cam_robot0,cam_robot1', help='Comma-separated RGB keys for model inputs')
@click.option('--edge-name', default=None, help='Edge name to evaluate for mixed multi-edge checkpoints')
def main(predictor_ckpt, dataset_path, epi_idx, device, camera_names_arg, rgb_keys_arg, edge_name):

    # Allow overriding global camera/rgb keys via CLI without using `global`
    names_arg = camera_names_arg.strip()
    keys_arg = rgb_keys_arg.strip()
    if names_arg:
        globals()['camera_names'] = [s.strip() for s in names_arg.split(',') if s.strip()]
    if keys_arg:
        globals()['rgb_keys'] = [s.strip() for s in keys_arg.split(',') if s.strip()]

    predictor_ckpt_root = "/".join(predictor_ckpt.split('/')[:-2])
    output_dir_suffix = f"eval_epi_{epi_idx}"
    if edge_name is not None:
        output_dir_suffix = f"{output_dir_suffix}_{edge_name}"
    output_dir = os.path.join(predictor_ckpt_root, output_dir_suffix)
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    """ --- 加载数据集 --- """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    """ --- 加载模型 ---"""
    p_payload = torch.load(open(predictor_ckpt, 'rb'), pickle_module=dill)
    p_cfg = p_payload['cfg']

    cls = hydra.utils.get_class(p_cfg._target_)
    p_workspace = cls(p_cfg, output_dir=output_dir)
    p_workspace: BaseWorkspace
    ### in case that model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in p_workspace.__dict__.keys():
        p_workspace.model = hydra.utils.instantiate(p_cfg.policy)
    if "optimizer" not in p_workspace.__dict__.keys():
        p_workspace.optimizer = p_workspace.model.get_optimizer(**p_cfg.optimizer)
    p_workspace.load_payload(p_payload, exclude_keys=None, include_keys=None)

    ct_predictor = p_workspace.model.to(device)
    ct_predictor.eval()
    print("keypose policy loaded")

    """ --- 模型预测 --- """

    ### 循环数据版本 ###
    print(f"\n--- Evaluating episode {epi_idx} ---")
    ## episode path
    episode_path = os.path.join(dataset_path, f'episode_{epi_idx}_contact.hdf5')
    if not os.path.exists(episode_path):
        raise FileNotFoundError(f"Dataset path {episode_path} does not exist.")
    ## output path
    output_path = output_dir
    if not os.path.exists(output_path):
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    ## 评估
    evaluate_episode(ct_predictor, episode_path, output_path, p_cfg, epi_idx, device, edge_name=edge_name)

    # print("Evaluation completed for all episodes.")


if __name__ == "__main__":
    main()



    # ### 循环数据版本 ###
    # for epi_idx in range(epi_num):
    #     print(f"\n--- Evaluating episode {epi_idx} ---")
    #     ## episode path
    #     episode_path = os.path.join(dataset_path, f'kp_episode_{epi_idx}.hdf5')
    #     if not os.path.exists(episode_path):
    #         raise FileNotFoundError(f"Dataset path {episode_path} does not exist.")
    #     ## output path
    #     # output_path = os.path.join(output_dir, f'epi_{epi_idx}')\
    #     output_path = output_dir
    #     if not os.path.exists(output_path):
    #         pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    #     ## 评估
    #     evaluate_episode(ct_predictor, episode_path, output_path, epi_idx, device)
