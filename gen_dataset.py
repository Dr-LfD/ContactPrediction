''' 
output: episode_{demo_id}_contact.hdf5
    - obs
        - image:
            - view_wristview [T, 128, 128, 3] -- original image
            - view_agentview [T, 128, 128, 3] -- original image
    - label
        - obj1_obj2 [T] -- contact label (0/1)
        - ...

usage: python gen_dataset.py
note: totally export 46 episodes for kitchen task
'''

import argparse
import json
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import h5py
import networkx as nx
import numpy as np


def get_sg(hdf5_group, sg_name):
    sg_json = hdf5_group[sg_name][()] if sg_name in hdf5_group else None
    if sg_json is None:
        return None
    sg_str = sg_json.decode('utf-8')
    sg = nx.node_link_graph(json.loads(sg_str))
    return sg

def get_changed_edges(sg1, sg2):
    if sg1 is None or sg2 is None:
        return set()
    edges1 = set(map(frozenset, sg1.edges()))
    edges2 = set(map(frozenset, sg2.edges()))
    return edges1.symmetric_difference(edges2)
    
def get_excluded_frames(sg_info, demo_len):
    exclude_frames = set()
    for skill_name, skill_info in sg_info.items():
        if 'bimanual' in skill_name:
            pre_sg = get_sg(skill_info, 'pre_sg')
            exclude_start = pre_sg.graph['idx_list'][-1] + 1

            eff_sg = get_sg(skill_info, 'eff_sg')
            if eff_sg is not None:
                exclude_end = eff_sg.graph['idx_list'][0] - 1
                raise NotImplementedError("bimanual  eff_sg not implemented")
            else:
                exclude_end = demo_len
            exclude_frames = exclude_frames.union(range(exclude_start, exclude_end +1))
    return exclude_frames

def get_all_changes(sg_info):

    all_edges = set()
    all_sgs = {}
    for skill_name, skill_info in sg_info.items():
        if 'bimanual' in skill_name:
            ## all sg changes are extracted from unimanual skills only
            continue

        else:
            pre_sg = get_sg(skill_info, 'pre_sg')
            cur_sg = get_sg(skill_info, 'cur_sg')
            eff_sg = get_sg(skill_info, 'eff_sg')

            changed_edges = get_changed_edges(pre_sg, cur_sg)
            changed_edges = changed_edges.union(get_changed_edges(cur_sg, eff_sg))
            
            all_edges = all_edges.union(changed_edges)

            for sg in [pre_sg, cur_sg, eff_sg]:
                if sg is None:
                    continue
                all_sgs[sg.name] = sg

    all_edges = list(map(tuple, all_edges))

    all_edges_ordered = []
    for edge in all_edges:
        if 'robot' in edge[1]:
            edge = (edge[1], edge[0])
        all_edges_ordered.append(edge)
    return all_edges_ordered, all_sgs


def _filter_and_order_robot_object_edges(edges, robots):
    robot_set = set(robots)
    robot_object_edges = []
    for edge in edges:
        robot_nodes = [node for node in edge if node in robot_set]
        if len(robot_nodes) != 1:
            continue

        robot_node = robot_nodes[0]
        object_node = edge[1] if edge[0] == robot_node else edge[0]
        robot_object_edges.append((robot_node, object_node))
    return list(dict.fromkeys(robot_object_edges))


def _resize_image_sequence(image_sequence, image_size):
    width, height = image_size
    return np.asarray(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in image_sequence],
        dtype=image_sequence.dtype,
    )


def _build_binary_label(length, positive_indices):
    label = np.zeros(length, dtype=np.float32)
    if len(positive_indices):
        label[np.asarray(sorted(positive_indices), dtype=np.int32)] = 1.0
    return label


PROCESSED_CONTACT_INDEX_KEYS = (
    "grasp_ids",
    "holding_ids",
)

REAL_ALOHA_ARM_CHOICES = ("left_arm", "right_arm")


def _read_processed_index_set(obj_group, key: str) -> set:
    if key not in obj_group:
        return set()
    return {int(idx) for idx in obj_group[key][()].tolist()}


def _build_grasp_hold_reconstructed_indices(grasp_ids, holding_ids):
    if not grasp_ids or not holding_ids:
        return set()
    return set(range(min(grasp_ids), max(holding_ids) + 1))


def _build_post_holding_invalid_indices(holding_ids, length):
    """Frames strictly after max(holding_ids) are invalid for training."""
    if not holding_ids:
        return set()
    invalid_start = max(holding_ids) + 1
    if invalid_start >= length:
        return set()
    return set(range(invalid_start, length))


def _resolve_arm_export(
    arm,
    eff_robot_name,
    pre_robot_name,
    left_image_key,
    right_image_key,
):
    if arm not in REAL_ALOHA_ARM_CHOICES:
        raise ValueError(
            f"arm must be one of {REAL_ALOHA_ARM_CHOICES}; got {arm!r}"
        )
    if arm == "left_arm":
        return eff_robot_name, left_image_key
    return pre_robot_name, right_image_key


def _build_bimanual_unknown_indices(pre_contact_indices, eff_contact_indices):
    if not pre_contact_indices or not eff_contact_indices:
        return set()

    unknown_start = max(pre_contact_indices) + 1
    unknown_end = min(eff_contact_indices) - 1
    if unknown_start > unknown_end:
        return set()
    return set(range(unknown_start, unknown_end + 1))


def _build_valid_mask(length, unknown_indices):
    valid = np.ones(length, dtype=bool)
    bounded_unknown_indices = sorted(idx for idx in unknown_indices if 0 <= idx < length)
    if bounded_unknown_indices:
        valid[np.asarray(bounded_unknown_indices, dtype=np.int32)] = False
    return valid


def _build_robot_object_sg_json_sequence(length, edge_positive_indices, bimanual_unknown_edges=None):
    bimanual_unknown_edges = bimanual_unknown_edges or {}
    nodes = set()
    for robot_name, object_name in edge_positive_indices:
        nodes.add(robot_name)
        nodes.add(object_name)
    for edge in bimanual_unknown_edges:
        nodes.update(edge)

    sg_json_sequence = []
    for frame_idx in range(length):
        sg = nx.Graph(name=f"frame_{frame_idx}")
        sg.graph["idx_list"] = [frame_idx]
        sg.add_nodes_from(sorted(nodes))
        for edge, unknown_indices in bimanual_unknown_edges.items():
            if frame_idx in unknown_indices:
                sg.add_edge(*edge)
        for edge, positive_indices in edge_positive_indices.items():
            if frame_idx in positive_indices:
                sg.add_edge(*edge)
        sg_json_sequence.append(json.dumps(nx.node_link_data(sg)))
    return sg_json_sequence


def _iter_episode_ids(folder_path):
    episode_pattern = re.compile(r"^episode_(\d+)\.hdf5$")
    for file_name in sorted(os.listdir(folder_path)):
        match = episode_pattern.match(file_name)
        if match is None:
            continue
        yield int(match.group(1))


def export_real_aloha_contacts_from_folders(
    raw_dir,
    processed_dir,
    output_dir,
    arm,
    object_name="cup",
    left_image_key="cam_left_wrist",
    right_image_key="cam_right_wrist",
    high_image_key="cam_high",
    eff_robot_name="left_arm",
    pre_robot_name="right_arm",
    image_size=(128, 128),
    label_source="reconstructed_sg",
):
    if arm is None:
        raise ValueError("arm is required for export_real_aloha_contacts_from_folders")
    if label_source not in {"reconstructed_sg", "event_ids"}:
        raise ValueError("label_source must be one of: reconstructed_sg, event_ids")

    robot_name, wrist_image_key = _resolve_arm_export(
        arm,
        eff_robot_name,
        pre_robot_name,
        left_image_key,
        right_image_key,
    )
    edge_name = f"{robot_name}_{object_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for episode_id in _iter_episode_ids(processed_dir):
        raw_path = os.path.join(raw_dir, f"episode_{episode_id}.hdf5")
        processed_path = os.path.join(processed_dir, f"episode_{episode_id}.hdf5")

        if not os.path.exists(raw_path):
            print(f"skip episode_{episode_id}: missing raw file {raw_path}")
            continue

        with h5py.File(raw_path, "r") as raw_file, h5py.File(processed_path, "r") as processed_file:
            if object_name not in processed_file:
                print(f"skip episode_{episode_id}: missing object group '{object_name}' in {processed_path}")
                continue

            left_images = raw_file[f"observations/images/{left_image_key}"][()]
            right_images = raw_file[f"observations/images/{right_image_key}"][()]
            high_images = raw_file[f"observations/images/{high_image_key}"][()]
            demo_len = left_images.shape[0]

            if right_images.shape[0] != demo_len:
                raise ValueError(f"episode_{episode_id} wrist image length mismatch: {left_images.shape[0]} vs {right_images.shape[0]}")
            if high_images.shape[0] != demo_len:
                raise ValueError(
                    f"episode_{episode_id} {high_image_key} length mismatch: "
                    f"{high_images.shape[0]} vs {demo_len}"
                )

            obj_group = processed_file[object_name]
            present_index_keys = [
                key for key in PROCESSED_CONTACT_INDEX_KEYS if key in obj_group
            ]
            if not present_index_keys:
                print(
                    f"skip episode_{episode_id}: no contact index keys "
                    f"{PROCESSED_CONTACT_INDEX_KEYS} in {processed_path}"
                )
                continue

            missing_index_keys = [
                key for key in PROCESSED_CONTACT_INDEX_KEYS if key not in obj_group
            ]
            if missing_index_keys:
                print(
                    f"note episode_{episode_id}: missing processed keys {missing_index_keys} "
                    f"in {processed_path} — treating as empty"
                )

            grasp_ids = _read_processed_index_set(obj_group, "grasp_ids")
            holding_ids = _read_processed_index_set(obj_group, "holding_ids")

            if label_source == "event_ids":
                positive_indices = grasp_ids | holding_ids
            else:
                positive_indices = _build_grasp_hold_reconstructed_indices(
                    grasp_ids, holding_ids
                )

            out_of_bounds = [idx for idx in positive_indices if idx < 0 or idx >= demo_len]
            if out_of_bounds:
                raise ValueError(
                    f"episode_{episode_id} edge {edge_name} has out-of-bounds indices: "
                    f"{out_of_bounds[:10]}"
                )

            left_images = _resize_image_sequence(left_images, image_size)
            right_images = _resize_image_sequence(right_images, image_size)
            high_images = _resize_image_sequence(high_images, image_size)

            contact_label = _build_binary_label(demo_len, positive_indices)
            invalid_indices = _build_post_holding_invalid_indices(holding_ids, demo_len)
            valid_mask = _build_valid_mask(demo_len, invalid_indices)
            sg_json_sequence = _build_robot_object_sg_json_sequence(
                demo_len,
                {(robot_name, object_name): positive_indices},
            )

        save_path = os.path.join(output_dir, f"episode_{episode_id}_contact.hdf5")
        with h5py.File(save_path, "w") as out_file:
            obs_group = out_file.create_group("observations")
            image_group = obs_group.create_group("images")
            image_group.create_dataset(left_image_key, data=left_images)
            image_group.create_dataset(right_image_key, data=right_images)
            image_group.create_dataset(high_image_key, data=high_images)

            label_group = out_file.create_group("label")
            label_group.create_dataset(edge_name, data=contact_label)

            valid_group = out_file.create_group("valid")
            valid_group.create_dataset(edge_name, data=valid_mask)

            sg_group = out_file.create_group("sg_info")
            sg_group.create_dataset(
                "robot_object_contact_sg_seq",
                data=np.asarray(sg_json_sequence, dtype=h5py.string_dtype(encoding="utf-8")),
            )
            sg_group.attrs["label_source"] = label_source
            sg_group.attrs["arm"] = arm
            sg_group.attrs["edge"] = edge_name
        

def identify_contacts_and_export_hdf5(
    hdf5_file_path,
    start_demo_id,
    end_demo_id,
    robots=['robot0'],
    dataset_save_path=None,
):

    f_in = h5py.File(hdf5_file_path, 'r')

    demos = list(f_in["data"].keys())
    inds = [int(demo.split('_')[-1]) for demo in demos]
    inds = sorted(inds)

    # sg_params = json.loads(f_in['sg_params'][()])
    # instance_name2id = sg_params['instance_name2id']
    # interested_objs = sg_params['interested_objs']

    # task_prefix = hdf5_file_path.split('/')[-1].split('.hdf5')[0].split('_')[0]
    # root_dir = f'data/training/{task_prefix}'
    # if os.path.exists(root_dir) == False:
    #     os.makedirs(root_dir)

    assert dataset_save_path is not None
    if os.path.exists(dataset_save_path) == False:
        os.makedirs(dataset_save_path)

    for demo_id in range(start_demo_id, end_demo_id):
        if demo_id not in inds:
            print(f'demo_{demo_id} not in the hdf5 file!!!')
            continue

        ## 获取 scene graph 信息
        sg_info = f_in[f'data/demo_{demo_id}/sg_info']
        all_edges, all_sgs = get_all_changes(sg_info)
        all_edges = _filter_and_order_robot_object_edges(all_edges, robots)
        
        ## 获取原始图像 - wristview & agentview
        # wristview_image = f_in[f'data/demo_{demo_id}/obs/robot0_eye_in_hand_image']
        agentview_image = f_in[f'data/demo_{demo_id}/obs/agentview_image']

        image_dict = {
            # 'cam_robot0': wristview_image,
            'cam_agent': agentview_image
        }

        for rbt in robots:
            wristview_image = f_in[f'data/demo_{demo_id}/obs/{rbt}_eye_in_hand_image']
            image_dict[f'cam_{rbt}'] = wristview_image

        demo_len = agentview_image.shape[0]
        valid_mask = _build_valid_mask(demo_len, get_excluded_frames(sg_info, demo_len))

        ## if edge connected, then idx from this sg is positive.
        ## NOTE: if there are 2 same objs, nn may be confused
        contact_positive_samples = {tuple(edge) :set() for edge in all_edges}
        for sg in all_sgs.values():
            for edge in all_edges:
                if sg.has_edge(edge[0], edge[1]):
                    contact_positive_samples[edge]= contact_positive_samples[edge].union(set(sg.graph['idx_list']))
        
        
        ## *** contact_nagative_samples = {{'aa_bb': {0,1,2....}}} -- contact_relation, negative idx set
        contact_negative_samples = {edge :set() for edge in all_edges}
        for edge in all_edges:
            contact_negative_samples[edge] = set(range(demo_len)) - contact_positive_samples[edge]


        ## *** contact_positive_samples_filtered = {{'aa_bb': {0,1,2....}}} -- contact_relation, filtered positive idx set
        contact_positive_samples_filtered = {tuple(edge) :set() for edge in all_edges}
        for edge, idx_set in contact_positive_samples.items():
        ## adjust the positive sample according to gripper action
            for i in idx_set:
                if 'robot0' in edge:
                    gripper_action = f_in[f'data/demo_{demo_id}/actions'][()][:, 6]
                    if gripper_action[i] < -0.1:
                        continue
                if 'robot1' in edge:
                    gripper_action = f_in[f'data/demo_{demo_id}/actions'][()][:, 13]
                    if gripper_action[i] < -0.1:
                        continue

                contact_positive_samples_filtered[edge].add(i)

        # print(f"DEMO_{demo_id} - getting contact samples done!")

        ''' 整理 contact_positive_samples_filtered, 生成 contact_label_dict '''
        contact_label_dict = {}
        for key, item in contact_positive_samples_filtered.items():
            ## key --> string
            contact_key = key[0] + '_' + key[1]
            ## positive index list
            contact_positive_index = list(item)

            # print(f'-- {contact_key} - positive num: {len(contact_positive_index)}')

            contact_label_dict[contact_key] = _build_binary_label(demo_len, contact_positive_index)

        # print(f"DEMO_{demo_id} - getting contact labels done!")

        ''' --- 保存训练数据 hdf5 --- '''

        save_path = f'{dataset_save_path}/episode_{demo_id}_contact.hdf5'

        with h5py.File(save_path, 'w') as f_out:
            obs_grp = f_out.create_group('observations')
            img_grp = obs_grp.create_group('images')
            for img_key, img_data in image_dict.items():
                img_grp.create_dataset(img_key, data = img_data)

            grp = f_out.create_group('label')
            for contact_key, contact_label in contact_label_dict.items():
                grp.create_dataset(contact_key, data = contact_label)

            valid_grp = f_out.create_group('valid')
            for contact_key in contact_label_dict:
                valid_grp.create_dataset(contact_key, data=valid_mask)

        # print(f"{save_path} - saving contact data done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sg_info", "real_aloha"], default="real_aloha")
    parser.add_argument("--hdf5-file-path", default=None)
    parser.add_argument("--start-demo-id", type=int, default=0)
    parser.add_argument("--end-demo-id", type=int, default= 0)
    parser.add_argument("--robots", nargs="*", default=["robot0"])
    parser.add_argument("--dataset-save-path", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--object-name", default="cup")
    parser.add_argument("--left-image-key", default="cam_left_wrist")
    parser.add_argument("--right-image-key", default="cam_right_wrist")
    parser.add_argument("--high-image-key", default="cam_high")
    parser.add_argument("--eff-robot-name", default="left_arm")
    parser.add_argument("--pre-robot-name", default="right_arm")
    parser.add_argument(
        "--arm",
        choices=list(REAL_ALOHA_ARM_CHOICES),
        default=None,
        help="Which arm to export (required for --mode real_aloha). "
             "left_arm uses --eff-robot-name and cam_left_wrist; "
             "right_arm uses --pre-robot-name and cam_right_wrist.",
    )
    parser.add_argument("--label-source", choices=["reconstructed_sg", "event_ids"], default="reconstructed_sg")
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "sg_info":
        if args.hdf5_file_path is None or args.dataset_save_path is None:
            raise ValueError("--hdf5-file-path and --dataset-save-path are required for mode=sg_info")
        identify_contacts_and_export_hdf5(
            args.hdf5_file_path,
            args.start_demo_id,
            args.end_demo_id,
            robots=args.robots,
            dataset_save_path=args.dataset_save_path,
        )
    else:
        if args.raw_dir is None or args.processed_dir is None or args.output_dir is None:
            raise ValueError("--raw-dir, --processed-dir, and --output-dir are required for mode=real_aloha")
        if args.arm is None:
            raise ValueError("--arm is required for mode=real_aloha")

        export_real_aloha_contacts_from_folders(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            arm=args.arm,
            object_name=args.object_name,
            left_image_key=args.left_image_key,
            right_image_key=args.right_image_key,
            high_image_key=args.high_image_key,
            eff_robot_name=args.eff_robot_name,
            pre_robot_name=args.pre_robot_name,
            image_size=(args.image_width, args.image_height),
            label_source=args.label_source,
        )
