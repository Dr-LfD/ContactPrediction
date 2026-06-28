#!/usr/bin/env python3
import h5py, glob, os, numpy as np

def convert_one(index, src_path, out_dir, edge, camera):
    src_name = os.path.basename(src_path)
    ann_key = "annotations/%s/binary" % edge
    with h5py.File(src_path, "r") as src:
        cam = src[camera][()]
        ann = src[ann_key][()].astype(np.float32)
        T = cam.shape[0]
    out_path = os.path.join(out_dir, "episode_%d_contact.hdf5" % index)
    with h5py.File(out_path, "w") as dst:
        dst.create_dataset("observations/images/%s" % camera, data=cam)
        dst.create_dataset("label/%s" % edge, data=ann)
        dst.create_dataset("valid/%s" % edge, data=np.ones(T, dtype=bool))
        dst.attrs["source_file"] = src_name
    print("  [%02d] %s: T=%d pos=%d neg=%d" % (index, src_name, T, int(ann.sum()), T-int(ann.sum())))
    return int(ann.sum()), T - int(ann.sum())

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tasks = [
    dict(name="screwdriver",
         src_dirs=[BASE+"/data/selfplay/rollout/screwdriver"],
         out_dir=BASE+"/data/training/screwdriver_rollout/contact",
         edge="left_arm_screwdriver", camera="cam_left_wrist"),
    dict(name="clean_cup",
         src_dirs=[BASE+"/data/rollout/sponge",
                   BASE+"/data/selfplay/rollout/sponge"],
         out_dir=BASE+"/data/training/clean_cup_rollout/contact",
         val_src_dirs=[BASE+"/data/rollout/sponge/validate",
                       BASE+"/data/selfplay/rollout/sponge/validate"],
         val_out_dir=BASE+"/data/training/clean_cup_rollout_validate/contact",
         edge="right_arm_sponge", camera="cam_right_wrist"),
    dict(name="handoff_cup",
         src_dirs=[BASE+"/data/selfplay/rollout/cup"],
         out_dir=BASE+"/data/training/handoff_cup_rollout/contact",
         edge="right_arm_cup", camera="cam_right_wrist"),
]

for task in tasks:
    print("\n=== %s ===" % task["name"])
    files = []
    for d in task["src_dirs"]:
        files += sorted(glob.glob(os.path.join(d, "*.hdf5")))
    os.makedirs(task["out_dir"], exist_ok=True)
    total_pos = total_neg = 0
    for idx, fpath in enumerate(files):
        p, n = convert_one(idx, fpath, task["out_dir"], task["edge"], task["camera"])
        total_pos += p
        total_neg += n
    print("  Done: %d episodes, pos=%d, neg=%d" % (len(files), total_pos, total_neg))

    if "val_src_dirs" in task:
        print("\n=== %s (validate) ===" % task["name"])
        val_files = []
        for d in task["val_src_dirs"]:
            val_files += sorted(glob.glob(os.path.join(d, "*.hdf5")))
        os.makedirs(task["val_out_dir"], exist_ok=True)
        val_pos = val_neg = 0
        for idx, fpath in enumerate(val_files):
            p, n = convert_one(idx, fpath, task["val_out_dir"], task["edge"], task["camera"])
            val_pos += p
            val_neg += n
        print("  Done: %d episodes, pos=%d, neg=%d" % (len(val_files), val_pos, val_neg))
