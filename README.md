# Contact Detector

Per-edge **contact predictor** for dual-arm learning-from-demonstration. Given a wrist-camera RGB
frame and an *edge text* (e.g. `right_arm_cup`), it predicts whether that arm is in contact with
that object. The model fuses a ResNet-18 image encoder with a BERT edge-text embedding through an
MLP (`ContactMlpPredictor`), and is used to mine and label policy-rollout failures.

This package is a submodule of **DR-LfD-all**, synced into `contact_detector/`.

## Installation

Set up the shared `dr-lfd` conda environment and the contact-detector dependencies per
[DR-LfD-all → Installation](../docs/source/Installation.md) (section *"Contact detector
(submodule)"*). Then set the path config:

```bash
cp .env.example .env     # set WS_ROOT to your external workspace
```

Repo-internal paths resolve automatically via the committed `.repo_root` marker; only `WS_ROOT`
(raw ALOHA data, rollout dumps, DMG datasets) needs setting.

## Data Preparation

Training consumes per-episode `episode_<i>_contact.hdf5` files under
`data/training/<task>/contact/`, each holding `/observations/images/<cam>`, `/label/<edge>`, and
`/valid/<edge>`.

### From DMG simulation (`--mode sg_info`)

Build episodes from a DexMimicGen `*_pc_instance_sg_*.hdf5` (must carry per-demo `sg_info` and a
root `sg_params` or `instance_name2id`):

```bash
python gen_dataset.py --mode sg_info \
    --hdf5-file-path ${WS_ROOT}/imitation_learning/dexmimicgen/datasets/generated/two_arm_threading_pc_instance_sg_100.hdf5 \
    --start-demo-id 0 --end-demo-id 50 --robots robot0 robot1 \
    --dataset-save-path data/training/threading
```

### From real ALOHA recordings (`--mode real_aloha`)

Export contact episodes from raw + processed ALOHA folders. Binary contact labels are derived from
the processed grasp/holding indices:

```bash
python gen_dataset.py --mode real_aloha \
    --raw-dir ${WS_ROOT}/aloha_data/screwdriver/raw \
    --processed-dir ${WS_ROOT}/aloha_data/screwdriver/processed \
    --output-dir data/training/screwdriver/contact \
    --arm left_arm --object-name screwdriver
```

### From policy rollouts / self-play

Convert annotated rollouts (hard negatives / self-play) into the training format:

```bash
# one rollout dir -> training episodes
python tools/convert_rollout_to_training.py \
    --rollout-dir data/rollout --pattern "test_08_*.hdf5" \
    --edge right_arm_cup --camera cam_right_wrist \
    --output-dir data/training/handoff_cup_rollout/contact

# batch-convert the configured self-play + rollout dirs
python tools/convert_selfplay.py
```

## Rollout Annotation (`tools/annotate_rollout.py`)

Manually annotate ground-truth contact on policy rollout recordings — used to collect failure data
where the predictor is unreliable.

**Prerequisite:** run `eval_contact_predictor_rollout.py` first so that
`predictions/<edge>/{binary,label}` exist in the rollout HDF5 (the annotation seeds from them).

```bash
# List available edges in a rollout file
python tools/annotate_rollout.py --hdf5 data/rollout/test_08_18.33.55.hdf5

# Annotate one edge (seeds from model prediction)
python tools/annotate_rollout.py \
    --hdf5 data/rollout/test_08_18.33.55.hdf5 --edge right_arm_cup

# Resume / overwrite a prior session
python tools/annotate_rollout.py --hdf5 ... --edge right_arm_cup --resume-existing
python tools/annotate_rollout.py --hdf5 ... --edge right_arm_cup --overwrite
```

**GUI controls:**

| Key | Action |
|-----|--------|
| `←` / `→` | Step ±1 frame |
| `↑` / `↓` | Step ±10 frames |
| `Home` / `End` | First / last frame |
| `Space` | Play / pause |
| `I` | Set in-point at current frame |
| `O` | Fill [in-point, current] with contact=1, clear in-point |
| `D` | Delete the positive interval containing the current frame |
| `N` / `P` | Jump to next / previous disagreement with the prediction seed |
| `C` | Clear all — set entire annotation to no-contact |
| `R` | Reset annotation to prediction seed |
| `S` | Save to `annotations/<edge>/binary` in the HDF5 |
| `Q` | Quit (press twice if there are unsaved changes) |

The GUI shows both wrist cameras side-by-side and a timeline with the model's prediction seed (gray)
and your annotation (blue). `N` / `P` are the key shortcut for failure mining — they jump straight to
frames where your annotation disagrees with the model. Annotations are written atomically to
`annotations/<edge>/binary` `(T,) uint8` with provenance attrs (`annotator`, `annotated_at_utc`,
`edits_count`, `source_predictions_threshold`).

## Training

Configs live under `contact_pred/scripts/config/`; each `train_*_workspace.yaml` selects a task.
With the `dr-lfd` env active:

```bash
# clean-cup (right-arm sponge) + self-play
python train.py --config-name train_clean_cup_workspace

# screwdriver (left arm) + self-play
python train.py --config-name train_contact_mlp_multi_edge_screwdriver_workspace

# joint screwdriver + sponge + cup
python train.py --config-name train_screwdriver_sponge_cup_workspace
```

`train_contact_mlp_workspace` (single-task base) and `train_contact_mlp_multi_edge_workspace`
(multi-edge base) hold the shared predictor/optimizer defaults the task workspaces inherit. Models
use `cam_wrist` RGB + edge-text conditioning with edge-aware positive weighting in the contact BCE
loss. Checkpoints land under `data/outputs/<date>/<run>/checkpoints/`.

## Inference / Evaluation

Run the predictor frame-by-frame over rollout HDF5s; it writes `predictions/<edge>/{label,binary}`
back into each file:

```bash
python eval_contact_predictor_rollout.py \
    --checkpoint data/outputs/<date>/<run>/checkpoints/latest.ckpt \
    --rollout-dir data/rollout \
    --edge-text right_arm_sponge \
    --overwrite
```

For labeled rollouts, `tools/eval_rollout_all.sh` writes predictions and reports accuracy /
precision / recall / F1:

```bash
bash tools/eval_rollout_all.sh <checkpoint> [edge] [rollout-dir] [threshold]
# e.g.
bash tools/eval_rollout_all.sh \
    data/outputs/<date>/<run>/checkpoints/latest.ckpt \
    left_arm_screwdriver data/rollout/screwdriver_test 0.5
```

## Results — Screwdriver (`left_arm_screwdriver`, 2026-05-26)

Trained with `train_contact_mlp_multi_edge_screwdriver_workspace` on
`data/training/screwdriver/contact` (real episodes) + `data/training/screwdriver_rollout/contact`
(self-play hard negatives), evaluated on `data/rollout/screwdriver_test`:

| Metric | Value |
|--------|-------|
| Accuracy | 96.37% |
| Recall | 95.47% |
| Precision | 94.13% |
| FPR | 3.14% |
| FNR | 4.53% |
| F1 | 0.948 |
