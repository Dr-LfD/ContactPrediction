if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import math
import shutil
from collections import defaultdict
from contact_pred.scripts.workspace.base_workspace import BaseWorkspace
from contact_pred.scripts.policy.contact_mlp_predictor import ContactMlpPredictor
from contact_pred.scripts.dataset.contact_base_dataset import ContactBaseDataset
from contact_pred.scripts.common.checkpoint_util import TopKCheckpointManager
from contact_pred.scripts.common.contact_accuracy import (
    compute_batch_accuracy_metrics,
    finalize_accuracy_metrics,
    merge_accuracy_metrics,
)
from contact_pred.scripts.common.json_logger import JsonLogger
from contact_pred.scripts.common.pytorch_util import dict_apply, optimizer_to
from contact_pred.scripts.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainContactMlpWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: ContactMlpPredictor = hydra.utils.instantiate(cfg.predictor)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: ContactBaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, ContactBaseDataset)
        train_dataloader_kwargs = OmegaConf.to_container(cfg.dataloader, resolve=True)
        train_dataloader = DataLoader(dataset, **train_dataloader_kwargs)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader_kwargs = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
        val_dataloader = DataLoader(val_dataset, **val_dataloader_kwargs)

        self.model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        ## debug mode
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                self.model.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        accuracy_metrics = {
                            "overall_correct": 0,
                            "overall_total": 0,
                            "per_edge": {},
                        }
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # val_loss must be unweighted BCE; per-sample loss_weight
                                # is built from train-split positive rates and would bias
                                # both the logged val_loss and the topk checkpoint monitor.
                                unweighted_batch = {k: v for k, v in batch.items() if k != "loss_weight"}
                                loss = self.model.compute_loss(unweighted_batch)
                                val_losses.append(loss.item())
                                val_metrics = self._compute_accuracy_metrics(
                                    batch=batch,
                                    predictor=self.model,
                                    edge_id_to_name=getattr(val_dataset, "edge_id_to_name", {}),
                                )
                                merge_accuracy_metrics(accuracy_metrics, val_metrics)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = float(np.mean(val_losses))
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            finalized_metrics = finalize_accuracy_metrics(accuracy_metrics)
                            step_log["val_accuracy_overall"] = finalized_metrics["overall_accuracy"]
                            step_log["val_accuracy_mean_per_edge"] = finalized_metrics["mean_per_edge_accuracy"]
                            step_log["val_accuracy_min_per_edge"] = finalized_metrics["min_per_edge_accuracy"]
                            step_log["val_accuracy_weighted"] = finalized_metrics["weighted_accuracy"]
                            step_log["val_f1_mean_per_edge"] = finalized_metrics["mean_per_edge_f1"]
                            step_log["val_f1_min_per_edge"] = finalized_metrics["min_per_edge_f1"]
                            for edge_name, accuracy in finalized_metrics["per_edge_accuracy"].items():
                                step_log[f"val_accuracy_edge_{edge_name}"] = accuracy
                            for edge_name, value in finalized_metrics["per_edge_precision"].items():
                                step_log[f"val_precision_edge_{edge_name}"] = value
                            for edge_name, value in finalized_metrics["per_edge_recall"].items():
                                step_log[f"val_recall_edge_{edge_name}"] = value
                            for edge_name, value in finalized_metrics["per_edge_f1"].items():
                                step_log[f"val_f1_edge_{edge_name}"] = value
                            for edge_name, value in finalized_metrics["per_edge_positive_rate"].items():
                                step_log[f"val_pos_rate_edge_{edge_name}"] = value
                            for edge_name, value in finalized_metrics["per_edge_sample_count"].items():
                                step_log[f"val_n_edge_{edge_name}"] = value
                            for edge_name, confusion in finalized_metrics["per_edge_confusion"].items():
                                for cm_key, cm_value in confusion.items():
                                    step_log[f"val_{cm_key}_edge_{edge_name}"] = cm_value

                ## run rollout
                # Train-side sanity reading on a cached batch. The val twin was removed
                # because val_accuracy_overall now aggregates over the full val loader
                # with the same 0.5 threshold and is the authoritative metric.
                if (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        train_label_ratio = self.model.eval_sampling(
                            self.model,
                            train_sampling_batch,
                            device,
                        )
                    step_log['contact_correct_train'] = train_label_ratio
                    del train_label_ratio
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    if "val_loss" in step_log:
                        metric_dict["log_val_loss"] = math.log(step_log["val_loss"])
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def _compute_accuracy_metrics(self, batch, predictor, edge_id_to_name):
        result = predictor.predict_keypose_and_mode(batch["obs"])
        return compute_batch_accuracy_metrics(
            pred_label=result["label"],
            gt_label=batch["label"],
            edge_ids=batch["edge_id"],
            edge_id_to_name=edge_id_to_name,
        )

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainContactMlpWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
