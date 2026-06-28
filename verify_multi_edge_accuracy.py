import pathlib

import click
import dill
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from contact_pred.scripts.common.contact_accuracy import (
    compare_accuracy_results,
    compare_target_accuracy,
    evaluate_predictor_accuracy,
)
from contact_pred.scripts.workspace.base_workspace import BaseWorkspace


def _load_workspace_and_model(checkpoint_path, output_dir, device):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    if "model" not in workspace.__dict__:
        workspace.model = hydra.utils.instantiate(cfg.predictor)
    if "optimizer" not in workspace.__dict__:
        workspace.optimizer = workspace.model.get_optimizer(**cfg.optimizer)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    model = workspace.model.to(device)
    model.eval()
    return cfg, workspace, model


def _build_val_dataloader(cfg):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    dataloader_kwargs = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
    dataloader_kwargs["shuffle"] = False
    dataloader_kwargs.pop("sampler", None)
    dataloader = DataLoader(val_dataset, **dataloader_kwargs)
    return val_dataset, dataloader


def _format_regressions(regressions):
    lines = []
    for edge_name in sorted(regressions.keys()):
        item = regressions[edge_name]
        lines.append(
            f"{edge_name}: mixed={item['mixed']:.6f} baseline={item['baseline']:.6f}"
        )
    return "\n".join(lines)


@click.command()
@click.option("--mixed-ckpt", required=True, type=click.Path(exists=True))
@click.option(
    "--baseline-ckpt",
    "baseline_ckpts",
    multiple=True,
    type=click.Path(exists=True),
)
@click.option("--target-min-accuracy", type=float, default=None)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--json-output", is_flag=True, default=False)
def main(mixed_ckpt, baseline_ckpts, target_min_accuracy, device, json_output):
    if not baseline_ckpts and target_min_accuracy is None:
        raise click.UsageError("Provide at least one --baseline-ckpt or --target-min-accuracy.")
    device = torch.device(device)
    output_dir = str(pathlib.Path(mixed_ckpt).resolve().parent.parent / "accuracy_verification")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    mixed_cfg, _, mixed_model = _load_workspace_and_model(mixed_ckpt, output_dir, device)
    mixed_val_dataset, mixed_val_loader = _build_val_dataloader(mixed_cfg)
    mixed_metrics = evaluate_predictor_accuracy(
        predictor=mixed_model,
        dataloader=mixed_val_loader,
        device=device,
        edge_id_to_name=mixed_val_dataset.edge_id_to_name,
    )

    baseline_metrics = {}
    for baseline_ckpt in baseline_ckpts:
        baseline_cfg, _, baseline_model = _load_workspace_and_model(
            baseline_ckpt,
            output_dir,
            device,
        )
        baseline_val_dataset, baseline_val_loader = _build_val_dataloader(baseline_cfg)
        metrics = evaluate_predictor_accuracy(
            predictor=baseline_model,
            dataloader=baseline_val_loader,
            device=device,
            edge_id_to_name=baseline_val_dataset.edge_id_to_name,
        )
        if len(metrics["per_edge_accuracy"]) != 1:
            raise ValueError(
                f"Baseline checkpoint {baseline_ckpt} must evaluate exactly one edge."
            )
        edge_name, accuracy = next(iter(metrics["per_edge_accuracy"].items()))
        baseline_metrics[edge_name] = accuracy

    regressions = compare_accuracy_results(
        mixed_per_edge_accuracy=mixed_metrics["per_edge_accuracy"],
        baseline_per_edge_accuracy=baseline_metrics,
    ) if baseline_metrics else {}
    shortfalls = compare_target_accuracy(
        mixed_per_edge_accuracy=mixed_metrics["per_edge_accuracy"],
        target_accuracy=target_min_accuracy,
    ) if target_min_accuracy is not None else {}

    payload = {
        "mean_per_edge_accuracy": mixed_metrics["mean_per_edge_accuracy"],
        "overall_accuracy": mixed_metrics["overall_accuracy"],
        "per_edge_accuracy": mixed_metrics["per_edge_accuracy"],
        "baseline_regressions": regressions,
        "target_shortfalls": shortfalls,
        "target_min_accuracy": target_min_accuracy,
    }

    if json_output:
        import json
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"Mixed mean_per_edge_accuracy={mixed_metrics['mean_per_edge_accuracy']:.6f}")
        for edge_name in sorted(mixed_metrics["per_edge_accuracy"].keys()):
            print(f"MIXED {edge_name}={mixed_metrics['per_edge_accuracy'][edge_name]:.6f}")
        for edge_name in sorted(baseline_metrics.keys()):
            print(f"BASELINE {edge_name}={baseline_metrics[edge_name]:.6f}")
        if target_min_accuracy is not None:
            print(f"TARGET {target_min_accuracy:.6f}")

    if regressions:
        raise SystemExit(
            "Mixed checkpoint regressed below baseline on:\n" + _format_regressions(regressions)
        )
    if shortfalls:
        lines = []
        for edge_name in sorted(shortfalls.keys()):
            item = shortfalls[edge_name]
            lines.append(f"{edge_name}: mixed={item['mixed']:.6f} target={item['target']:.6f}")
        raise SystemExit("Mixed checkpoint is below target on:\n" + "\n".join(lines))

    if not json_output:
        if baseline_metrics:
            print("All compared mixed per-edge validation accuracies meet or exceed baseline.")
        if target_min_accuracy is not None:
            print("All mixed per-edge validation accuracies meet or exceed target.")


if __name__ == "__main__":
    main()
