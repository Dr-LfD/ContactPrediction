from collections import defaultdict
from typing import Dict

import torch

from contact_pred.scripts.common.pytorch_util import dict_apply


_DECISION_THRESHOLD = 0.5


def _empty_edge_counters():
    return {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}


def compute_batch_accuracy_metrics(pred_label, gt_label, edge_ids, edge_id_to_name):
    pred_binary = pred_label > _DECISION_THRESHOLD
    gt_binary = gt_label > _DECISION_THRESHOLD
    flat_shape = (gt_binary.shape[0], -1)
    correct_mask = pred_binary.eq(gt_binary).reshape(flat_shape).all(dim=1)
    pred_pos_any = pred_binary.reshape(flat_shape).any(dim=1)
    gt_pos_any = gt_binary.reshape(flat_shape).any(dim=1)

    edge_ids = edge_ids.reshape(-1).detach().cpu().tolist()
    correct_list = correct_mask.detach().cpu().tolist()
    pred_pos_list = pred_pos_any.detach().cpu().tolist()
    gt_pos_list = gt_pos_any.detach().cpu().tolist()

    per_edge = defaultdict(_empty_edge_counters)
    for edge_id, is_correct, pred_pos, gt_pos in zip(
        edge_ids, correct_list, pred_pos_list, gt_pos_list
    ):
        edge_name = edge_id_to_name[int(edge_id)]
        bucket = per_edge[edge_name]
        bucket["total"] += 1
        bucket["correct"] += int(is_correct)
        if gt_pos and pred_pos:
            bucket["tp"] += 1
        elif gt_pos and not pred_pos:
            bucket["fn"] += 1
        elif (not gt_pos) and pred_pos:
            bucket["fp"] += 1
        else:
            bucket["tn"] += 1

    return {
        "overall_correct": int(correct_mask.sum().item()),
        "overall_total": int(correct_mask.numel()),
        "per_edge": dict(per_edge),
    }


_EDGE_COUNTER_KEYS = ("correct", "total", "tp", "fp", "fn", "tn")


def merge_accuracy_metrics(accumulator, batch_metrics):
    accumulator["overall_correct"] += batch_metrics["overall_correct"]
    accumulator["overall_total"] += batch_metrics["overall_total"]
    for edge_name, metrics in batch_metrics["per_edge"].items():
        bucket = accumulator["per_edge"].setdefault(edge_name, _empty_edge_counters())
        for key in _EDGE_COUNTER_KEYS:
            bucket[key] += metrics.get(key, 0)
    return accumulator


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def finalize_accuracy_metrics(metrics):
    result = {
        "overall_accuracy": 0.0,
        "per_edge_accuracy": {},
        "per_edge_precision": {},
        "per_edge_recall": {},
        "per_edge_f1": {},
        "per_edge_positive_rate": {},
        "per_edge_sample_count": {},
        "per_edge_confusion": {},
    }
    if metrics["overall_total"] > 0:
        result["overall_accuracy"] = metrics["overall_correct"] / metrics["overall_total"]

    accuracies = []
    f1_scores = []
    weighted_correct = 0
    weighted_total = 0
    for edge_name in sorted(metrics["per_edge"].keys()):
        bucket = metrics["per_edge"][edge_name]
        total = bucket["total"]
        if total == 0:
            continue
        tp, fp, fn, tn = bucket["tp"], bucket["fp"], bucket["fn"], bucket["tn"]
        accuracy = bucket["correct"] / total
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        positive_rate = _safe_div(tp + fn, total)

        result["per_edge_accuracy"][edge_name] = accuracy
        result["per_edge_precision"][edge_name] = precision
        result["per_edge_recall"][edge_name] = recall
        result["per_edge_f1"][edge_name] = f1
        result["per_edge_positive_rate"][edge_name] = positive_rate
        result["per_edge_sample_count"][edge_name] = total
        result["per_edge_confusion"][edge_name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        accuracies.append(accuracy)
        f1_scores.append(f1)
        weighted_correct += bucket["correct"]
        weighted_total += total

    if accuracies:
        result["mean_per_edge_accuracy"] = sum(accuracies) / len(accuracies)
        result["min_per_edge_accuracy"] = min(accuracies)
        result["mean_per_edge_f1"] = sum(f1_scores) / len(f1_scores)
        result["min_per_edge_f1"] = min(f1_scores)
        result["weighted_accuracy"] = _safe_div(weighted_correct, weighted_total)
    else:
        result["mean_per_edge_accuracy"] = 0.0
        result["min_per_edge_accuracy"] = 0.0
        result["mean_per_edge_f1"] = 0.0
        result["min_per_edge_f1"] = 0.0
        result["weighted_accuracy"] = 0.0
    return result


def evaluate_predictor_accuracy(predictor, dataloader, device, edge_id_to_name):
    predictor.eval()
    metrics = {
        "overall_correct": 0,
        "overall_total": 0,
        "per_edge": {},
    }
    with torch.no_grad():
        for batch in dataloader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            pred_label = predictor.predict_keypose_and_mode(batch["obs"])["label"]
            batch_metrics = compute_batch_accuracy_metrics(
                pred_label=pred_label,
                gt_label=batch["label"],
                edge_ids=batch["edge_id"],
                edge_id_to_name=edge_id_to_name,
            )
            merge_accuracy_metrics(metrics, batch_metrics)
    return finalize_accuracy_metrics(metrics)


def compare_accuracy_results(mixed_per_edge_accuracy: Dict[str, float], baseline_per_edge_accuracy: Dict[str, float]):
    regressions = {}
    for edge_name, baseline_accuracy in baseline_per_edge_accuracy.items():
        if edge_name not in mixed_per_edge_accuracy:
            raise ValueError(f"Missing mixed accuracy for edge '{edge_name}'")
        mixed_accuracy = mixed_per_edge_accuracy[edge_name]
        if mixed_accuracy < baseline_accuracy:
            regressions[edge_name] = {
                "mixed": mixed_accuracy,
                "baseline": baseline_accuracy,
            }
    return regressions


def compare_target_accuracy(mixed_per_edge_accuracy: Dict[str, float], target_accuracy: float):
    shortfalls = {}
    for edge_name, mixed_accuracy in mixed_per_edge_accuracy.items():
        if mixed_accuracy < target_accuracy:
            shortfalls[edge_name] = {
                "mixed": mixed_accuracy,
                "target": target_accuracy,
            }
    return shortfalls
