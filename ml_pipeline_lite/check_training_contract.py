#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import yaml


CORE_KEYS = (
    "x",
    "x_mask",
    "conditions",
    "conditions_mask",
    "x_invisible",
    "x_invisible_mask",
    "classification",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that preprocessed EveNet parquet files match generated_event_info.yaml "
            "before training. This catches class-index overflow and tensor-shape drift on CPU."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing train_*.parquet files.",
    )
    parser.add_argument(
        "--event-info",
        type=Path,
        required=True,
        help="Path to generated_event_info.yaml used by training.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional limit on the number of train parquet files to scan. Default scans all files.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def event_info_contract(config: dict[str, Any]) -> dict[str, Any]:
    sequential = config.get("INPUTS", {}).get("SEQUENTIAL", {}).get("Source", {})
    global_cond = config.get("INPUTS", {}).get("GLOBAL", {}).get("Conditions", {})
    invisible = config.get("GENERATIONS", {}).get("Neutrinos", {})
    grouped = config.get("GROUPED_INPUTS", {}).get("SEQUENTIAL", {}).get("Source", {})
    projected = grouped.get("projected_feature_names", list(sequential.keys()))
    class_map = config.get("CLASSLABEL", {}).get("EVENT", {})
    if len(class_map) != 1:
        raise ValueError("Expected exactly one EVENT classification target in generated_event_info.yaml.")
    _, raw_labels = next(iter(class_map.items()))
    if not raw_labels or not isinstance(raw_labels[0], list):
        raise ValueError("CLASSLABEL.EVENT must contain a nested list of class labels.")
    class_labels = [str(label) for label in raw_labels[0]]
    return {
        "raw_sequential_dim": len(sequential),
        "projected_sequential_dim": len(projected),
        "global_dim": len(global_cond),
        "invisible_dim": len(invisible),
        "class_labels": class_labels,
    }


def require_keys(events: ak.Array, path: Path) -> None:
    print(events.fields)
    missing = [key for key in CORE_KEYS if key not in events.fields]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")


def as_numpy(events: ak.Array, key: str) -> np.ndarray:
    return np.asarray(ak.to_numpy(events[key], allow_missing=False))


def check_batch(
    events: ak.Array,
    *,
    path: Path,
    contract: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    require_keys(events, path)

    x = as_numpy(events, "x")
    x_mask = as_numpy(events, "x_mask")
    conditions = as_numpy(events, "conditions")
    conditions_mask = as_numpy(events, "conditions_mask")
    x_invisible = as_numpy(events, "x_invisible")
    x_invisible_mask = as_numpy(events, "x_invisible_mask")
    classification = np.asarray(as_numpy(events, "classification"), dtype=np.int64)

    if x.ndim != 3:
        raise ValueError(f"{path}: x must be rank-3, got shape {x.shape}")
    if x.shape[-1] != contract["raw_sequential_dim"]:
        raise ValueError(
            f"{path}: x feature dim {x.shape[-1]} does not match event-info raw sequential dim "
            f"{contract['raw_sequential_dim']}"
        )
    if x_mask.shape != x.shape[:2]:
        raise ValueError(f"{path}: x_mask shape {x_mask.shape} does not match x slots {x.shape[:2]}")
    if conditions.ndim != 2:
        raise ValueError(f"{path}: conditions must be rank-2, got shape {conditions.shape}")
    if conditions.shape[-1] != contract["global_dim"]:
        raise ValueError(
            f"{path}: conditions dim {conditions.shape[-1]} does not match event-info global dim "
            f"{contract['global_dim']}"
        )
    if conditions_mask.shape != (conditions.shape[0], 1):
        raise ValueError(
            f"{path}: conditions_mask must have shape {(conditions.shape[0], 1)}, got {conditions_mask.shape}"
        )
    if x_invisible.ndim != 3:
        raise ValueError(f"{path}: x_invisible must be rank-3, got shape {x_invisible.shape}")
    if x_invisible.shape[1] != 2:
        raise ValueError(f"{path}: x_invisible second dim must be 2, got {x_invisible.shape}")
    if x_invisible.shape[-1] != contract["invisible_dim"]:
        raise ValueError(
            f"{path}: x_invisible feature dim {x_invisible.shape[-1]} does not match event-info invisible dim "
            f"{contract['invisible_dim']}"
        )
    if x_invisible_mask.shape != x_invisible.shape[:2]:
        raise ValueError(
            f"{path}: x_invisible_mask shape {x_invisible_mask.shape} does not match x_invisible slots "
            f"{x_invisible.shape[:2]}"
        )

    num_classes = len(contract["class_labels"])
    invalid_mask = (classification < -1) | (classification >= num_classes)
    if np.any(invalid_mask):
        invalid_values = np.unique(classification[invalid_mask]).tolist()
        raise ValueError(
            f"{path}: classification contains invalid values {invalid_values} for num_classes={num_classes}"
        )

    stats["events"] += classification.shape[0]
    stats["class_min"] = min(stats["class_min"], int(classification.min(initial=0)))
    stats["class_max"] = max(stats["class_max"], int(classification.max(initial=-1)))
    stats["ignored"] += int(np.sum(classification == -1))
    bincount = np.bincount(np.clip(classification, 0, None), minlength=num_classes)
    stats["class_counts"] += bincount


def scan_train_dir(train_dir: Path, contract: dict[str, Any], limit_files: int) -> dict[str, Any]:
    parquet_paths = sorted(train_dir.glob("train_*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No train_*.parquet files found in {train_dir}")
    if limit_files > 0:
        parquet_paths = parquet_paths[:limit_files]

    stats = {
        "events": 0,
        "class_min": sys.maxsize,
        "class_max": -1,
        "ignored": 0,
        "class_counts": np.zeros(len(contract["class_labels"]), dtype=np.int64),
    }

    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096):
            events = ak.from_arrow(batch)
            check_batch(events, path=path, contract=contract, stats=stats)
    return stats


def main() -> None:
    args = parse_args()
    contract = event_info_contract(read_yaml(args.event_info))
    stats = scan_train_dir(args.train_dir, contract, args.limit_files)

    print("[check-training-contract] OK")
    print(f"[check-training-contract] train_dir={args.train_dir}")
    print(f"[check-training-contract] event_info={args.event_info}")
    print(
        "[check-training-contract] dims "
        f"x={contract['raw_sequential_dim']} projected={contract['projected_sequential_dim']} "
        f"conditions={contract['global_dim']} invisible={contract['invisible_dim']} "
        f"classes={len(contract['class_labels'])}"
    )
    print(
        "[check-training-contract] labels "
        f"min={stats['class_min']} max={stats['class_max']} ignored={stats['ignored']} events={stats['events']}"
    )
    for index, (label, count) in enumerate(zip(contract["class_labels"], stats["class_counts"])):
        print(f"[check-training-contract] class[{index:02d}] {label}: {int(count)}")


if __name__ == "__main__":
    main()
