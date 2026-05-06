#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FOUR_VECTOR_FEATURES = ("energy", "pt", "eta", "phi")
DEFAULT_GLOBAL_FIELDS = (
    "Event_totalChargedEnergy",
    "Event_totalEMEnergy",
    "Event_totalHadronicEnergy",
    "thrust_Mag",
    "charged_E",
    "missing_px",
    "missing_py",
    "missing_pt",
    "isolation_angle",
    "thrust_x",
    "thrust_y",
    "thrust_z",
)


@dataclass(frozen=True)
class FeatureConfig:
    raw_sequential_fields: tuple[str, ...]
    global_fields: tuple[str, ...]
    grouped_sequential_config: dict[str, Any] | None


@dataclass(frozen=True)
class EveNetConfig:
    process_topologies: dict[str, dict[str, tuple[str, ...]]]
    generation_conditions: tuple[str, ...]
    generation_global_targets: tuple[str, ...]
    generation_events: tuple[str, ...]
    sequential_tags: dict[str, str]
    global_tags: dict[str, str]
    invisible_features: tuple[str, ...]
    invisible_tags: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the EveNet event-info YAML used by preprocess/train configs "
            "from analysis.yaml and the EveNet schema."
        )
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("ml_pipeline/config/analysis.yaml"),
        help="Analysis YAML with Samples / Inputs / Normalization / Subcategories.",
    )
    parser.add_argument(
        "--evenet-config",
        type=Path,
        default=Path("ml_pipeline/config/evenet_schema.yaml"),
        help="EveNet schema YAML with Processes / Generations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml_pipeline_lite/generated_event_info.yaml"),
        help="Output generated event-info YAML path.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Optional subset of sample keys from analysis.yaml to keep in the class label list.",
    )
    parser.add_argument(
        "--write-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a small JSON summary next to the generated YAML.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def normalize_part_feature_name(raw_name: str) -> str:
    if raw_name in FOUR_VECTOR_FEATURES:
        return f"Part_{raw_name}"
    return raw_name


def default_sequential_tags(raw_sequential_fields: tuple[str, ...]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for field in raw_sequential_fields:
        if field in {"Part_energy", "Part_pt"}:
            tags[field] = "log_normalize"
        elif field == "Part_eta":
            tags[field] = "normalize"
        elif field == "Part_phi":
            tags[field] = "normalize_uniform"
        else:
            tags[field] = "none"
    return tags


def default_global_tags(global_fields: tuple[str, ...]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for field in global_fields:
        if field in {
            "Event_totalChargedEnergy",
            "Event_totalEMEnergy",
            "Event_totalHadronicEnergy",
            "charged_E",
            "missing_pt",
        }:
            tags[field] = "log_normalize"
        elif field in {
            "thrust_Mag",
            "missing_px",
            "missing_py",
            "isolation_angle",
            "thrust_x",
            "thrust_y",
            "thrust_z",
        }:
            tags[field] = "normalize"
        else:
            tags[field] = "none"
    return tags


def default_invisible_tags(invisible_features: tuple[str, ...]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for field in invisible_features:
        if field in {"energy", "pt"}:
            tags[field] = "log_normalize"
        elif field == "phi":
            tags[field] = "normalize_uniform"
        elif field in {"E", "px", "py", "pz", "mass", "eta"}:
            tags[field] = "normalize"
        else:
            tags[field] = "none"
    return tags


def merge_tags(default_tags: dict[str, str], raw_tags: dict[str, Any] | None) -> dict[str, str]:
    merged = dict(default_tags)
    if not raw_tags:
        return merged
    raw_default = raw_tags.get("default")
    if raw_default is not None:
        for key in list(merged):
            merged[key] = raw_default
    for key, value in raw_tags.items():
        if key == "default":
            continue
        merged[key] = str(value)
    return merged


def parse_feature_config(config: dict[str, Any]) -> FeatureConfig:
    inputs_cfg = config.get("Inputs") or {}
    part_cfg = inputs_cfg.get("Part") or {}
    global_cfg = inputs_cfg.get("Global") or {}

    momentum_cfg = part_cfg.get("Momentum", {})
    if isinstance(momentum_cfg, dict):
        momentum_inputs = momentum_cfg.get("input", [])
    else:
        momentum_inputs = momentum_cfg
    momentum_fields = tuple(str(item) for item in momentum_inputs if isinstance(item, str))
    if momentum_fields != FOUR_VECTOR_FEATURES:
        raise ValueError(
            "Inputs.Part.Momentum must expose exactly ['energy', 'pt', 'eta', 'phi'] for ml_pipeline_lite."
        )

    raw_sequential_fields: list[str] = [normalize_part_feature_name(name) for name in momentum_fields]
    for key, value in part_cfg.items():
        if key == "Momentum":
            continue
        if isinstance(value, dict):
            raw_sequential_fields.append(key)
        elif isinstance(value, list):
            raw_sequential_fields.extend(normalize_part_feature_name(str(item)) for item in value)

    deduped_fields: list[str] = []
    seen: set[str] = set()
    for field in raw_sequential_fields:
        if field in seen:
            continue
        seen.add(field)
        deduped_fields.append(field)

    grouped_sequential_config = None
    if any(isinstance(value, dict) for value in part_cfg.values()):
        grouped_sequential_config = {
            "SEQUENTIAL": {
                "Source": {
                    "projected_feature_names": deduped_fields,
                    "groups": {
                        key: value
                        for key, value in part_cfg.items()
                        if isinstance(value, dict)
                    },
                }
            }
        }

    global_fields = tuple(str(item) for item in global_cfg.get("Fields", DEFAULT_GLOBAL_FIELDS))
    return FeatureConfig(
        raw_sequential_fields=tuple(deduped_fields),
        global_fields=global_fields,
        grouped_sequential_config=grouped_sequential_config,
    )


def parse_evenet_config(schema_config: dict[str, Any], analysis_config: dict[str, Any], feature_config: FeatureConfig) -> EveNetConfig:
    normalization_cfg = analysis_config.get("Normalization") or analysis_config.get("EveNet", {}).get("FeatureTags", {})
    generations_cfg = schema_config.get("Generations") or {}
    invisible_cfg = normalization_cfg.get("Invisible") or {}
    invisible_features = tuple(
        key for key in invisible_cfg.keys() if key != "default"
    ) or FOUR_VECTOR_FEATURES

    process_topologies: dict[str, dict[str, tuple[str, ...]]] = {}
    for process_name, resonance_map in (schema_config.get("Processes") or {}).items():
        process_topologies[process_name] = {
            str(resonance_name): tuple(str(item) for item in products)
            for resonance_name, products in resonance_map.items()
        }

    return EveNetConfig(
        process_topologies=process_topologies,
        generation_conditions=tuple(generations_cfg.get("Conditions", feature_config.global_fields)),
        generation_global_targets=tuple(generations_cfg.get("GlobalTargets", ())),
        generation_events=tuple(generations_cfg.get("Events", feature_config.raw_sequential_fields)),
        sequential_tags=merge_tags(default_sequential_tags(feature_config.raw_sequential_fields), normalization_cfg.get("Sequential")),
        global_tags=merge_tags(default_global_tags(feature_config.global_fields), normalization_cfg.get("Global")),
        invisible_features=tuple(str(item) for item in invisible_features),
        invisible_tags=merge_tags(default_invisible_tags(tuple(str(item) for item in invisible_features)), invisible_cfg),
    )


def ordered_class_labels(analysis_config: dict[str, Any], selected_keys: set[str] | None) -> list[str]:
    samples_cfg = analysis_config.get("Samples") or {}
    subcategories_cfg = analysis_config.get("Subcategories") or {}
    labels: list[str] = []

    for sample_key, sample_cfg in samples_cfg.items():
        if selected_keys and sample_key not in selected_keys:
            continue
        if bool(sample_cfg.get("is_data", False)):
            continue

        sample_name = str(sample_cfg.get("name", sample_key))
        sample_subcategories = subcategories_cfg.get(sample_key)
        if sample_subcategories:
            labels.extend(str(label) for label in sample_subcategories.keys())
        else:
            labels.append(sample_name)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduplicated.append(label)
    return deduplicated


def lookup_feature_tag(feature_name: str, tags: dict[str, str], default: str = "none") -> str:
    if feature_name in tags:
        return tags[feature_name]
    stripped = feature_name[5:] if feature_name.startswith("Part_") else feature_name
    return tags.get(stripped, default)


def build_event_info_payload(
    class_labels: list[str],
    feature_config,
    evenet_config,
) -> dict[str, Any]:
    missing_processes = [label for label in class_labels if label not in evenet_config.process_topologies]
    if missing_processes:
        raise ValueError(
            "Missing EveNet process topology definitions for: " + ", ".join(missing_processes)
        )

    classification_name = "signal"
    payload: dict[str, Any] = {
        "INPUTS": {
            "SEQUENTIAL": {
                "Source": {
                    feature_name: lookup_feature_tag(feature_name, evenet_config.sequential_tags)
                    for feature_name in feature_config.raw_sequential_fields
                }
            },
            "GLOBAL": {
                "Conditions": {
                    feature_name: lookup_feature_tag(feature_name, evenet_config.global_tags)
                    for feature_name in feature_config.global_fields
                }
            },
        },
        "EVENT": {
            label: {
                resonance_name: list(products)
                for resonance_name, products in evenet_config.process_topologies[label].items()
            }
            for label in class_labels
        },
        "CLASSIFICATIONS": {
            "EVENT": [classification_name],
        },
        "CLASSLABEL": {
            "EVENT": {
                classification_name: [class_labels],
            }
        },
        "GENERATIONS": {
            "Conditions": list(evenet_config.generation_conditions),
            "GlobalTargets": list(evenet_config.generation_global_targets),
            "Events": list(evenet_config.generation_events),
            "Neutrinos": {
                feature_name: evenet_config.invisible_tags.get(feature_name, "none")
                for feature_name in evenet_config.invisible_features
            },
        },
    }
    if feature_config.grouped_sequential_config is not None:
        payload["GROUPED_INPUTS"] = feature_config.grouped_sequential_config
    return payload


def main() -> None:
    args = parse_args()
    analysis_config = read_yaml(args.analysis_config)
    evenet_schema = read_yaml(args.evenet_config)

    feature_config = parse_feature_config(analysis_config)
    evenet_config = parse_evenet_config(evenet_schema, analysis_config, feature_config)

    selected_keys = set(args.samples or [])
    class_labels = ordered_class_labels(analysis_config, selected_keys if selected_keys else None)
    if not class_labels:
        raise ValueError("No MC class labels were derived from analysis.yaml.")

    payload = build_event_info_payload(class_labels, feature_config, evenet_config)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    print(f"[ml_pipeline_lite] wrote {output_path}")

    if args.write_summary:
        summary = {
            "analysis_config": str(args.analysis_config),
            "evenet_config": str(args.evenet_config),
            "class_labels": class_labels,
            "point_cloud_features": list(feature_config.raw_sequential_fields),
            "global_features": list(feature_config.global_fields),
            "invisible_features": list(evenet_config.invisible_features),
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[ml_pipeline_lite] wrote {summary_path}")


if __name__ == "__main__":
    main()
