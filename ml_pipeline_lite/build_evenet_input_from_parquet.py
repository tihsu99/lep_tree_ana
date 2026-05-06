#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import vector
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import build_observables
from ml_pipeline_lite.common import (
    build_classification_lookup,
    classification_targets_for_sample,
    process_latex_label,
)
from ml_pipeline_lite.generate_event_info_yaml import parse_feature_config


vector.register_awkward()

MAX_VISIBLE_ENERGY_GEV = 91.25
PART_MOMENTUM_SOURCE_FIELDS = (
    "Part_fourMomentum_fCoordinates_fX",
    "Part_fourMomentum_fCoordinates_fY",
    "Part_fourMomentum_fCoordinates_fZ",
    "Part_fourMomentum_fCoordinates_fT",
)
DEFAULT_INVISIBLE_FEATURES = ("pt", "eta", "phi")
CORE_MODEL_KEYS = {
    "x",
    "x_mask",
    "conditions",
    "conditions_mask",
    "x_invisible",
    "x_invisible_mask",
}
COMPARISON_1D_KEYS = (
    "nprong",
    "visible_energy",
    "visible_pt",
    "visible_eta",
    "visible_phi",
)

MONITOR_1D_SPECS: dict[str, tuple[float, float, int]] = {
    "nprong": (-0.5, 7.5, 8),
    "visible_energy": (0.0, 100.0, 80),
    "visible_pt": (0.0, 60.0, 80),
    "visible_eta": (-3.0, 3.0, 80),
    "visible_phi": (-math.pi, math.pi, 80),
    "target_pt": (0.0, 60.0, 80),
    "target_eta": (-3.0, 3.0, 80),
    "target_phi": (-math.pi, math.pi, 80),
    "truth_theta_cm": (0.0, 1.0, 80),
    "truth_mtautau": (0.0, 150.0, 80),
    "truth_cos_theta_A_k": (-1.0, 1.0, 80),
    "truth_cos_theta_B_k": (-1.0, 1.0, 80),
    "truth_cos_theta_A_k_times_cos_theta_B_k": (-1.0, 1.0, 80),
}

MONITOR_2D_SPECS: dict[str, tuple[tuple[float, float], tuple[float, float], int]] = {
    "truth_vs_rebuilt_theta_cm": ((0.0, 1.0), (0.0, 1.0), 60),
    "truth_vs_rebuilt_mtautau": ((0.0, 150.0), (0.0, 150.0), 60),
    "truth_vs_rebuilt_cos_theta_A_k": ((-1.0, 1.0), (-1.0, 1.0), 60),
    "truth_vs_rebuilt_cos_theta_B_k": ((-1.0, 1.0), (-1.0, 1.0), 60),
    "truth_vs_rebuilt_cos_theta_A_k_times_cos_theta_B_k": ((-1.0, 1.0), (-1.0, 1.0), 60),
}


@dataclass(frozen=True)
class Sample:
    key: str
    name: str
    is_data: bool
    is_signal: bool
    files: tuple[str, ...]
    file_source: str
    norm_factor: float | None
    lumi: float | None
    total_initial_num_events: float | None
    plot_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build lightweight EveNet parquet shards from central parquet inputs. "
            "This rewrite keeps only the fields needed downstream."
        )
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("ml_pipeline/config/analysis.yaml"),
        help="Analysis YAML with Samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for lite parquet shards and monitoring.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Optional subset of sample keys from analysis.yaml.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Rows per parquet record-batch read.",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=100000,
        help="Selected rows per output shard.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers. Each worker processes one input parquet file.",
    )
    parser.add_argument(
        "--monitoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write merged monitoring plots from worker histogram payloads.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def expand_files(patterns: tuple[str, ...]) -> list[str]:
    output: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            output.extend(matched)
        else:
            output.append(pattern)
    return output


def parse_samples(config: dict[str, Any], selected_keys: list[str] | None) -> list[Sample]:
    selected = set(selected_keys or [])
    samples: list[Sample] = []
    for key, sample_cfg in (config.get("Samples") or {}).items():
        if selected and key not in selected:
            continue
        input_files = sample_cfg.get("input_files") or []
        raw_files = sample_cfg.get("raw_files") or []
        if input_files:
            file_list = input_files
            file_source = "input_files"
        else:
            file_list = raw_files
            file_source = "raw_files"
        samples.append(
            Sample(
                key=key,
                name=str(sample_cfg.get("name", key)),
                is_data=bool(sample_cfg.get("is_data", False)),
                is_signal=bool(sample_cfg.get("is_signal", False)),
                files=tuple(str(item) for item in file_list),
                file_source=file_source,
                norm_factor=float(sample_cfg["norm_factor"]) if "norm_factor" in sample_cfg else None,
                lumi=float(sample_cfg["lumi"]) if "lumi" in sample_cfg else None,
                total_initial_num_events=None,
                plot_label=process_latex_label(str(sample_cfg.get("name", key))),
            )
        )
    return samples


def read_file_initial_total_num_events(path: str) -> float | None:
    parquet = pq.ParquetFile(path)
    schema_names = {field.name for field in parquet.schema_arrow}
    if "initial_total_num_events" not in schema_names:
        return None
    for record_batch in parquet.iter_batches(batch_size=1, columns=["initial_total_num_events"]):
        values = ak.to_numpy(ak.from_arrow(record_batch)["initial_total_num_events"], allow_missing=False)
        if len(values) == 0:
            continue
        return float(values[0])
    return None


def attach_sample_total_initial_events(samples: list[Sample]) -> list[Sample]:
    resolved_samples: list[Sample] = []
    for sample in samples:
        if sample.is_data:
            resolved_samples.append(replace(sample))
            continue
        total = 0.0
        for path in expand_files(sample.files):
            file_total = read_file_initial_total_num_events(path)
            if file_total is None:
                raise ValueError(
                    f"Sample '{sample.key}' file '{path}' is missing initial_total_num_events."
                )
            total += float(file_total)
        if total <= 0.0:
            raise ValueError(f"Sample '{sample.key}' has non-positive total_initial_num_events={total}.")
        resolved_samples.append(replace(sample, total_initial_num_events=total))
    return resolved_samples


def infer_luminosity(samples: list[Sample]) -> float | None:
    data_lumis = [sample.lumi for sample in samples if sample.is_data and sample.lumi is not None]
    return sum(data_lumis) if data_lumis else None


def infer_remove_neutral_non_photon(config: dict[str, Any]) -> bool:
    return bool((config.get("EveNetInput") or {}).get("remove_neutral_non_photon", False))


def infer_invisible_features(config: dict[str, Any]) -> tuple[str, ...]:
    invisible_cfg = ((config.get("Normalization") or {}).get("Invisible") or {})
    features = tuple(str(key) for key in invisible_cfg if key != "default")
    return features or DEFAULT_INVISIBLE_FEATURES


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(getattr(values, "fields", []))
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return vector.zip({"px": values["x"], "py": values["y"], "pz": values["z"], "E": values["t"]})
    raise ValueError(f"Unsupported four-vector fields: {sorted(fields)}")


def cast_array_like(values: Any, dtype=np.float64):
    if isinstance(values, ak.Array):
        return ak.values_astype(values, dtype)
    return np.asarray(values, dtype=dtype)


def build_momentum4d(px: Any, py: Any, pz: Any, energy: Any) -> ak.Array:
    return ak.zip(
        {
            "px": cast_array_like(px, np.float64),
            "py": cast_array_like(py, np.float64),
            "pz": cast_array_like(pz, np.float64),
            "E": cast_array_like(energy, np.float64),
        },
        with_name="Momentum4D",
    )


def materialize_p4(values: ak.Array) -> ak.Array:
    vector_values = rebuild_vector(values)
    return ak.zip(
        {
            "x": vector_values.px,
            "y": vector_values.py,
            "z": vector_values.pz,
            "t": vector_values.E,
        }
    )


def finite_p4_mask(values: ak.Array) -> np.ndarray:
    return (
        np.isfinite(ak.to_numpy(values.px, allow_missing=False))
        & np.isfinite(ak.to_numpy(values.py, allow_missing=False))
        & np.isfinite(ak.to_numpy(values.pz, allow_missing=False))
        & np.isfinite(ak.to_numpy(values.E, allow_missing=False))
    )


def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values, allow_missing=False).astype(dtype)


def to_numpy_array(values: Any, dtype=None) -> np.ndarray:
    if isinstance(values, ak.Array):
        values = ak.to_numpy(values, allow_missing=False)
    else:
        values = np.asarray(values)
    if dtype is not None:
        values = values.astype(dtype)
    return values


def part_input_mask(events: ak.Array) -> ak.Array:
    if "Part_fourMomentum_fCoordinates_fT" not in events.fields:
        reference_field = next(field for field in events.fields if field.startswith("Part_"))
        return ak.ones_like(events[reference_field], dtype=bool)
    energy = events["Part_fourMomentum_fCoordinates_fT"]
    return ak.values_astype(np.isfinite(energy) & (energy <= MAX_VISIBLE_ENERGY_GEV), bool)


def build_input_particle_mask(events: ak.Array, remove_neutral_non_photon: bool) -> ak.Array:
    mask = part_input_mask(events)
    if not remove_neutral_non_photon:
        return mask
    missing_fields = [field for field in ("Part_charge", "Part_pdgId") if field not in events.fields]
    if missing_fields:
        raise ValueError(
            "Cannot remove neutral non-photon particles because required fields are missing: "
            f"{missing_fields}."
        )
    charge = events["Part_charge"]
    abs_pdg_id = abs(events["Part_pdgId"])
    keep_particle = (charge != 0) | (abs_pdg_id == 21)
    return mask & ak.values_astype(keep_particle, bool)


def filtered_part_momentum(events: ak.Array, input_part_mask: ak.Array | None = None) -> ak.Array:
    mask = input_part_mask if input_part_mask is not None else part_input_mask(events)
    return build_momentum4d(
        events["Part_fourMomentum_fCoordinates_fX"][mask],
        events["Part_fourMomentum_fCoordinates_fY"][mask],
        events["Part_fourMomentum_fCoordinates_fZ"][mask],
        events["Part_fourMomentum_fCoordinates_fT"][mask],
    )


def build_part_inputs(events: ak.Array, feature_config, input_part_mask: ak.Array) -> tuple[dict[str, ak.Array], np.ndarray]:
    part_p4 = filtered_part_momentum(events, input_part_mask=input_part_mask)
    momentum_lookup = {
        "Part_energy": part_p4.E,
        "Part_pt": part_p4.pt,
        "Part_eta": ak.where(np.isfinite(part_p4.eta), part_p4.eta, 0.0),
        "Part_phi": part_p4.phi,
    }
    fields: dict[str, ak.Array] = {}

    for feature_name in feature_config.all_sequential_fields:
        if feature_name in momentum_lookup:
            fields[feature_name] = momentum_lookup[feature_name]
        else:
            fields[feature_name] = events[feature_name][input_part_mask]
    num_vectors = ak.to_numpy(ak.num(next(iter(fields.values())), axis=1), allow_missing=False).astype(np.int32)
    return fields, num_vectors


def pad_and_flatten_part_feature(values: ak.Array, max_particles: int) -> ak.Array:
    padded = ak.pad_none(values, max_particles, axis=1, clip=True)
    filled = ak.fill_none(padded, 0)
    regular = ak.to_regular(filled, axis=1)
    return ak.values_astype(regular, np.float32)[..., np.newaxis]


def flatten_global_feature(values: ak.Array) -> ak.Array:
    filled = ak.fill_none(values, 0)
    return ak.values_astype(filled, np.float32)[..., np.newaxis]


def build_point_cloud(
    events: ak.Array,
    max_particles: int,
    feature_config,
    remove_neutral_non_photon: bool,
) -> tuple[ak.Array, ak.Array, np.ndarray, ak.Array, list[str]]:
    input_part_mask = build_input_particle_mask(events, remove_neutral_non_photon)
    part_p4 = filtered_part_momentum(events, input_part_mask=input_part_mask)
    eta = ak.where(np.isfinite(part_p4.eta), part_p4.eta, 0)

    available_momentum_features = {
        "Part_energy": part_p4.E,
        "Part_pt": part_p4.pt,
        "Part_eta": eta,
        "Part_phi": part_p4.phi,
    }
    features = []
    feature_names: list[str] = []
    for feature_name in feature_config.raw_sequential_fields:
        if feature_name in available_momentum_features:
            values = available_momentum_features[feature_name]
        else:
            values = events[feature_name][input_part_mask]
        features.append(pad_and_flatten_part_feature(values, max_particles))
        feature_names.append(feature_name)
    x = ak.concatenate(features, axis=2)
    num_particles = ak.to_numpy(ak.num(events["Part_pdgId"][input_part_mask], axis=1), allow_missing=False).astype(np.float32)
    x_mask = ak.Array(np.arange(max_particles)[None, :] < num_particles[:, None])
    return x, x_mask, num_particles, input_part_mask, feature_names


def _p4_component(p4: ak.Array, component: str) -> ak.Array:
    fields = set(getattr(p4, "fields", []))
    if component == "px":
        return p4["px"] if "px" in fields else p4["x"]
    if component == "py":
        return p4["py"] if "py" in fields else p4["y"]
    if component == "pz":
        return p4["pz"] if "pz" in fields else p4["z"]
    if component in {"E", "energy"}:
        return p4["E"] if "E" in fields else p4["t"]
    raise ValueError(f"Unsupported p4 component '{component}'.")


def resolve_global_feature(events: ak.Array, field_name: str) -> ak.Array:
    if field_name in events.fields:
        return events[field_name]
    if "missing_p4" in events.fields:
        missing_p4 = events["missing_p4"]
        if field_name == "missing_px":
            return _p4_component(missing_p4, "px")
        if field_name == "missing_py":
            return _p4_component(missing_p4, "py")
        if field_name == "missing_pz":
            return _p4_component(missing_p4, "pz")
        if field_name in {"missing_E", "missing_energy"}:
            return _p4_component(missing_p4, "E")
        if field_name == "missing_pt":
            px = _p4_component(missing_p4, "px")
            py = _p4_component(missing_p4, "py")
            return np.sqrt(px * px + py * py)
    preview = ", ".join(events.fields[:25])
    suffix = " ..." if len(events.fields) > 25 else ""
    raise KeyError(f"Global feature '{field_name}' is missing. Available fields include: {preview}{suffix}")


def build_global_inputs(events: ak.Array, feature_config) -> dict[str, np.ndarray]:
    return {
        field_name: to_numpy(resolve_global_feature(events, field_name), np.float32)
        for field_name in feature_config.global_fields
    }


def build_global_conditions(events: ak.Array, feature_config) -> tuple[ak.Array, ak.Array, list[str]]:
    features = []
    feature_names: list[str] = []
    for field_name in feature_config.global_fields:
        features.append(flatten_global_feature(resolve_global_feature(events, field_name)))
        feature_names.append(field_name)
    conditions = ak.concatenate(features, axis=1)
    conditions_mask = ak.Array(np.ones((len(events), 1), dtype=bool))
    return conditions, conditions_mask, feature_names


def compute_event_totals(num_sequential_vectors: np.ndarray, conditions_mask: ak.Array) -> np.ndarray:
    return num_sequential_vectors + ak.to_numpy(ak.values_astype(conditions_mask[:, 0], np.float32), allow_missing=False)


def source_event_key_array(events: ak.Array, fallback_index: np.ndarray) -> np.ndarray:
    for key in ("evtNumber", "Event_evtNumber"):
        if key in events.fields:
            return to_numpy_array(events[key], np.int64)
    return fallback_index.astype(np.int64, copy=False)


def validate_output_contract(
    output_events: ak.Array,
    feature_config,
    invisible_features: tuple[str, ...],
    max_particles: int,
) -> None:
    missing = sorted(CORE_MODEL_KEYS - set(output_events.fields))
    if missing:
        raise ValueError(f"Lite builder output is missing core model keys: {missing}")

    x = to_numpy_array(output_events["x"], np.float32)
    x_mask = to_numpy_array(output_events["x_mask"], bool)
    conditions = to_numpy_array(output_events["conditions"], np.float32)
    conditions_mask = to_numpy_array(output_events["conditions_mask"], bool)
    x_invisible = to_numpy_array(output_events["x_invisible"], np.float32)
    x_invisible_mask = to_numpy_array(output_events["x_invisible_mask"], bool)

    expected_x_shape = (len(output_events), max_particles, len(feature_config.raw_sequential_fields))
    if x.shape != expected_x_shape:
        raise ValueError(f"Output x shape mismatch: got {x.shape}, expected {expected_x_shape}")
    if x_mask.shape != (len(output_events), max_particles):
        raise ValueError(f"Output x_mask shape mismatch: got {x_mask.shape}")
    if conditions.shape != (len(output_events), len(feature_config.global_fields)):
        raise ValueError(f"Output conditions shape mismatch: got {conditions.shape}")
    if conditions_mask.shape != (len(output_events), 1):
        raise ValueError(f"Output conditions_mask shape mismatch: got {conditions_mask.shape}")
    if x_invisible.shape != (len(output_events), 2, len(invisible_features)):
        raise ValueError(f"Output x_invisible shape mismatch: got {x_invisible.shape}")
    if x_invisible_mask.shape != (len(output_events), 2):
        raise ValueError(f"Output x_invisible_mask shape mismatch: got {x_invisible_mask.shape}")


def required_columns(schema_names: set[str], feature_config, sample: Sample) -> list[str]:
    columns = {
        "lead_a_visible_p4",
        "lead_b_visible_p4",
        "nprong",
        "initial_total_num_events",
        "weight",
        "central_weight",
        "Part_pdgId",
    }
    if "evtNumber" in schema_names:
        columns.add("evtNumber")
    if "Event_evtNumber" in schema_names:
        columns.add("Event_evtNumber")
    if sample.is_signal:
        columns.update(
            {
                "truth_tau_a_p4",
                "truth_tau_b_p4",
                "event_category",
                "truth_QI_region",
                "analyzing_power",
                "analyzing_power_a",
                "analyzing_power_b",
            }
        )
    for feature_name in feature_config.all_sequential_fields:
        if feature_name in {"Part_energy", "Part_pt", "Part_eta", "Part_phi"}:
            columns.update(PART_MOMENTUM_SOURCE_FIELDS)
        else:
            columns.add(feature_name)
    for field_name in feature_config.global_fields:
        if field_name in schema_names:
            columns.add(field_name)
        elif field_name.startswith("missing_") or field_name in {"missing_pt", "missing_E", "missing_energy"}:
            columns.add("missing_p4")
    columns.update(name for name in schema_names if name.endswith("_cut"))
    return sorted(name for name in columns if name in schema_names)


def ensure_required_fields(events: ak.Array, sample: Sample, path: str, feature_config) -> None:
    required = {"lead_a_visible_p4", "lead_b_visible_p4"}
    if sample.is_signal:
        required.update({"truth_tau_a_p4", "truth_tau_b_p4", "event_category"})
    missing = sorted(required - set(events.fields))
    if missing:
        raise ValueError(
            f"Sample '{sample.key}' file '{path}' is missing required fields: {missing}"
        )
    if any(feature_name in {"Part_energy", "Part_pt", "Part_eta", "Part_phi"} for feature_name in feature_config.all_sequential_fields):
        missing_momentum = [field for field in PART_MOMENTUM_SOURCE_FIELDS if field not in events.fields]
        if missing_momentum:
            raise ValueError(
                f"Sample '{sample.key}' file '{path}' is missing particle momentum fields: {missing_momentum}"
            )
    missing_part_features = [
        feature_name
        for feature_name in feature_config.all_sequential_fields
        if feature_name not in {"Part_energy", "Part_pt", "Part_eta", "Part_phi"} and feature_name not in events.fields
    ]
    if missing_part_features:
        raise ValueError(
            f"Sample '{sample.key}' file '{path}' is missing particle input fields: {missing_part_features}"
        )
    if any(field_name not in events.fields for field_name in feature_config.global_fields):
        missing_global = []
        for field_name in feature_config.global_fields:
            if field_name in events.fields:
                continue
            if (field_name.startswith("missing_") or field_name in {"missing_pt", "missing_E", "missing_energy"}) and "missing_p4" in events.fields:
                continue
            missing_global.append(field_name)
        if missing_global:
            raise ValueError(
                f"Sample '{sample.key}' file '{path}' is missing required global fields: {missing_global}"
            )


def preselection_mask(events: ak.Array) -> np.ndarray:
    mask = np.ones(len(events), dtype=bool)
    mask &= to_numpy(events["nprong"], np.int64) == 2 # 2 prong legs event only
    visible_a = rebuild_vector(events["lead_a_visible_p4"])
    visible_b = rebuild_vector(events["lead_b_visible_p4"])
    mask &= to_numpy(visible_a.E) < MAX_VISIBLE_ENERGY_GEV
    mask &= to_numpy(visible_b.E) < MAX_VISIBLE_ENERGY_GEV
    return mask


def nominal_event_weight(sample: Sample, luminosity: float | None, events: ak.Array) -> np.ndarray:
    if sample.is_data:
        return np.ones(len(events), dtype=np.float32)
    if luminosity is not None and sample.norm_factor is not None and sample.total_initial_num_events is not None:
        scale = np.float32(luminosity * sample.norm_factor / sample.total_initial_num_events)
        return np.full(len(events), scale, dtype=np.float32)
    if "central_weight" in events.fields:
        return to_numpy(events["central_weight"], np.float32)
    raise ValueError(
        f"Sample '{sample.key}' needs (luminosity, norm_factor, total_initial_num_events) or central_weight."
    )


def build_target_missing(truth_tau: ak.Array, visible_tau: ak.Array) -> tuple[ak.Array, np.ndarray]:
    target = truth_tau - visible_tau
    valid = finite_p4_mask(truth_tau) & finite_p4_mask(visible_tau)
    valid &= np.isfinite(to_numpy(target.pt)) & np.isfinite(to_numpy(target.eta)) & np.isfinite(to_numpy(target.phi))
    return target, valid


def build_truth_observables(truth_tau_a: ak.Array, truth_tau_b: ak.Array, visible_a: ak.Array, visible_b: ak.Array) -> dict[str, np.ndarray]:
    observables = build_observables(
        tau_a_p4=truth_tau_a,
        tau_b_p4=truth_tau_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )
    return {f"truth_{name}": np.asarray(values, dtype=np.float32) for name, values in observables.items()}


def stack_tau_pair(values_a: ak.Array, values_b: ak.Array) -> ak.Array:
    return ak.concatenate([values_a[:, np.newaxis], values_b[:, np.newaxis]], axis=1)


def stack_tau_pair_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> ak.Array:
    return ak.Array(np.stack([mask_a, mask_b], axis=1))


def zero_p4(num_events: int, num_slots: int = 2) -> ak.Array:
    zeros = np.zeros((num_events, num_slots), dtype=np.float32)
    return build_momentum4d(zeros, zeros, zeros, zeros)


def features_from_p4_local(p4: ak.Array, feature_names: tuple[str, ...]) -> np.ndarray:
    components: list[np.ndarray] = []
    for feature_name in feature_names:
        if feature_name in {"energy", "E"}:
            values = ak.to_numpy(p4.E, allow_missing=False)
        elif feature_name == "px":
            values = ak.to_numpy(p4.px, allow_missing=False)
        elif feature_name == "py":
            values = ak.to_numpy(p4.py, allow_missing=False)
        elif feature_name == "pz":
            values = ak.to_numpy(p4.pz, allow_missing=False)
        elif feature_name == "pt":
            values = ak.to_numpy(p4.pt, allow_missing=False)
        elif feature_name == "eta":
            values = ak.to_numpy(ak.where(np.isfinite(p4.eta), p4.eta, 0.0), allow_missing=False)
        elif feature_name == "phi":
            values = ak.to_numpy(ak.where(np.isfinite(p4.phi), p4.phi, 0.0), allow_missing=False)
        elif feature_name == "mass":
            values = ak.to_numpy(ak.where(np.isfinite(p4.mass), p4.mass, 0.0), allow_missing=False)
        else:
            raise ValueError(f"Unsupported four-vector feature '{feature_name}'.")
        components.append(values.astype(np.float32))
    return np.stack(components, axis=-1).astype(np.float32)


def build_visible_tau_targets(selected_events: ak.Array) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array]:
    visible_a = rebuild_vector(selected_events["lead_a_visible_p4"])
    visible_b = rebuild_vector(selected_events["lead_b_visible_p4"])
    visible_pair = stack_tau_pair(visible_a, visible_b)
    visible_mask = stack_tau_pair_mask(finite_p4_mask(visible_a), finite_p4_mask(visible_b))
    return visible_pair, visible_mask, visible_pair, visible_mask


def build_invisible_targets_simple(
    sample: Sample,
    selected_events: ak.Array,
    visible_pair: ak.Array,
    visible_mask: ak.Array,
) -> tuple[ak.Array, ak.Array, np.ndarray, np.ndarray, ak.Array, ak.Array]:
    num_events = len(selected_events)
    tau_vis_target_p4 = visible_pair
    tau_vis_target_mask = visible_mask
    if not sample.is_signal:
        return (
            zero_p4(num_events),
            ak.Array(np.zeros((num_events, 2), dtype=bool)),
            np.zeros(num_events, dtype=np.int64),
            np.zeros(num_events, dtype=np.int64),
            tau_vis_target_p4,
            tau_vis_target_mask,
        )

    truth_tau_a = rebuild_vector(selected_events["truth_tau_a_p4"])
    truth_tau_b = rebuild_vector(selected_events["truth_tau_b_p4"])
    truth_pair = stack_tau_pair(truth_tau_a, truth_tau_b)
    truth_mask = stack_tau_pair_mask(finite_p4_mask(truth_tau_a), finite_p4_mask(truth_tau_b))
    x_invisible_p4 = truth_pair - tau_vis_target_p4
    x_invisible_mask = ak.Array(
        np.logical_and(
            ak.to_numpy(truth_mask, allow_missing=False),
            ak.to_numpy(tau_vis_target_mask, allow_missing=False),
        )
    )
    valid_counts = np.sum(ak.to_numpy(x_invisible_mask, allow_missing=False).astype(np.int64), axis=1).astype(np.int64)
    return (
        x_invisible_p4,
        x_invisible_mask,
        np.full(num_events, 2, dtype=np.int64),
        valid_counts,
        tau_vis_target_p4,
        tau_vis_target_mask,
    )


def histogram1d(
    values: np.ndarray,
    spec: tuple[float, float, int],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    low, high, bins = spec
    return np.histogram(values, bins=bins, range=(low, high), weights=weights)[0].astype(np.float64)


def histogram2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    spec: tuple[tuple[float, float], tuple[float, float], int],
) -> np.ndarray:
    (x_low, x_high), (y_low, y_high), bins = spec
    return np.histogram2d(
        x_values,
        y_values,
        bins=bins,
        range=((x_low, x_high), (y_low, y_high)),
    )[0].astype(np.float64)


def empty_monitor_state() -> dict[str, Any]:
    return {
        "counts_1d": {name: np.zeros(spec[2], dtype=np.float64) for name, spec in MONITOR_1D_SPECS.items()},
        "counts_2d": {
            name: np.zeros((spec[2], spec[2]), dtype=np.float64)
            for name, spec in MONITOR_2D_SPECS.items()
        },
        "rows_seen": 0,
        "rows_selected": 0,
        "sum_event_weight": 0.0,
    }


def fill_monitor_state(state: dict[str, Any], selected_events: ak.Array, output_events: ak.Array) -> None:
    state["rows_selected"] += len(selected_events)
    if len(selected_events) == 0:
        return

    event_weight = to_numpy(output_events["event_weight"], np.float64)
    state["sum_event_weight"] += float(np.sum(event_weight))
    visible_a = rebuild_vector(selected_events["lead_a_visible_p4"])
    visible_b = rebuild_vector(selected_events["lead_b_visible_p4"])
    target_pt = np.concatenate([to_numpy(output_events["target_invisible_slot0_pt"]), to_numpy(output_events["target_invisible_slot1_pt"])])
    target_eta = np.concatenate([to_numpy(output_events["target_invisible_slot0_eta"]), to_numpy(output_events["target_invisible_slot1_eta"])])
    target_phi = np.concatenate([to_numpy(output_events["target_invisible_slot0_phi"]), to_numpy(output_events["target_invisible_slot1_phi"])])
    doubled_event_weight = np.concatenate([event_weight, event_weight])
    visible_energy = np.concatenate([to_numpy(visible_a.E), to_numpy(visible_b.E)])
    visible_pt = np.concatenate([to_numpy(visible_a.pt), to_numpy(visible_b.pt)])
    visible_eta = np.concatenate([to_numpy(visible_a.eta), to_numpy(visible_b.eta)])
    visible_phi = np.concatenate([to_numpy(visible_a.phi), to_numpy(visible_b.phi)])

    values_1d = {
        "visible_energy": visible_energy,
        "visible_pt": visible_pt,
        "visible_eta": visible_eta,
        "visible_phi": visible_phi,
    }
    if "target_invisible_slot0_pt" in output_events.fields:
        values_1d.update(
            {
                "target_pt": target_pt,
                "target_eta": target_eta,
                "target_phi": target_phi,
            }
        )
    if "truth_theta_cm" in output_events.fields:
        values_1d.update(
            {
                "truth_theta_cm": to_numpy(output_events["truth_theta_cm"]),
                "truth_mtautau": to_numpy(output_events["truth_mtautau"]),
                "truth_cos_theta_A_k": to_numpy(output_events["truth_cos_theta_A_k"]),
                "truth_cos_theta_B_k": to_numpy(output_events["truth_cos_theta_B_k"]),
                "truth_cos_theta_A_k_times_cos_theta_B_k": to_numpy(output_events["truth_cos_theta_A_k_times_cos_theta_B_k"]),
            }
        )
    if "nprong" in selected_events.fields:
        values_1d["nprong"] = to_numpy(selected_events["nprong"], np.float64)

    for name, values in values_1d.items():
        finite = np.isfinite(values)
        if np.any(finite):
            weights = event_weight if len(values) == len(event_weight) else doubled_event_weight
            state["counts_1d"][name] += histogram1d(values[finite], MONITOR_1D_SPECS[name], weights=weights[finite])

    if {"truth_tau_a_p4", "truth_tau_b_p4", "truth_theta_cm"}.issubset(set(output_events.fields)):
        rebuilt = build_observables(
            tau_a_p4=rebuild_vector(output_events["truth_tau_a_p4"]),
            tau_b_p4=rebuild_vector(output_events["truth_tau_b_p4"]),
            vis_a_p4=rebuild_vector(output_events["lead_a_visible_p4"]),
            vis_b_p4=rebuild_vector(output_events["lead_b_visible_p4"]),
        )
        rebuilt_map = {f"truth_{name}": np.asarray(values, dtype=np.float64) for name, values in rebuilt.items()}
        for short_name in (
            "theta_cm",
            "mtautau",
            "cos_theta_A_k",
            "cos_theta_B_k",
            "cos_theta_A_k_times_cos_theta_B_k",
        ):
            stored_name = f"truth_{short_name}"
            key = f"truth_vs_rebuilt_{short_name}"
            stored = to_numpy(output_events[stored_name], np.float64)
            reco = rebuilt_map[stored_name]
            finite = np.isfinite(stored) & np.isfinite(reco)
            if np.any(finite):
                state["counts_2d"][key] += histogram2d(stored[finite], reco[finite], MONITOR_2D_SPECS[key])


def merge_monitor_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    merged = empty_monitor_state()
    for state in states:
        merged["rows_seen"] += int(state["rows_seen"])
        merged["rows_selected"] += int(state["rows_selected"])
        merged["sum_event_weight"] += float(state["sum_event_weight"])
        for name in merged["counts_1d"]:
            merged["counts_1d"][name] += state["counts_1d"][name]
        for name in merged["counts_2d"]:
            merged["counts_2d"][name] += state["counts_2d"][name]
    return merged


def write_histogram_plots(output_dir: Path, sample: Sample, state: dict[str, Any]) -> dict[str, str]:
    monitor_dir = output_dir / "monitoring" / sample.key
    monitor_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}

    for name, spec in MONITOR_1D_SPECS.items():
        counts = state["counts_1d"][name]
        low, high, bins = spec
        edges = np.linspace(low, high, bins + 1)
        fig, axis = plt.subplots(figsize=(6.2, 4.6), dpi=160)
        axis.step(edges[:-1], counts, where="post", linewidth=1.7)
        axis.set_ylabel("Weighted yield")
        axis.set_title(f"{sample.plot_label}: {name}")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        plot_path = monitor_dir / f"{name}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        output_paths[name] = str(plot_path.relative_to(output_dir))

    for name, spec in MONITOR_2D_SPECS.items():
        counts = state["counts_2d"][name]
        (x_low, x_high), (y_low, y_high), bins = spec
        x_edges = np.linspace(x_low, x_high, bins + 1)
        y_edges = np.linspace(y_low, y_high, bins + 1)
        fig, axis = plt.subplots(figsize=(5.8, 5.1), dpi=160)
        mesh = axis.pcolormesh(x_edges, y_edges, counts.T, cmap="Blues", shading="auto")
        fig.colorbar(mesh, ax=axis, label="Entries")
        axis.plot([x_low, x_high], [y_low, y_high], color="black", linestyle="--", linewidth=1.0)
        axis.set_xlabel(name.replace("truth_vs_rebuilt_", "stored "))
        axis.set_ylabel(name.replace("truth_vs_rebuilt_", "rebuilt "))
        axis.set_title(f"{sample.plot_label}: {name}")
        axis.grid(alpha=0.16)
        fig.tight_layout()
        plot_path = monitor_dir / f"{name}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        output_paths[name] = str(plot_path.relative_to(output_dir))

    summary_path = monitor_dir / "summary.json"
    summary_payload = {
        "rows_seen": int(state["rows_seen"]),
        "rows_selected": int(state["rows_selected"]),
        "sum_event_weight": float(state["sum_event_weight"]),
        "selection_fraction": float(state["rows_selected"] / state["rows_seen"]) if state["rows_seen"] else 0.0,
        "plots": output_paths,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    output_paths["summary"] = str(summary_path.relative_to(output_dir))
    return output_paths


def write_data_mc_comparison_plots(
    output_dir: Path,
    samples: list[Sample],
    merged_states: dict[str, dict[str, Any]],
) -> dict[str, str]:
    monitor_dir = output_dir / "monitoring" / "comparison"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    data_samples = [sample for sample in samples if sample.is_data]
    mc_samples = [sample for sample in samples if not sample.is_data]
    if not data_samples or not mc_samples:
        return output_paths

    for name in COMPARISON_1D_KEYS:
        spec = MONITOR_1D_SPECS[name]
        low, high, bins = spec
        edges = np.linspace(low, high, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig, (axis, ratio_axis) = plt.subplots(
            2,
            1,
            figsize=(7.0, 6.0),
            dpi=160,
            sharex=True,
            gridspec_kw={"height_ratios": [3.5, 1.1], "hspace": 0.05},
        )
        stack_base = np.zeros(bins, dtype=np.float64)

        for sample in mc_samples:
            counts = merged_states[sample.key]["counts_1d"][name]
            axis.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                bottom=stack_base,
                align="edge",
                alpha=0.8,
                label=sample.plot_label,
                linewidth=0.3,
                edgecolor="black",
            )
            stack_base += counts

        data_counts = np.zeros(bins, dtype=np.float64)
        for sample in data_samples:
            data_counts += merged_states[sample.key]["counts_1d"][name]
        data_error = np.sqrt(np.clip(data_counts, 0.0, None))
        axis.errorbar(
            centers,
            data_counts,
            yerr=data_error,
            fmt="o",
            color="black",
            markersize=3.8,
            linewidth=1.0,
            label="data",
        )

        axis.set_ylabel("Weighted yield")
        axis.set_title(f"Data vs stacked MC: {name}")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)

        ratio = np.divide(
            data_counts,
            stack_base,
            out=np.full_like(data_counts, np.nan),
            where=stack_base > 0.0,
        )
        data_ratio_error = np.divide(
            data_error,
            stack_base,
            out=np.zeros_like(data_error),
            where=stack_base > 0.0,
        )
        ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ratio_axis.errorbar(
            centers,
            ratio,
            yerr=data_ratio_error,
            fmt="o",
            color="black",
            markersize=3.2,
            linewidth=1.0,
        )
        ratio_axis.set_xlabel(name)
        ratio_axis.set_ylabel("Data/MC")
        ratio_axis.set_ylim(0.0, 2.0)
        ratio_axis.grid(alpha=0.2)
        fig.tight_layout()
        plot_path = monitor_dir / f"{name}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        output_paths[name] = str(plot_path.relative_to(output_dir))

    summary_path = monitor_dir / "summary.json"
    summary_path.write_text(json.dumps({"plots": output_paths}, indent=2) + "\n")
    output_paths["summary"] = str(summary_path.relative_to(output_dir))
    return output_paths


def build_output_events(
    selected_events: ak.Array,
    sample: Sample,
    luminosity: float | None,
    classification_lookup,
    feature_config,
    invisible_features: tuple[str, ...],
    max_particles: int,
    remove_neutral_non_photon: bool,
    source_sample_index: int,
    source_file_index: int,
    source_row_index: np.ndarray,
) -> ak.Array:
    visible_a = rebuild_vector(selected_events["lead_a_visible_p4"])
    visible_b = rebuild_vector(selected_events["lead_b_visible_p4"])
    event_weight = nominal_event_weight(sample, luminosity, selected_events)
    event_categories = (
        to_numpy(selected_events["event_category"], np.int64)
        if "event_category" in selected_events.fields
        else None
    )
    classification_indices, classification_names = classification_targets_for_sample(
        sample_key=sample.key,
        sample_name=sample.name,
        is_data=sample.is_data,
        num_rows=len(selected_events),
        event_categories=event_categories,
        lookup=classification_lookup,
    )
    if "central_weight" in selected_events.fields:
        central_weight = to_numpy(selected_events["central_weight"], np.float32)
    else:
        central_weight = event_weight.copy()
    x, x_mask, num_sequential_vectors, input_part_mask, point_cloud_feature_names = build_point_cloud(
        selected_events,
        max_particles,
        feature_config,
        remove_neutral_non_photon,
    )
    if tuple(point_cloud_feature_names) != tuple(feature_config.raw_sequential_fields):
        raise ValueError("Point-cloud feature ordering drifted from feature_config.raw_sequential_fields.")
    part_inputs, _ = build_part_inputs(selected_events, feature_config, input_part_mask)
    conditions, conditions_mask, condition_names = build_global_conditions(selected_events, feature_config)
    if tuple(condition_names) != tuple(feature_config.global_fields):
        raise ValueError("Global condition feature ordering drifted from feature_config.global_fields.")
    num_vectors = compute_event_totals(num_sequential_vectors, conditions_mask)
    global_inputs = build_global_inputs(selected_events, feature_config)
    tau_vis_prong_p4, tau_vis_prong_mask, tau_vis_rho_p4, tau_vis_rho_mask = build_visible_tau_targets(selected_events)
    slot_for_a = np.zeros(len(selected_events), dtype=np.int8)
    slot_for_b = np.ones(len(selected_events), dtype=np.int8)
    x_invisible_p4, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target_p4, tau_vis_target_mask = build_invisible_targets_simple(
        sample,
        selected_events,
        tau_vis_prong_p4,
        tau_vis_prong_mask,
    )
    tau_vis_prong = features_from_p4_local(tau_vis_prong_p4, ("energy", "pt", "eta", "phi"))
    tau_vis_rho = features_from_p4_local(tau_vis_rho_p4, ("energy", "pt", "eta", "phi"))
    tau_vis_target = features_from_p4_local(tau_vis_target_p4, ("energy", "pt", "eta", "phi"))
    x_invisible = features_from_p4_local(x_invisible_p4, invisible_features)

    fields: dict[str, Any] = {
        "sample_key": ak.Array([sample.key] * len(selected_events)),
        "sample_name": ak.Array([sample.name] * len(selected_events)),
        "sample_is_data": np.full(len(selected_events), sample.is_data, dtype=bool),
        "sample_is_signal": np.full(len(selected_events), sample.is_signal, dtype=bool),
        "source_sample_index": np.full(len(selected_events), source_sample_index, dtype=np.int64),
        "source_file_index": np.full(len(selected_events), source_file_index, dtype=np.int32),
        "source_event_index": source_row_index.astype(np.int64),
        "source_event_key": source_event_key_array(selected_events, source_row_index),
        "source_slot_for_a": slot_for_a,
        "source_slot_for_b": slot_for_b,
        "classification": classification_indices.astype(np.int64),
        "classification_target_index": classification_indices,
        "classification_target_name": ak.Array(classification_names.tolist()),
        "event_weight": event_weight.astype(np.float32),
        "central_weight": central_weight.astype(np.float32),
        "x": to_numpy_array(x, np.float32),
        "x_mask": to_numpy_array(x_mask, bool),
        "conditions": to_numpy_array(conditions, np.float32),
        "conditions_mask": to_numpy_array(conditions_mask, bool),
        "num_vectors": num_vectors.astype(np.float32),
        "num_sequential_vectors": num_sequential_vectors.astype(np.float32),
        "x_invisible": to_numpy_array(x_invisible, np.float32),
        "x_invisible_mask": to_numpy_array(x_invisible_mask, bool),
        "num_invisible_raw": num_invisible_raw.astype(np.int64),
        "num_invisible_valid": num_invisible_valid.astype(np.int64),
        "tau_vis_prong": to_numpy_array(tau_vis_prong, np.float32),
        "tau_vis_prong_mask": to_numpy_array(tau_vis_prong_mask, bool),
        "tau_vis_rho": to_numpy_array(tau_vis_rho, np.float32),
        "tau_vis_rho_mask": to_numpy_array(tau_vis_rho_mask, bool),
        "tau_vis_target": to_numpy_array(tau_vis_target, np.float32),
        "tau_vis_target_mask": to_numpy_array(tau_vis_target_mask, bool),
        "lead_a_visible_p4": materialize_p4(visible_a),
        "lead_b_visible_p4": materialize_p4(visible_b),
        "tau_vis_prong_slot0_valid": finite_p4_mask(visible_a),
        "tau_vis_prong_slot0_energy": to_numpy(visible_a.E, np.float32),
        "tau_vis_prong_slot0_pt": to_numpy(visible_a.pt, np.float32),
        "tau_vis_prong_slot0_eta": to_numpy(visible_a.eta, np.float32),
        "tau_vis_prong_slot0_phi": to_numpy(visible_a.phi, np.float32),
        "tau_vis_prong_slot1_valid": finite_p4_mask(visible_b),
        "tau_vis_prong_slot1_energy": to_numpy(visible_b.E, np.float32),
        "tau_vis_prong_slot1_pt": to_numpy(visible_b.pt, np.float32),
        "tau_vis_prong_slot1_eta": to_numpy(visible_b.eta, np.float32),
        "tau_vis_prong_slot1_phi": to_numpy(visible_b.phi, np.float32),
    }
    fields.update(part_inputs)
    fields.update(global_inputs)
    if sample.is_signal:
        truth_tau_a = rebuild_vector(selected_events["truth_tau_a_p4"])
        truth_tau_b = rebuild_vector(selected_events["truth_tau_b_p4"])
        target_a, target_valid_a = build_target_missing(truth_tau_a, visible_a)
        target_b, target_valid_b = build_target_missing(truth_tau_b, visible_b)
        truth_observables = build_truth_observables(truth_tau_a, truth_tau_b, visible_a, visible_b)
        fields.update(
            {
                "truth_tau_a_p4": materialize_p4(truth_tau_a),
                "truth_tau_b_p4": materialize_p4(truth_tau_b),
                "target_invisible_slot0_valid": target_valid_a.astype(bool),
                "target_invisible_slot0_pt": to_numpy(target_a.pt, np.float32),
                "target_invisible_slot0_eta": to_numpy(target_a.eta, np.float32),
                "target_invisible_slot0_phi": to_numpy(target_a.phi, np.float32),
                "target_invisible_slot1_valid": target_valid_b.astype(bool),
                "target_invisible_slot1_pt": to_numpy(target_b.pt, np.float32),
                "target_invisible_slot1_eta": to_numpy(target_b.eta, np.float32),
                "target_invisible_slot1_phi": to_numpy(target_b.phi, np.float32),
            }
        )
        fields.update(truth_observables)
    if sample.total_initial_num_events is not None:
        fields["initial_total_num_events"] = np.full(
            len(selected_events),
            sample.total_initial_num_events,
            dtype=np.float64,
        )
    if "nprong" in selected_events.fields:
        fields["nprong"] = to_numpy(selected_events["nprong"], np.int32)
    passthrough = {
        "event_category",
        "truth_QI_region",
        "analyzing_power",
        "analyzing_power_a",
        "analyzing_power_b",
        "weight",
        "central_weight",
    }
    passthrough.update(name for name in selected_events.fields if name.endswith("_cut"))
    for field in sorted(passthrough):
        if field in selected_events.fields and field not in fields:
            fields[field] = selected_events[field]
    return ak.Array(fields)


def write_shard(events: ak.Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(events, path, compression="snappy")


def worker_process_file(
    sample_payload: dict[str, Any],
    luminosity: float | None,
    classification_lookup,
    feature_config,
    invisible_features: tuple[str, ...],
    max_particles: int,
    remove_neutral_non_photon: bool,
    source_sample_index: int,
    file_path: str,
    source_file_index: int,
    output_dir: str,
    batch_size: int,
    rows_per_shard: int,
    do_monitoring: bool,
) -> dict[str, Any]:
    sample = Sample(**sample_payload)
    output_root = Path(output_dir)
    state = empty_monitor_state()
    parquet = pq.ParquetFile(file_path)
    schema_names = {field.name for field in parquet.schema_arrow}
    columns = required_columns(schema_names, feature_config, sample)
    shard_buffers: list[ak.Array] = []
    selected_in_buffer = 0
    written_shards: list[dict[str, Any]] = []
    row_offset = 0
    shard_index = 0

    def flush_buffer() -> None:
        nonlocal shard_buffers, selected_in_buffer, shard_index
        if not shard_buffers:
            return
        shard_events = shard_buffers[0] if len(shard_buffers) == 1 else ak.concatenate(shard_buffers, axis=0)
        shard_path = (
            output_root
            / "shards"
            / sample.key
            / f"{sample.key}__file{source_file_index:03d}__shard{shard_index:05d}.parquet"
        )
        write_shard(shard_events, shard_path)
        written_shards.append(
            {
                "path": str(shard_path.relative_to(output_root)),
                "rows": int(len(shard_events)),
                "source_file": file_path,
                "source_file_index": source_file_index,
            }
        )
        shard_index += 1
        shard_buffers = []
        selected_in_buffer = 0

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        events = ak.from_arrow(record_batch)
        state["rows_seen"] += len(events)
        ensure_required_fields(events, sample, file_path, feature_config)
        vector_fields = ["lead_a_visible_p4", "lead_b_visible_p4"]
        if sample.is_signal:
            vector_fields.extend(["truth_tau_a_p4", "truth_tau_b_p4"])
        if "missing_p4" in events.fields:
            vector_fields.append("missing_p4")
        for field in vector_fields:
            events[field] = rebuild_vector(events[field])
        mask = preselection_mask(events)
        if not np.any(mask):
            row_offset += len(events)
            continue
        selected = events[mask]
        selected_indices = np.flatnonzero(mask).astype(np.int64) + row_offset
        output_events = build_output_events(
            selected_events=selected,
            sample=sample,
            luminosity=luminosity,
            classification_lookup=classification_lookup,
            feature_config=feature_config,
            invisible_features=invisible_features,
            max_particles=max_particles,
            remove_neutral_non_photon=remove_neutral_non_photon,
            source_sample_index=source_sample_index,
            source_file_index=source_file_index,
            source_row_index=selected_indices,
        )
        validate_output_contract(output_events, feature_config, invisible_features, max_particles)
        if do_monitoring:
            fill_monitor_state(state, selected, output_events)
        shard_buffers.append(output_events)
        selected_in_buffer += len(output_events)
        if selected_in_buffer >= rows_per_shard:
            flush_buffer()
        row_offset += len(events)

    flush_buffer()
    return {
        "sample_key": sample.key,
        "source_file": file_path,
        "source_file_index": source_file_index,
        "shards": written_shards,
        "monitor": state,
    }


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    samples: list[Sample],
    luminosity: float | None,
    classification_lookup,
    feature_config,
    invisible_features: tuple[str, ...],
    max_particles: int,
    remove_neutral_non_photon: bool,
    worker_results: list[dict[str, Any]],
    monitoring_outputs: dict[str, dict[str, str]],
) -> None:
    sample_manifest: dict[str, Any] = {}
    for sample in samples:
        sample_manifest[sample.key] = {
            "name": sample.name,
            "is_data": sample.is_data,
            "is_signal": sample.is_signal,
            "file_source": sample.file_source,
            "plot_label": sample.plot_label,
            "files": expand_files(sample.files),
            "total_initial_num_events": sample.total_initial_num_events,
        }
    shards_by_sample: dict[str, list[dict[str, Any]]] = {sample.key: [] for sample in samples}
    for result in worker_results:
        shards_by_sample[result["sample_key"]].extend(result["shards"])
    payload = {
        "format": "ml_pipeline_lite_evenet_input_v2",
        "analysis_config": str(args.analysis_config),
        "batch_size": args.batch_size,
        "rows_per_shard": args.rows_per_shard,
        "num_workers": args.num_workers,
        "luminosity": luminosity,
        "max_particles": max_particles,
        "remove_neutral_non_photon": remove_neutral_non_photon,
        "class_labels": list(classification_lookup.class_labels),
        "sequential_features": list(feature_config.all_sequential_fields),
        "global_condition_features": list(feature_config.global_fields),
        "invisible_features": list(invisible_features),
        "samples": sample_manifest,
        "shards": shards_by_sample,
        "monitoring": monitoring_outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[ml_pipeline_lite] wrote {manifest_path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(args.analysis_config)
    feature_config = parse_feature_config(config)
    invisible_features = infer_invisible_features(config)
    remove_neutral_non_photon = infer_remove_neutral_non_photon(config)
    samples = attach_sample_total_initial_events(parse_samples(config, args.samples))

    if not samples:
        raise ValueError("No samples selected.")

    selected_keys = {sample.key for sample in samples}
    classification_lookup = build_classification_lookup(config, selected_keys)
    sample_index_lookup = {sample.key: index for index, sample in enumerate(samples)}

    luminosity = infer_luminosity(samples)
    jobs: list[tuple[Sample, str, int]] = []
    for sample in samples:
        files = expand_files(sample.files)
        for file_index, file_path in enumerate(files):
            jobs.append((sample, file_path, file_index))

    max_particles = 0
    max_scan_columns = sorted(set(PART_MOMENTUM_SOURCE_FIELDS + ("Part_pdgId", "nprong", "lead_a_visible_p4", "lead_b_visible_p4", "Part_charge")))
    for sample in samples:
        if sample.is_data:
            continue
        for file_path in expand_files(sample.files):
            parquet = pq.ParquetFile(file_path)
            schema_names = {field.name for field in parquet.schema_arrow}
            scan_columns = [column for column in max_scan_columns if column in schema_names]
            row_offset = 0
            for record_batch in parquet.iter_batches(batch_size=args.batch_size, columns=scan_columns):
                events = ak.from_arrow(record_batch)
                events["lead_a_visible_p4"] = rebuild_vector(events["lead_a_visible_p4"])
                events["lead_b_visible_p4"] = rebuild_vector(events["lead_b_visible_p4"])
                mask = preselection_mask(events)
                if np.any(mask):
                    selected = events[mask]
                    input_part_mask = build_input_particle_mask(selected, remove_neutral_non_photon)
                    if len(selected) > 0:
                        batch_max = int(ak.max(ak.num(selected["Part_pdgId"][input_part_mask], axis=1)))
                        max_particles = max(max_particles, batch_max)
                row_offset += len(events)
    if max_particles <= 0:
        raise ValueError("Unable to infer a positive max_particles from MC inputs.")

    print(f"[ml_pipeline_lite] output_dir={output_dir}")
    print(f"[ml_pipeline_lite] samples={[sample.key for sample in samples]}")
    print(f"[ml_pipeline_lite] class_labels={list(classification_lookup.class_labels)}")
    print(f"[ml_pipeline_lite] jobs={len(jobs)} workers={args.num_workers}")
    print(f"[ml_pipeline_lite] invisible_features={list(invisible_features)}")
    print(f"[ml_pipeline_lite] remove_neutral_non_photon={remove_neutral_non_photon}")
    print(f"[ml_pipeline_lite] max_particles={max_particles}")
    for sample in samples:
        print(
            f"[ml_pipeline_lite] sample={sample.key} source={sample.file_source} "
            f"total_initial_num_events={sample.total_initial_num_events}"
        )

    worker_results: list[dict[str, Any]] = []
    max_workers = max(1, min(args.num_workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                worker_process_file,
                sample_payload={
                    "key": sample.key,
                    "name": sample.name,
                    "is_data": sample.is_data,
                    "is_signal": sample.is_signal,
                    "files": sample.files,
                    "file_source": sample.file_source,
                    "norm_factor": sample.norm_factor,
                    "lumi": sample.lumi,
                    "total_initial_num_events": sample.total_initial_num_events,
                    "plot_label": sample.plot_label,
                },
                luminosity=luminosity,
                classification_lookup=classification_lookup,
                feature_config=feature_config,
                invisible_features=invisible_features,
                max_particles=max_particles,
                remove_neutral_non_photon=remove_neutral_non_photon,
                source_sample_index=sample_index_lookup[sample.key],
                file_path=file_path,
                source_file_index=file_index,
                output_dir=str(output_dir),
                batch_size=args.batch_size,
                rows_per_shard=args.rows_per_shard,
                do_monitoring=bool(args.monitoring),
            ): (sample.key, file_path)
            for sample, file_path, file_index in jobs
        }
        for future in as_completed(future_map):
            sample_key, file_path = future_map[future]
            result = future.result()
            worker_results.append(result)
            rows_written = sum(int(item["rows"]) for item in result["shards"])
            print(
                f"[ml_pipeline_lite] finished sample={sample_key} file={file_path} "
                f"shards={len(result['shards'])} rows={rows_written}"
            )

    monitoring_outputs: dict[str, dict[str, str]] = {}
    if args.monitoring:
        states_by_sample: dict[str, list[dict[str, Any]]] = {sample.key: [] for sample in samples}
        for result in worker_results:
            states_by_sample[result["sample_key"]].append(result["monitor"])
        merged_states: dict[str, dict[str, Any]] = {}
        for sample in samples:
            merged = merge_monitor_states(states_by_sample[sample.key])
            merged_states[sample.key] = merged
            monitoring_outputs[sample.key] = write_histogram_plots(output_dir, sample, merged)
        comparison_outputs = write_data_mc_comparison_plots(output_dir, samples, merged_states)
        if comparison_outputs:
            monitoring_outputs["comparison"] = comparison_outputs

    write_manifest(
        output_dir,
        args,
        samples,
        luminosity,
        classification_lookup,
        feature_config,
        invisible_features,
        max_particles,
        remove_neutral_non_photon,
        worker_results,
        monitoring_outputs,
    )


if __name__ == "__main__":
    main()
