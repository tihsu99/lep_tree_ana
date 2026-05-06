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


vector.register_awkward()

MAX_VISIBLE_ENERGY_GEV = 91.25

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


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(getattr(values, "fields", []))
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return vector.zip({"px": values["x"], "py": values["y"], "pz": values["z"], "E": values["t"]})
    raise ValueError(f"Unsupported four-vector fields: {sorted(fields)}")


def build_momentum4d(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> ak.Array:
    return ak.zip(
        {
            "px": np.asarray(px, dtype=np.float64),
            "py": np.asarray(py, dtype=np.float64),
            "pz": np.asarray(pz, dtype=np.float64),
            "E": np.asarray(energy, dtype=np.float64),
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


def required_columns(schema_names: set[str]) -> list[str]:
    columns = {
        "lead_a_visible_p4",
        "lead_b_visible_p4",
        "truth_tau_a_p4",
        "truth_tau_b_p4",
        "nprong",
        "event_category",
        "truth_QI_region",
        "analyzing_power",
        "analyzing_power_a",
        "analyzing_power_b",
        "initial_total_num_events",
        "weight",
        "central_weight",
    }
    columns.update(name for name in schema_names if name.endswith("_cut"))
    return sorted(name for name in columns if name in schema_names)


def ensure_required_fields(events: ak.Array, sample: Sample, path: str) -> None:
    required = {"lead_a_visible_p4", "lead_b_visible_p4", "truth_tau_a_p4", "truth_tau_b_p4"}
    missing = sorted(required - set(events.fields))
    if missing:
        raise ValueError(
            f"Sample '{sample.key}' file '{path}' is missing required fields: {missing}"
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
        "target_pt": target_pt,
        "target_eta": target_eta,
        "target_phi": target_phi,
        "truth_theta_cm": to_numpy(output_events["truth_theta_cm"]),
        "truth_mtautau": to_numpy(output_events["truth_mtautau"]),
        "truth_cos_theta_A_k": to_numpy(output_events["truth_cos_theta_A_k"]),
        "truth_cos_theta_B_k": to_numpy(output_events["truth_cos_theta_B_k"]),
        "truth_cos_theta_A_k_times_cos_theta_B_k": to_numpy(output_events["truth_cos_theta_A_k_times_cos_theta_B_k"]),
    }
    if "nprong" in selected_events.fields:
        values_1d["nprong"] = to_numpy(selected_events["nprong"], np.float64)

    for name, values in values_1d.items():
        finite = np.isfinite(values)
        if np.any(finite):
            weights = event_weight if len(values) == len(event_weight) else doubled_event_weight
            state["counts_1d"][name] += histogram1d(values[finite], MONITOR_1D_SPECS[name], weights=weights[finite])

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


def write_histogram_plots(output_dir: Path, sample_key: str, state: dict[str, Any]) -> dict[str, str]:
    monitor_dir = output_dir / "monitoring" / sample_key
    monitor_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}

    for name, spec in MONITOR_1D_SPECS.items():
        counts = state["counts_1d"][name]
        low, high, bins = spec
        edges = np.linspace(low, high, bins + 1)
        fig, axis = plt.subplots(figsize=(6.2, 4.6), dpi=160)
        axis.step(edges[:-1], counts, where="post", linewidth=1.7)
        axis.set_ylabel("Weighted yield")
        axis.set_title(f"{sample_key}: {name}")
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
        axis.set_title(f"{sample_key}: {name}")
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

    for name, spec in MONITOR_1D_SPECS.items():
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
                label=sample.key,
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

        axis.set_xlabel(name)
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
    source_file_index: int,
    source_row_index: np.ndarray,
) -> ak.Array:
    visible_a = rebuild_vector(selected_events["lead_a_visible_p4"])
    visible_b = rebuild_vector(selected_events["lead_b_visible_p4"])
    truth_tau_a = rebuild_vector(selected_events["truth_tau_a_p4"])
    truth_tau_b = rebuild_vector(selected_events["truth_tau_b_p4"])
    target_a, target_valid_a = build_target_missing(truth_tau_a, visible_a)
    target_b, target_valid_b = build_target_missing(truth_tau_b, visible_b)
    truth_observables = build_truth_observables(truth_tau_a, truth_tau_b, visible_a, visible_b)
    event_weight = nominal_event_weight(sample, luminosity, selected_events)
    if "central_weight" in selected_events.fields:
        central_weight = to_numpy(selected_events["central_weight"], np.float32)
    else:
        central_weight = event_weight.copy()

    fields: dict[str, Any] = {
        "sample_key": ak.Array([sample.key] * len(selected_events)),
        "sample_name": ak.Array([sample.name] * len(selected_events)),
        "sample_is_data": np.full(len(selected_events), sample.is_data, dtype=bool),
        "sample_is_signal": np.full(len(selected_events), sample.is_signal, dtype=bool),
        "source_file_index": np.full(len(selected_events), source_file_index, dtype=np.int32),
        "source_event_index": source_row_index.astype(np.int64),
        "source_slot_for_a": np.zeros(len(selected_events), dtype=np.int8),
        "source_slot_for_b": np.ones(len(selected_events), dtype=np.int8),
        "event_weight": event_weight.astype(np.float32),
        "central_weight": central_weight.astype(np.float32),
        "lead_a_visible_p4": materialize_p4(visible_a),
        "lead_b_visible_p4": materialize_p4(visible_b),
        "truth_tau_a_p4": materialize_p4(truth_tau_a),
        "truth_tau_b_p4": materialize_p4(truth_tau_b),
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
        "target_invisible_slot0_valid": target_valid_a.astype(bool),
        "target_invisible_slot0_pt": to_numpy(target_a.pt, np.float32),
        "target_invisible_slot0_eta": to_numpy(target_a.eta, np.float32),
        "target_invisible_slot0_phi": to_numpy(target_a.phi, np.float32),
        "target_invisible_slot1_valid": target_valid_b.astype(bool),
        "target_invisible_slot1_pt": to_numpy(target_b.pt, np.float32),
        "target_invisible_slot1_eta": to_numpy(target_b.eta, np.float32),
        "target_invisible_slot1_phi": to_numpy(target_b.phi, np.float32),
    }
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
    fields.update(truth_observables)
    return ak.Array(fields)


def write_shard(events: ak.Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(events, path, compression="snappy")


def worker_process_file(
    sample_payload: dict[str, Any],
    luminosity: float | None,
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
    columns = required_columns(schema_names)
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
        ensure_required_fields(events, sample, file_path)
        for field in ("lead_a_visible_p4", "lead_b_visible_p4", "truth_tau_a_p4", "truth_tau_b_p4"):
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
            source_file_index=source_file_index,
            source_row_index=selected_indices,
        )
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
            "files": expand_files(sample.files),
            "total_initial_num_events": sample.total_initial_num_events,
        }
    shards_by_sample: dict[str, list[dict[str, Any]]] = {sample.key: [] for sample in samples}
    for result in worker_results:
        shards_by_sample[result["sample_key"]].extend(result["shards"])
    payload = {
        "format": "ml_pipeline_lite_evenet_input_v1",
        "analysis_config": str(args.analysis_config),
        "batch_size": args.batch_size,
        "rows_per_shard": args.rows_per_shard,
        "num_workers": args.num_workers,
        "luminosity": luminosity,
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
    samples = attach_sample_total_initial_events(parse_samples(config, args.samples))

    if not samples:
        raise ValueError("No samples selected.")

    luminosity = infer_luminosity(samples)
    jobs: list[tuple[Sample, str, int]] = []
    for sample in samples:
        files = expand_files(sample.files)
        for file_index, file_path in enumerate(files):
            jobs.append((sample, file_path, file_index))

    print(f"[ml_pipeline_lite] output_dir={output_dir}")
    print(f"[ml_pipeline_lite] samples={[sample.key for sample in samples]}")
    print(f"[ml_pipeline_lite] jobs={len(jobs)} workers={args.num_workers}")
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
                },
                luminosity=luminosity,
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
            monitoring_outputs[sample.key] = write_histogram_plots(output_dir, sample.key, merged)
        comparison_outputs = write_data_mc_comparison_plots(output_dir, samples, merged_states)
        if comparison_outputs:
            monitoring_outputs["comparison"] = comparison_outputs

    write_manifest(output_dir, args, samples, luminosity, worker_results, monitoring_outputs)


if __name__ == "__main__":
    main()
