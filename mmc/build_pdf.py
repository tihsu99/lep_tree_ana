#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build and validate per-p_tau-bin PDFs for dR(vis, invis), where:

    PDF(dR | p) = w * Gaussian(mu, sigma) + h * Landau(A, B)

with six free parameters:
    w, mu, sigma, h, A, B

Definitions:
    - dR is the angular distance between the visible and invisible tau decay products
    - p is the true tau momentum magnitude
    - p_tau is binned from 0 to 45 GeV with a bin width of 0.5 GeV

Decay modes treated separately:
  1) leptonic tau decay : visible = lepton (e / mu), invisible = neutrino
  2) rho-like tau decay : visible = pion,            invisible = neutrino

Main functionalities:
  1) Build PDF parameters from a parquet file containing true tau, neutrino,
     and visible decay-product kinematics.
     The fit is performed independently in each p_tau bin and separately for
     the leptonic and rho-like decay channels.

  2) Perform a closure test using an existing PDF parameter file and a parquet
     file of truth-level events.
     In the closure test, the predicted dR distribution from the stored PDF is
     compared with the true dR distribution from the parquet input, bin-by-bin
     in p_tau, in order to validate the PDF parameterization.

Output format:
    The output HDF5 file stores two groups:
      - /lep : fitted parameter arrays for leptonic tau decay
      - /rho : fitted parameter arrays for rho-like tau decay

    Each group contains datasets mapping:
      parameter name -> array over p_tau bins

Typical datasets include:
      w, mu, sigma, h, A, B,
      ptau_bin_low, ptau_bin_high, ptau_bin_center,
      success, n_entries

Usage:
    Build PDFs:
        python fit_tau_dr_pdf.py build input.parquet

    Closure test:
        python fit_tau_dr_pdf.py closure --input-parquet input.parquet --input-pdf pdf_params.h5 --output output_dir

Optional:
    python fit_tau_dr_pdf.py build input.parquet \
        --output output.h5 --min-events 50 --dr-max 5.0 --dr-bins 120

    python fit_tau_dr_pdf.py closure \
        --input-parquet input.parquet --input-pdf pdf_params.h5 --output output_dir --modes lep rho
"""

import argparse
import math
import warnings
import os

import awkward as ak
import numpy as np
import vector
import h5py
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from MMC_util import mixture_pdf, get_ptau_bin_edges
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme


# ============================================================
# Physics / fit helpers
# ============================================================

def histogram_density(x, bins, x_range=None):
    """
    Return histogram bin centers and density histogram.
    """
    hist, edges = np.histogram(x, bins=bins, range=x_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def initial_guess(x):
    """
    Heuristic initial guess for fit parameters.
    """
    if len(x) == 0:
        return [0.5, 0.5, 0.2, 0.5, 1.0, 0.3]

    x = np.asarray(x)
    med = np.median(x)
    std = np.std(x) if np.std(x) > 1e-6 else 0.2
    q75 = np.quantile(x, 0.75)

    # Since histogram is density-normalized, amplitudes are O(1)
    return [
        0.5,                 # w
        max(0.0, med * 0.7), # mu
        max(0.05, std * 0.6),# sigma
        0.5,                 # h
        max(0.0, q75),       # A
        max(0.05, std),      # B
    ]


def fit_one_bin(dr_values, dr_hist_bins=120, dr_range=(0.0, 5.0)):
    """
    Fit one p_tau bin.
    Returns fitted parameters and a status flag.
    """
    dr_values = np.asarray(dr_values, dtype=float)
    dr_values = dr_values[np.isfinite(dr_values)]
    dr_values = dr_values[(dr_values >= dr_range[0]) & (dr_values <= dr_range[1])]

    if len(dr_values) == 0:
        return {
            "w": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "h": np.nan,
            "A": np.nan,
            "B": np.nan,
            "success": False,
            "n_entries": 0,
        }

    xhist, yhist = histogram_density(dr_values, bins=dr_hist_bins, x_range=dr_range)

    # Keep only non-zero bins to avoid fitting a lot of empty bins
    mask = np.isfinite(yhist) & (yhist > 0)
    xfit = xhist[mask]
    yfit = yhist[mask]

    if len(xfit) < 6:
        return {
            "w": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "h": np.nan,
            "A": np.nan,
            "B": np.nan,
            "success": False,
            "n_entries": len(dr_values),
        }

    p0 = initial_guess(dr_values)

    # Broad but physical bounds
    bounds_lower = [0.0, 0.0, 1e-4, 0.0, 0.0, 1e-4]
    bounds_upper = [100.0, 5.0, 5.0, 100.0, 5.0, 5.0]

    try:
        popt, _ = curve_fit(
            mixture_pdf,
            xfit,
            yfit,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=20000,
        )
        return {
            "w": float(popt[0]),
            "mu": float(popt[1]),
            "sigma": float(popt[2]),
            "h": float(popt[3]),
            "A": float(popt[4]),
            "B": float(popt[5]),
            "success": True,
            "n_entries": len(dr_values),
        }
    except Exception:
        return {
            "w": np.nan,
            "mu": np.nan,
            "sigma": np.nan,
            "h": np.nan,
            "A": np.nan,
            "B": np.nan,
            "success": False,
            "n_entries": len(dr_values),
        }


# ============================================================
# Kinematics extraction
# ============================================================

def build_decay_sample(arr, tau_pdgId, vis_pdgId):
    """
    Build per-event arrays for:
      - p_tau
      - dR(vis, mis)
    """
    pdgId = arr['GenPart_pdgId']
    flag_valid = (ak.sum(pdgId == vis_pdgId, axis=1) == 1)
    # there are always more than 1 electron in the event, so set all events to valid
    if abs(vis_pdgId) == 11:
        flag_valid = np.ones_like(flag_valid, dtype=bool)
        print(f"[WARNING] Skipping visible particle count check for electrons (pdgId={vis_pdgId}). All events (len={len(arr)}) will be treated as valid.")
    if ak.sum(~flag_valid) > 0:
        print(f"[WARNING] Found {ak.sum(~flag_valid)} events with !=1 visible particle (pdgId={vis_pdgId}). These events will be skipped.")
        print(f"Valid events: {ak.sum(flag_valid)}, Invalid events: {ak.sum(~flag_valid)}")

    # Filter to events with exactly one visible particle of the requested type
    arr = arr[flag_valid]
    pdgId = arr['GenPart_pdgId']
    tau_p4 = get_p4_from_ak_events(arr, pdgId == tau_pdgId, prefix='GenPart_vector')
    vis_p4 = get_p4_from_ak_events(arr, pdgId == vis_pdgId, prefix='GenPart_vector')
    mis_p4 = tau_p4 - vis_p4

    p_tau = tau_p4.p.to_numpy(allow_missing=False)
    dr = vis_p4.deltaR(mis_p4).to_numpy(allow_missing=False)
    return p_tau, dr


# ============================================================
# Main fitting workflow
# ============================================================

def fit_decay_mode(p_tau, dr, ptau_edges, min_events=50, dr_hist_bins=120, dr_range=(0.0, 5.0)):
    """
    Fit all p_tau bins for one decay mode.
    """
    n_bins = len(ptau_edges) - 1

    out = {
        "ptau_bin_id": [],
        "ptau_bin_low": [],
        "ptau_bin_high": [],
        "ptau_bin_center": [],
        "w": [],
        "mu": [],
        "sigma": [],
        "h": [],
        "A": [],
        "B": [],
        "success": [],
        "n_entries": [],
    }

    for i in range(n_bins):
        lo = ptau_edges[i]
        hi = ptau_edges[i + 1]

        # include left edge, exclude right edge except for final bin
        if i < n_bins - 1:
            mask = (p_tau >= lo) & (p_tau < hi)
        else:
            mask = (p_tau >= lo) & (p_tau <= hi)

        dr_bin = dr[mask]
        n_entries = len(dr_bin)

        if n_entries < min_events:
            fitres = {
                "w": np.nan,
                "mu": np.nan,
                "sigma": np.nan,
                "h": np.nan,
                "A": np.nan,
                "B": np.nan,
                "success": False,
                "n_entries": n_entries,
            }
        else:
            fitres = fit_one_bin(
                dr_bin,
                dr_hist_bins=dr_hist_bins,
                dr_range=dr_range,
            )

        out["ptau_bin_id"].append(i)
        out["ptau_bin_low"].append(float(lo))
        out["ptau_bin_high"].append(float(hi))
        out["ptau_bin_center"].append(float(0.5 * (lo + hi)))
        out["w"].append(fitres["w"])
        out["mu"].append(fitres["mu"])
        out["sigma"].append(fitres["sigma"])
        out["h"].append(fitres["h"])
        out["A"].append(fitres["A"])
        out["B"].append(fitres["B"])
        out["success"].append(bool(fitres["success"]))
        out["n_entries"].append(int(fitres["n_entries"]))

    return {k: np.asarray(v) for k, v in out.items()}


def get_event_category_list(decay_mode: str, charge: int):
    decay_mode_truth_category = {
        "rho": 2, 
        "el": 3,
        "mu": 4,
    }[decay_mode.lower()]

    # 5, which corresponds to "other" category, is not used for now
    if charge < 0:
        return [i*10 + decay_mode_truth_category for i in range(1, 5)]
    elif charge > 0:
        return [10*decay_mode_truth_category + i for i in range(1, 5)]


def build_ptau_dr_arrays(arr, decay_mode: str):
    # Build p_tau and dR arrays for the specified decay mode.
    # decay_mode can be "lep" or "rho"
    p_tau_all, dr_all = np.array([]), np.array([])
    detailed_decay_modes = [("el", 11), ("mu", 13)] if decay_mode == "lep" else [("rho", 211)] 
    for vis_mode, vis_pdgId in detailed_decay_modes:
        for charge in [1, -1]:
            event_category_list = get_event_category_list(vis_mode, charge)
            mask = np.isin(arr['event_category'], event_category_list)
            p_tau_bin, dr_bin = build_decay_sample(
                arr[mask],
                tau_pdgId=-1 * charge * 15,
                vis_pdgId=-1 * charge * vis_pdgId if decay_mode == "lep" else charge * vis_pdgId,
            )
            p_tau_all = np.concatenate([p_tau_all, p_tau_bin])
            dr_all = np.concatenate([dr_all, dr_bin])
    return p_tau_all, dr_all


def run_build(args):
    print(f"[INFO] Reading input parquet: {args.input_parquet}")
    arr = ak.from_parquet(args.input_parquet)

    print("[INFO] Building leptonic decay sample...")
    p_tau_lep, dr_lep = build_ptau_dr_arrays(arr, decay_mode="lep")

    print("[INFO] Building rho decay sample...")
    p_tau_rho, dr_rho = build_ptau_dr_arrays(arr, decay_mode="rho")
    print(f"ptau min, max: {p_tau_lep.min():.3f}, {p_tau_lep.max():.3f} (lep), {p_tau_rho.min():.3f}, {p_tau_rho.max():.3f} (rho)")

    ptau_edges = get_ptau_bin_edges()
    print("[INFO] Fitting leptonic decay bins...")
    lep_result = fit_decay_mode(
        p_tau_lep,
        dr_lep,
        ptau_edges=ptau_edges,
        min_events=args.min_events,
        dr_hist_bins=args.dr_bins,
        dr_range=(0.0, args.dr_max),
    )

    print("[INFO] Fitting rho decay bins...")
    rho_result = fit_decay_mode(
        p_tau_rho,
        dr_rho,
        ptau_edges=ptau_edges,
        min_events=args.min_events,
        dr_hist_bins=args.dr_bins,
        dr_range=(0.0, args.dr_max),
    )

    # Save results to h5py file
    with h5py.File(args.output, "w") as f:
        lep_group = f.create_group("lep")
        for k, v in lep_result.items():
            lep_group.create_dataset(k, data=v)

        rho_group = f.create_group("rho")
        for k, v in rho_result.items():
            rho_group.create_dataset(k, data=v)

    print("[INFO] Done.")
    print(f"[INFO] Leptonic events used: {len(p_tau_lep)}")
    print(f"[INFO] Rho events used     : {len(p_tau_rho)}")


def run_closure(args):
    print(f"[INFO] Running closure test with input parquet: {args.input_parquet} and input PDF: {args.input_pdf}")
    arr = ak.from_parquet(args.input_parquet)
    pdf_file = h5py.File(args.input_pdf, "r")

    for mode in args.modes:
        print(f"[INFO] Validating decay mode: {mode}")
        output_path = os.path.join(args.output_dir, mode)
        os.makedirs(output_path, exist_ok=True)

        # Build p_tau and dR arrays for this decay mode
        p_tau, dr = build_ptau_dr_arrays(arr, decay_mode=mode)

        # Get PDF parameters for this decay mode
        pdf_params = pdf_file[mode]

        # Plot closure test for each p_tau bin
        n_bins = len(pdf_params["ptau_bin_id"])
        for i in range(n_bins):
            pt_low = pdf_params["ptau_bin_low"][i]
            pt_high = pdf_params["ptau_bin_high"][i]
            pt_center = pdf_params["ptau_bin_center"][i]

            mask = (p_tau >= pt_low) & (p_tau < pt_high)
            dr_bin = dr[mask]

            if len(dr_bin) == 0:
                print(f"[WARNING] No events in p_tau bin {i} ({pt_low} - {pt_high} GeV). Skipping.")
                continue

            w, mu, sigma, h, A, B = 0, 0, 0, 0, 0, 0
            if pdf_params["success"][i]:
                w = pdf_params["w"][i]
                mu = pdf_params["mu"][i]
                sigma = pdf_params["sigma"][i]
                h = pdf_params["h"][i]
                A = pdf_params["A"][i]
                B = pdf_params["B"][i]

            # Generate PDF curve
            dr_values = np.linspace(0.0, args.dr_max, args.dr_bins)
            pdf_values = mixture_pdf(dr_values, w, mu, sigma, h, A, B)

            # Plot histogram and PDF
            plt.figure(figsize=(8, 6))
            plt.hist(dr_bin, bins=args.dr_bins, range=(0.0, args.dr_max), density=True, alpha=0.6, label="True dR")
            plt.plot(dr_values, pdf_values, label="Fitted PDF", color="red")
            plt.title(f"Closure Test - {mode.upper()} Decay - p_tau {pt_center:.3f} GeV")
            plt.xlabel(r"$\Delta R$")
            plt.ylabel("Density")
            plt.legend()
            plt.grid()

            plot_filename = os.path.join(output_path, f"closure_ptau_{pt_center:.3f}_GeV.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"[INFO] Saved closure plot for p_tau bin {i} to {plot_filename}")






def make_parser():
    parser = argparse.ArgumentParser(
        description="Build and validate tau dR PDFs."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --------------------------------------------------
    # build subcommand
    # --------------------------------------------------
    p_build = subparsers.add_parser(
        "build",
        help="Build PDF parameters from parquet"
    )
    p_build.add_argument("input_parquet", help="Input parquet file",
            default="/eos/user/c/cmo/project/ZtautauLep/tree_ana/run/20260311-pipi/Ztautau/filtered___tautau.parquet")
    p_build.add_argument("--output", help="Output PDF file, e.g. pdf.h5", default="mmc_lep_rho_parameters.h5")
    p_build.add_argument(
        "--dr-max", type=float, default=0.5,
        help="Maximum dR used for fit"
    )
    p_build.add_argument(
        "--dr-bins", type=int, default=120,
        help="Number of dR histogram bins"
    )
    p_build.add_argument(
        "--min-events", type=int, default=50,
        help="Minimum events required in a tau momentum bin"
    )

    # --------------------------------------------------
    # closure subcommand
    # --------------------------------------------------
    p_closure = subparsers.add_parser(
        "closure",
        help="Run closure test using an existing PDF file and parquet sample"
    )
    p_closure.add_argument("--input-parquet", help="Input parquet file for validation",
        default="/eos/user/c/cmo/project/ZtautauLep/tree_ana/run/20260311-pipi/Ztautau/filtered___tautau.parquet")
    p_closure.add_argument("--input-pdf", help="Existing PDF parameter file, e.g. pdf.h5",
        default="mmc_lep_rho_parameters.h5"
    )

    p_closure.add_argument(
        "--output-dir", default="closure_output",
        help="Directory to store closure plots/results"
    )
    p_closure.add_argument(
        "--dr-max", type=float, default=0.5,
        help="Maximum dR shown in closure comparison"
    )
    p_closure.add_argument(
        "--dr-bins", type=int, default=120,
        help="Number of dR bins for closure histogram"
    )
    p_closure.add_argument(
        "--modes", nargs="+", choices=["lep", "rho"], default=["lep", "rho"],
        help="Decay modes to validate"
    )

    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "build":
        run_build(args)
    elif args.command == "closure":
        run_closure(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()