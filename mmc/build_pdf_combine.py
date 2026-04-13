#!/usr/bin/env python3
"""
Build Unified PDF from raw parquet for ALL leptonic channels by:
1. Filtering event_category in [33, 44, 34, 43]
2. Extracting taus, decay electrons (E < 45 GeV), and muons regardless of charge
3. Fitting generic 'electron' and 'muon' PDF distributions
"""

import pyarrow.parquet as pq
import awkward as ak
import numpy as np
import vector
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm, landau
#import pylandau  # Using pylandau to fix the scipy.stats issue!
import h5py
import argparse

vector.register_awkward()

def mixture_pdf(x, w, mu, sigma, h, A, B):
    """Mixture PDF: w*Gaussian + h*Landau"""
    sigma = max(sigma, 1e-6)
    B = max(B, 1e-6)
    gauss = norm.pdf(x, loc=mu, scale=sigma)
    # Replaced scipy landau with pylandau, using mpv and eta arguments
    landau_pdf = landau.pdf(x, loc=A, scale=B)
    return w * gauss + h * landau_pdf

def get_ptau_bin_edges():
    """Return the edges of p_tau bins."""
    ptau_edges = np.concatenate([
        np.arange(10, 40, 10),
        np.arange(40, 44, 0.5),
        np.arange(44, 45, 0.1),
        np.arange(45, 45.6+0.001, 0.01),
    ])
    return ptau_edges

def extract_kinematics_combined(events):
    """
    Extract p_tau and dR grouped solely by 'electron' and 'muon'.
    Applies the E < 45 GeV veto to all electrons to filter out initial-state beams.
    """
    print("[INFO] Extracting particles from GenPart (Combined Channels)...")
    
    gen_pdg = events["GenPart_pdgId"]
    gen_p4_all = vector.zip({
        "px": events["GenPart_vector_fCoordinates_fX"], 
        "py": events["GenPart_vector_fCoordinates_fY"],
        "pz": events["GenPart_vector_fCoordinates_fZ"], 
        "E":  events["GenPart_vector_fCoordinates_fT"],
    })
    
    def get_safe_gen_p4(p4_array):
        return vector.zip({
            "px": ak.fill_none(p4_array.px, 0.0), "py": ak.fill_none(p4_array.py, 0.0),
            "pz": ak.fill_none(p4_array.pz, 0.0), "E":  ak.fill_none(p4_array.E,  0.0),
        })

    # Extract Taus
    tau_minus = get_safe_gen_p4(ak.firsts(gen_p4_all[gen_pdg == 15]))
    tau_plus  = get_safe_gen_p4(ak.firsts(gen_p4_all[gen_pdg == -15]))

    # Extract Electrons (with beam energy veto E < 45)
    e_minus_first = ak.firsts(gen_p4_all[(gen_pdg == 11) & (gen_p4_all.E < 45.0)])
    e_plus_first  = ak.firsts(gen_p4_all[(gen_pdg == -11) & (gen_p4_all.E < 45.0)])
    # Fallback just in case E < 45 removes everything for a weird event
    e_minus = ak.where(ak.is_none(e_minus_first), ak.firsts(gen_p4_all[gen_pdg == 11]), e_minus_first)
    e_plus  = ak.where(ak.is_none(e_plus_first), ak.firsts(gen_p4_all[gen_pdg == -11]), e_plus_first)
    
    e_minus_safe = get_safe_gen_p4(e_minus)
    e_plus_safe  = get_safe_gen_p4(e_plus)

    # Extract Muons (No beam veto needed for muons at LEP)
    mu_minus_safe = get_safe_gen_p4(ak.firsts(gen_p4_all[gen_pdg == 13]))
    mu_plus_safe  = get_safe_gen_p4(ak.firsts(gen_p4_all[gen_pdg == -13]))

    # --- KINEMATICS CALCULATION ---
    # Because Safe arrays pad missing particles with E=0, we use E>0 to filter valid matches
    
    # 1. Electrons
    valid_e_minus = e_minus_safe.E > 0
    nu_e_minus = tau_minus - e_minus_safe
    dr_e_minus = e_minus_safe.deltaR(nu_e_minus)

    valid_e_plus = e_plus_safe.E > 0
    nu_e_plus = tau_plus - e_plus_safe
    dr_e_plus = e_plus_safe.deltaR(nu_e_plus)

    # Combine e- and e+ into a single electron array
    p_tau_ele = np.concatenate([ak.to_numpy(tau_minus.p[valid_e_minus]), ak.to_numpy(tau_plus.p[valid_e_plus])])
    dr_ele = np.concatenate([ak.to_numpy(dr_e_minus[valid_e_minus]), ak.to_numpy(dr_e_plus[valid_e_plus])])

    # 2. Muons
    valid_mu_minus = mu_minus_safe.E > 0
    nu_mu_minus = tau_minus - mu_minus_safe
    dr_mu_minus = mu_minus_safe.deltaR(nu_mu_minus)

    valid_mu_plus = mu_plus_safe.E > 0
    nu_mu_plus = tau_plus - mu_plus_safe
    dr_mu_plus = mu_plus_safe.deltaR(nu_mu_plus)

    # Combine mu- and mu+ into a single muon array
    p_tau_mu = np.concatenate([ak.to_numpy(tau_minus.p[valid_mu_minus]), ak.to_numpy(tau_plus.p[valid_mu_plus])])
    dr_mu = np.concatenate([ak.to_numpy(dr_mu_minus[valid_mu_minus]), ak.to_numpy(dr_mu_plus[valid_mu_plus])])

    results = {
        'electron': (p_tau_ele, dr_ele),
        'muon': (p_tau_mu, dr_mu)
    }

    print(f"  -> Extracted {len(p_tau_ele)} valid electron decay pairings.")
    print(f"  -> Extracted {len(p_tau_mu)} valid muon decay pairings.")

    return results

def fit_one_bin(p_tau, dr, ptau_low, ptau_high, dr_range=(0.0, 5.0), dr_hist_bins=240, min_events=50):
    """Fit one p_tau bin with robust fallback mechanisms for highly boosted taus."""
    mask = (p_tau >= ptau_low) & (p_tau < ptau_high)
    n_events = np.sum(mask)
    
    if n_events < min_events:
        return False, None
    
    dr_bin = dr[mask]
    
    hist, bin_edges = np.histogram(dr_bin, bins=dr_hist_bins, range=dr_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    hist_fit = hist.astype(float)
    centers_fit = bin_centers
    
    if np.sum(hist_fit) < 10:
        return False, None
    
    peak_idx = np.argmax(hist_fit)
    peak_mu = centers_fit[peak_idx]
    
    mean_dr = np.average(centers_fit, weights=hist_fit)
    var_dr = np.average((centers_fit - mean_dr)**2, weights=hist_fit)
    rms_dr = np.sqrt(var_dr) if var_dr > 0 else 0.05
    
    sigma_guess = np.clip(rms_dr / 2.0, 0.002, 0.2)
    landau_mu_guess = min(peak_mu + 0.02, 4.0)
    landau_sigma_guess = sigma_guess
    
    p0 = [0.7, peak_mu, sigma_guess, 0.3, landau_mu_guess, landau_sigma_guess]
    
    try:
        popt, _ = curve_fit(
            mixture_pdf,
            centers_fit, hist_fit,
            p0=p0,
            bounds=(
                [0.0, 0.0, 0.002, 0.0, 0.0, 0.002], 
                [1.0, 5.0, 0.5,   1.0, 5.0, 0.5]    
            ),
            maxfev=15000,
            ftol=1e-6
        )
    except RuntimeError:
        try:
            p0_fallback = [0.9, peak_mu, 0.005, 0.1, peak_mu + 0.01, 0.01]
            popt, _ = curve_fit(
                mixture_pdf,
                centers_fit, hist_fit,
                p0=p0_fallback,
                bounds=(
                    [0.0, 0.0, 0.0005, 0.0, 0.0, 0.0005], 
                    [1.0, 5.0, 1.0,    1.0, 5.0, 1.0]
                ),
                maxfev=25000, 
                ftol=1e-5
            )
        except Exception as e:
            print(f"      -> Fit exception: {e}")
            return False, None
            
    w, mu, sigma, h, A, B = popt
    w_sum = w + h
    if w_sum > 1e-6:
        w = w / w_sum
        h = h / w_sum
    
    if (sigma > 1e-5 and B > 1e-5 and 
        0 <= w <= 1 and 0 <= h <= 1 and 
        not np.isnan([w, mu, sigma, h, A, B]).any()):
        return True, (w, mu, sigma, h, A, B)
        
    return False, None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_parquet", type=str, default="/eos/user/c/chenhua/lep_tree_ana/output/tau_v3_condor_new_event_category/Ztautau/filtered___raw.parquet")
    parser.add_argument("--output_h5", type=str, default="mmc_lep_lep_parameters.h5")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"[INFO] Reading raw parquet: {args.raw_parquet}")
    events = ak.from_parquet(args.raw_parquet)
    
    print(f"[INFO] Filtering event_category for ANY leptonic channel [33, 44, 34, 43]...")
    valid_cats = (events['event_category'] == 33) | (events['event_category'] == 44) | \
                 (events['event_category'] == 34) | (events['event_category'] == 43)
    events_filtered = events[valid_cats]
    
    print(f"[INFO] Filtered events: {len(events_filtered)}")
    
    # Extract kinematics cleanly into a dictionary of arrays
    kinematics_by_type = extract_kinematics_combined(events_filtered)
    ptau_edges = get_ptau_bin_edges()
    
    # Prepare to store all results
    results_by_type = {}
    
    for lep_type in ['electron', 'muon']:
        p_tau, dr = kinematics_by_type[lep_type]
        
        if len(p_tau) == 0:
            print(f"\n[INFO] Skipping {lep_type} (No events found)")
            continue

        print(f"\n[INFO] Fitting {lep_type}...")
        results = {
            'ptau_bin_low': [], 'ptau_bin_high': [], 'ptau_bin_center': [],
            'success': [], 'w': [], 'mu': [], 'sigma': [],
            'h': [], 'A': [], 'B': [], 'n_entries': []
        }
        
        n_success = 0
        for i in range(len(ptau_edges) - 1):
            ptau_low = ptau_edges[i]
            ptau_high = ptau_edges[i+1]
            ptau_center = (ptau_low + ptau_high) / 2
            
            n_events = np.sum((p_tau >= ptau_low) & (p_tau < ptau_high))
            success, params = fit_one_bin(p_tau, dr, ptau_low, ptau_high)
            
            results['ptau_bin_low'].append(ptau_low)
            results['ptau_bin_high'].append(ptau_high)
            results['ptau_bin_center'].append(ptau_center)
            results['n_entries'].append(n_events)
            results['success'].append(success if success else False)
            
            if success:
                n_success += 1
                w, mu, sigma, h, A, B = params
                results['w'].append(w)
                results['mu'].append(mu)
                results['sigma'].append(sigma)
                results['h'].append(h)
                results['A'].append(A)
                results['B'].append(B)
                print(f"  Bin {i:2d} ({ptau_low:.2f}-{ptau_high:.2f}): {n_events:4d} events ✓")
            else:
                for k in ['w', 'mu', 'sigma', 'h', 'A', 'B']:
                    results[k].append(np.nan)
                status = "✗ fit failed" if n_events >= 50 else f"✗ {n_events} events < 50"
                print(f"  Bin {i:2d} ({ptau_low:.2f}-{ptau_high:.2f}): {status}")
        
        results_by_type[lep_type] = (results, n_success)
    
    # Save to Master HDF5
    print(f"\n[INFO] Saving Master PDF to {args.output_h5}...")
    with h5py.File(args.output_h5, 'w') as f:
        for lep_type, (results, n_success) in results_by_type.items():
            type_group = f.create_group(lep_type)
            for key, values in results.items():
                type_group.create_dataset(key, data=values)
            print(f"  {lep_type:10s}: Successful fits: {n_success}/{len(ptau_edges)-1} bins")
    
    print(f"[INFO] Done! You can now configure Neutrino.py to use {args.output_h5}")

if __name__ == "__main__":
    main()