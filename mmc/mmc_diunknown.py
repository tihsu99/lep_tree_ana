import numpy as np
import vector
import time
from pathlib import Path
import argparse
import h5py
import pickle
import awkward as ak

from mmc.MMC_util import mixture_pdf, get_ptau_bin_id, parallel_calculation

vector.register_awkward()

class MMCDiUnknown():
    def __init__(
            self,
            hist_array_1: dict,
            hist_array_2: dict,
            sqrt_s: float = 91.2,        
            phi_grid_points: int = 50,   
            theta_grid_points: int = 50, 
            phi_search_range: float = 0.3,
            theta_search_range: float = 0.3,
    ):
        self.M_tau = 1.777  # GeV
        self.hist_array_1 = hist_array_1
        self.hist_array_2 = hist_array_2
        self.sqrt_s = sqrt_s
        self.theta_search_range = theta_search_range
        self.theta_grid_points = theta_grid_points
        self.phi_grid_points = phi_grid_points
        self.phi_search_range = phi_search_range
        self.E_tau = self.sqrt_s / 2.0
        self.P_tau_mag = np.sqrt(self.E_tau**2 - self.M_tau**2)

    def P_deltaR(self, delta_R: np.ndarray, p_tau: np.ndarray, is_tau_1: bool):
        bin_indices = get_ptau_bin_id(p_tau)
        current_hist = self.hist_array_1 if is_tau_1 else self.hist_array_2
        max_idx = len(current_hist["w"]) - 1
        bin_indices = np.clip(bin_indices, 0, max_idx)

        w = current_hist["w"][bin_indices]
        mu = current_hist["mu"][bin_indices]
        sigma = current_hist["sigma"][bin_indices]
        h = current_hist["h"][bin_indices]
        A = current_hist["A"][bin_indices]
        B = current_hist["B"][bin_indices]

        return mixture_pdf(delta_R, w, mu, sigma, h, A, B)
    
    def event_likelihood(self, delta_R1: np.ndarray, p_tau1: np.ndarray, delta_R2: np.ndarray, p_tau2: np.ndarray):
        """Vectorized version of event_likelihood using NumPy broadcasting."""

        # Create a mask for NaN values
        nan_mask = np.isnan(delta_R1) | np.isnan(delta_R2) | np.isnan(p_tau1) | np.isnan(p_tau2)

        # Compute P_deltaR values using broadcasting (Removed likelihood_scale)
        P1 = self.P_deltaR(delta_R1, p_tau1, is_tau_1=True)
        P2 = self.P_deltaR(delta_R2, p_tau2, is_tau_1=False)

        # Compute final probability while handling invalid values
        likelihoods = np.where(
            (P1 * P2 > 0) & (~nan_mask),  # Only compute when valid
            P1 * P2,
            0.0  # Assign 0.0 for invalid/zero probability cases (NOT np.inf)
        )

        return likelihoods

    def calculation(
            self,
            vis_1,
            vis_2
    ):
        # 1. Force inputs to pure NumPy to avoid Awkward Array crashes
        v1_phi = np.asarray(vis_1.phi)
        v1_theta = np.asarray(vis_1.theta)
        v2_phi = np.asarray(vis_2.phi)
        v2_theta = np.asarray(vis_2.theta)
        num_events = len(v1_phi)

        # 2. Build grids using pure NumPy
        phi_vals = np.linspace(v1_phi - self.phi_search_range, v1_phi + self.phi_search_range, self.phi_grid_points, axis=1)
        theta_vals = np.linspace(v1_theta - self.theta_search_range, v1_theta + self.theta_search_range, self.theta_grid_points, axis=1)

        phi_tau = phi_vals[:, :, None]
        theta_tau = theta_vals[:, None, :]
        phi_tau, theta_tau = np.broadcast_arrays(phi_tau, theta_tau)

        # Explicitly cast to numpy before reshaping
        phi_tau = np.asarray(phi_tau).reshape(num_events, -1)     
        theta_tau = np.asarray(theta_tau).reshape(num_events, -1) 

        # 3. Calculate independent tau momenta
        tau1_px = self.P_tau_mag * np.sin(theta_tau) * np.cos(phi_tau)
        tau1_py = self.P_tau_mag * np.sin(theta_tau) * np.sin(phi_tau)
        tau1_pz = self.P_tau_mag * np.cos(theta_tau)
        
        # 2D Assumption: Taus are perfectly back-to-back
        tau2_px = -tau1_px
        tau2_py = -tau1_py
        tau2_pz = -tau1_pz

        num_candidates = phi_tau.shape[1]

        # 4. Repeat visible momenta using NumPy (bypassing Awkward's lack of .reshape)
        v1_px_rep = np.repeat(np.asarray(vis_1.px)[:, None], num_candidates, axis=1)
        v1_py_rep = np.repeat(np.asarray(vis_1.py)[:, None], num_candidates, axis=1)
        v1_pz_rep = np.repeat(np.asarray(vis_1.pz)[:, None], num_candidates, axis=1)
        v1_E_rep  = np.repeat(np.asarray(vis_1.E)[:, None], num_candidates, axis=1)

        v2_px_rep = np.repeat(np.asarray(vis_2.px)[:, None], num_candidates, axis=1)
        v2_py_rep = np.repeat(np.asarray(vis_2.py)[:, None], num_candidates, axis=1)
        v2_pz_rep = np.repeat(np.asarray(vis_2.pz)[:, None], num_candidates, axis=1)
        v2_E_rep  = np.repeat(np.asarray(vis_2.E)[:, None], num_candidates, axis=1)

        # 5. Extract neutrino kinematics
        nu1_px = tau1_px - v1_px_rep
        nu1_py = tau1_py - v1_py_rep
        nu1_pz = tau1_pz - v1_pz_rep
        nu1_E  = self.E_tau - v1_E_rep

        nu2_px = tau2_px - v2_px_rep
        nu2_py = tau2_py - v2_py_rep
        nu2_pz = tau2_pz - v2_pz_rep
        nu2_E  = self.E_tau - v2_E_rep

        # 6. Rebuild vector arrays for safe deltaR math
        nu1 = vector.array({"px": nu1_px, "py": nu1_py, "pz": nu1_pz, "E": nu1_E})
        nu2 = vector.array({"px": nu2_px, "py": nu2_py, "pz": nu2_pz, "E": nu2_E})
        
        vis_1_rep = vector.array({"px": v1_px_rep, "py": v1_py_rep, "pz": v1_pz_rep, "E": v1_E_rep})
        vis_2_rep = vector.array({"px": v2_px_rep, "py": v2_py_rep, "pz": v2_pz_rep, "E": v2_E_rep})

       # --- FIX: Safe Missing Mass & Kinematic Bounds ---
        M_MU = 0.10566  # Muon mass in GeV
        max_mass_sq = (self.M_tau - M_MU)**2

        m_mis1_sq = nu1.E**2 - nu1.p**2
        m_mis2_sq = nu2.E**2 - nu2.p**2
        
        # Prevent floating point underflow (fixes the 'inf' Truth bug)
        m_mis1_sq = np.maximum(m_mis1_sq, 0.0)
        m_mis2_sq = np.maximum(m_mis2_sq, 0.0)

        # Enforce strict physical upper and lower bounds
        valid_mass_mask = (m_mis1_sq >= 0) & (m_mis2_sq >= 0) & \
                          (m_mis1_sq <= max_mass_sq) & (m_mis2_sq <= max_mass_sq)

        tau1 = vector.array({"px": tau1_px, "py": tau1_py, "pz": tau1_pz, "E": np.full_like(tau1_px, self.E_tau)})
        tau2 = vector.array({"px": tau2_px, "py": tau2_py, "pz": tau2_pz, "E": np.full_like(tau2_px, self.E_tau)})

        # 7. Calculate Likelihoods
        L_kinematics = self.event_likelihood(vis_1_rep.deltaR(nu1), tau1.p, vis_2_rep.deltaR(nu2), tau2.p)

        # Apply bounds to phase space
        L_phase_space_1 = np.where(valid_mass_mask, m_mis1_sq * (self.M_tau**2 - m_mis1_sq)**2, 0)
        L_phase_space_2 = np.where(valid_mass_mask, m_mis2_sq * (self.M_tau**2 - m_mis2_sq)**2, 0)
        L_ps_total = L_phase_space_1 * L_phase_space_2
        L_ps_total = L_ps_total / (np.max(L_ps_total, axis=1, keepdims=True) + 1e-10) 

        # NO MET CONSTRAINT FOR 2D SCAN
        L_total_prob = np.where(valid_mass_mask, L_kinematics * L_ps_total, 0)
        
        L_total = np.where(L_total_prob > 0, L_total_prob, np.inf)
        L_events = np.where(np.isfinite(L_total), -np.log(L_total), np.inf)

        best_indices = np.argmin(L_events, axis=1)

        best_nu1 = np.take_along_axis(nu1, best_indices[:, None], axis=1).squeeze(1)
        best_nu2 = np.take_along_axis(nu2, best_indices[:, None], axis=1).squeeze(1)
        best_likelihood = np.take_along_axis(L_events, best_indices[:, None], axis=1).squeeze(1)
        
        best_likelihood_exp = np.where(np.isfinite(best_likelihood), best_likelihood, np.inf)
        best_likelihood_exp = np.exp(-best_likelihood_exp)

        return best_nu1, best_nu2, best_likelihood_exp
    
    def plot_diagnostic(self, vis_1, vis_2, true_tau1=None, true_nu1=None, output_dir="./"):
        import matplotlib.pyplot as plt
        import numpy as np
        import vector
        
        # We only plot ONE event. Extract the scalar values.
        v1_phi = vis_1.phi[0]
        v1_theta = vis_1.theta[0]
        v1_px, v1_py, v1_pz, v1_E = vis_1.px[0], vis_1.py[0], vis_1.pz[0], vis_1.E[0]
        
        v2_px, v2_py, v2_pz, v2_E = vis_2.px[0], vis_2.py[0], vis_2.pz[0], vis_2.E[0]

        # 1. Create a dense 2D grid for the PoIs (phi_tau1, theta_tau1)
        phi_vals = np.linspace(v1_phi - self.phi_search_range, v1_phi + self.phi_search_range, 100)
        theta_vals = np.linspace(v1_theta - self.theta_search_range, v1_theta + self.theta_search_range, 100)
        Phi, Theta = np.meshgrid(phi_vals, theta_vals)

        # 2. Kinematics for tau1 across the whole grid
        tau1_px = self.P_tau_mag * np.sin(Theta) * np.cos(Phi)
        tau1_py = self.P_tau_mag * np.sin(Theta) * np.sin(Phi)
        tau1_pz = self.P_tau_mag * np.cos(Theta)
        tau1_E = np.full_like(tau1_px, self.E_tau)

        # 2D assumption: tau2 is exactly back-to-back
        tau2_px, tau2_py, tau2_pz = -tau1_px, -tau1_py, -tau1_pz
        tau2_E = tau1_E

        # 3. Calculate Neutrinos
        nu1_px = tau1_px - v1_px
        nu1_py = tau1_py - v1_py
        nu1_pz = tau1_pz - v1_pz
        nu1_E  = tau1_E - v1_E

        nu2_px = tau2_px - v2_px
        nu2_py = tau2_py - v2_py
        nu2_pz = tau2_pz - v2_pz
        nu2_E  = tau2_E - v2_E

        # Put everything into Vector arrays for the math functions
        nu1 = vector.array({"px": nu1_px, "py": nu1_py, "pz": nu1_pz, "E": nu1_E})
        nu2 = vector.array({"px": nu2_px, "py": nu2_py, "pz": nu2_pz, "E": nu2_E})
        tau1 = vector.array({"px": tau1_px, "py": tau1_py, "pz": tau1_pz, "E": tau1_E})
        tau2 = vector.array({"px": tau2_px, "py": tau2_py, "pz": tau2_pz, "E": tau2_E})

        v1_grid = vector.array({"px": np.full_like(Phi, v1_px), "py": np.full_like(Phi, v1_py), "pz": np.full_like(Phi, v1_pz), "E": np.full_like(Phi, v1_E)})
        v2_grid = vector.array({"px": np.full_like(Phi, v2_px), "py": np.full_like(Phi, v2_py), "pz": np.full_like(Phi, v2_pz), "E": np.full_like(Phi, v2_E)})

        # 4. Calculate Z-Axis Variable 1: dR(vis, mis)
        dR_1 = v1_grid.deltaR(nu1)
        dR_2 = v2_grid.deltaR(nu2)

        # 5. Calculate Z-Axis Variable 2: Likelihood
        L_kinematics = self.event_likelihood(dR_1, tau1.p, dR_2, tau2.p)

        # --- FIX: Safe Missing Mass & Kinematic Bounds for Diagnostic Plot ---
        M_MU = 0.10566
        max_mass_sq = (self.M_tau - M_MU)**2

        m_mis1_sq = nu1.E**2 - nu1.p**2
        m_mis2_sq = nu2.E**2 - nu2.p**2
        
        m_mis1_sq = np.maximum(m_mis1_sq, 0.0)
        m_mis2_sq = np.maximum(m_mis2_sq, 0.0)

        valid_mass_mask = (m_mis1_sq >= 0) & (m_mis2_sq >= 0) & \
                          (m_mis1_sq <= max_mass_sq) & (m_mis2_sq <= max_mass_sq)

        L_phase_space_1 = np.where(valid_mass_mask, m_mis1_sq * (self.M_tau**2 - m_mis1_sq)**2, 0)
        L_phase_space_2 = np.where(valid_mass_mask, m_mis2_sq * (self.M_tau**2 - m_mis2_sq)**2, 0)
        L_ps_total = L_phase_space_1 * L_phase_space_2
        if np.max(L_ps_total) > 0:
            L_ps_total = L_ps_total / np.max(L_ps_total)

        L_total_prob = np.where(valid_mass_mask, L_kinematics, 0)

        # 6. Calculate the Vanilla PoIs for comparison! 
        m_mis1_grid = np.sqrt(np.maximum(nu1_E**2 - nu1.p**2, 0))
        phi_mis1_grid = np.arctan2(nu1_py, nu1_px)

        # ==========================================
        # 7. PLOT THE SURFACES (Now 4 panels: 2x2)
        # ==========================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

        # Panel 1 (Top Left): dR(vis, mis) Surface
        im1 = axes[0, 0].pcolormesh(Phi, Theta, dR_1, shading='auto', cmap='viridis')
        axes[0, 0].plot(v1_phi, v1_theta, 'r*', markersize=15, label='Visible Muon')
        if true_tau1 is not None:
            axes[0, 0].plot(true_tau1.phi[0], true_tau1.theta[0], 'wX', markersize=12, markeredgecolor='k', label='TRUE Tau')
        axes[0, 0].set_title(r"$\Delta R$ Surface (Your PoIs)")
        axes[0, 0].set_xlabel(r"$\phi_{\tau 1}$")
        axes[0, 0].set_ylabel(r"$\theta_{\tau 1}$")
        axes[0, 0].legend()
        fig.colorbar(im1, ax=axes[0, 0], label=r"$\Delta R$")

        # Panel 2 (Top Right): Likelihood Surface
        im2 = axes[0, 1].pcolormesh(Phi, Theta, L_total_prob, shading='auto', cmap='plasma')
        axes[0, 1].plot(v1_phi, v1_theta, 'r*', markersize=15, label='Visible Muon')
        if true_tau1 is not None:
            axes[0, 1].plot(true_tau1.phi[0], true_tau1.theta[0], 'wX', markersize=12, markeredgecolor='k', label='TRUE Tau')
        axes[0, 1].set_title("Likelihood (Your PoIs)")
        axes[0, 1].set_xlabel(r"$\phi_{\tau 1}$")
        axes[0, 1].set_ylabel(r"$\theta_{\tau 1}$")
        axes[0, 1].legend()
        fig.colorbar(im2, ax=axes[0, 1], label="Likelihood Probability")

        # Panel 3 (Bottom Left): Missing Mass Surface
        im3 = axes[1, 0].pcolormesh(Phi, Theta, m_mis1_grid, shading='auto', cmap='coolwarm')
        axes[1, 0].plot(v1_phi, v1_theta, 'r*', markersize=15, label='Visible Muon')
        if true_tau1 is not None:
            axes[1, 0].plot(true_tau1.phi[0], true_tau1.theta[0], 'wX', markersize=12, markeredgecolor='k', label='TRUE Tau')
        axes[1, 0].set_title(r"Missing Mass $m_{mis 1}$ Surface (Your PoIs)")
        axes[1, 0].set_xlabel(r"$\phi_{\tau 1}$")
        axes[1, 0].set_ylabel(r"$\theta_{\tau 1}$")
        axes[1, 0].legend()
        fig.colorbar(im3, ax=axes[1, 0], label=r"$m_{mis 1}$ (GeV)")

        # Panel 4 (Bottom Right): Likelihood in Vanilla Space
        im4 = axes[1, 1].scatter(phi_mis1_grid, m_mis1_grid, c=L_total_prob, cmap='plasma', s=5, alpha=0.8)
        if true_nu1 is not None:
            true_m_mis = np.sqrt(np.maximum(true_nu1.E[0]**2 - true_nu1.p[0]**2, 0))
            axes[1, 1].plot(true_nu1.phi[0], true_m_mis, 'wX', markersize=12, markeredgecolor='k', label='TRUE Missing System')
        axes[1, 1].set_title("Likelihood (Vanilla MMC PoIs)")
        axes[1, 1].set_xlabel(r"$\phi_{mis 1}$")
        axes[1, 1].set_ylabel(r"$m_{mis 1}$ (GeV)")
        axes[1, 1].set_ylim(0, 1.8) 
        axes[1, 1].legend()
        fig.colorbar(im4, ax=axes[1, 1], label="Likelihood Probability")

        plt.tight_layout()
        out_path = Path(output_dir) / "cen_mo_diagnostic_pois_4panel_v3.png"
        plt.savefig(out_path)
        plt.close()
        print(f"\n✅ Saved 4-panel diagnostic plots with TRUTH markers to {out_path}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Quantum Information Analysis - LepLep (2D LEP Scan)")
    parser.add_argument("--sqrt_s", type=float, default=91.2, help="Center of Mass Energy in GeV")
    parser.add_argument("--phi_grid_points", type=int, default=300, help="Number of phi grid points")
    parser.add_argument("--theta_grid_points", type=int, default=300, help="Number of theta grid points")
    parser.add_argument("--phi_search_range", type=float, default=0.3, help="Phi search range")
    parser.add_argument("--theta_search_range", type=float, default=0.3, help="Theta search range")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--max_events", type=int, default=-1, help="Max events to process (-1 for all)")
    parser.add_argument("--raw_file_path", type=str, default="/eos/user/c/chenhua/lep_tree_ana/output/tau_v3_condor_new_event_category/Ztautau/filtered___tautau_inclusive_preselection_with_mumu_prad0p9_iso_170_loose_cuts_only_final_cut.parquet", help="Path to Pre-filtered Parquet data")
    parser.add_argument("--pdf_file_path", type=str, default="mmc_lep_lep_parameters_from_raw_new1.h5", help="Path to PDF HDF5 file")
    parser.add_argument("--output_file_path", type=str, default="./", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading pre-filtered dataset from {args.raw_file_path}...")
    data = ak.from_parquet(args.raw_file_path)
    
    # ========================================================
    # NEW: APPLY EVENT CATEGORY == 44 CUT HERE
    # ========================================================
    cat_branch = None
    if "event_category" in data.fields:
        cat_branch = "event_category"
    elif "Event_category" in data.fields:
        cat_branch = "Event_category"
    elif "category" in data.fields:
        cat_branch = "category"
        
    if cat_branch:
        mask = data[cat_branch] == 44
        data = data[mask]
        print(f"🎯 Applied {cat_branch} == 44 cut.")
    else:
        print("⚠️ Warning: Could not find category branch to apply cut.")
    # ========================================================

    total_events = len(data)
    print(f"Total pure events found in file after cut: {total_events}")
    
    total_events = len(data)
    print(f"Total pure events found in file: {total_events}")
    
    if args.max_events > 0 and total_events > args.max_events:
        print(f"Applying cap: only processing first {args.max_events} events.")
        data = data[:args.max_events]
    elif total_events == 0:
        print("❌ Error: 0 events in parquet file! Exiting.")
        exit()

    # Create NumPy Vectors directly from the explicit branches
    vis1_np = vector.arr({
        "px": ak.to_numpy(data["LeadA_px"]),
        "py": ak.to_numpy(data["LeadA_py"]),
        "pz": ak.to_numpy(data["LeadA_pz"]),
        "E":  ak.to_numpy(data["LeadA_E"])
    })
    
    vis2_np = vector.arr({
        "px": ak.to_numpy(data["LeadB_px"]),
        "py": ak.to_numpy(data["LeadB_py"]),
        "pz": ak.to_numpy(data["LeadB_pz"]),
        "E":  ak.to_numpy(data["LeadB_E"])
    })

    eventID = ak.to_numpy(data["evtNumber"])

    # --- Extract Truth for Diagnostic Plotting ---
    charge_a = ak.to_numpy(data["LeadA_charge"])
    
    true_tau1_px = np.where(charge_a > 0, ak.to_numpy(data["GenTauPlus_px"]), ak.to_numpy(data["GenTauMinus_px"]))
    true_tau1_py = np.where(charge_a > 0, ak.to_numpy(data["GenTauPlus_py"]), ak.to_numpy(data["GenTauMinus_py"]))
    true_tau1_pz = np.where(charge_a > 0, ak.to_numpy(data["GenTauPlus_pz"]), ak.to_numpy(data["GenTauMinus_pz"]))
    true_tau1_E  = np.where(charge_a > 0, ak.to_numpy(data["GenTauPlus_E"]),  ak.to_numpy(data["GenTauMinus_E"]))
    true_tau1_np = vector.arr({"px": true_tau1_px, "py": true_tau1_py, "pz": true_tau1_pz, "E": true_tau1_E})
    
    true_nu1_np = true_tau1_np - vis1_np
    # ---------------------------------------------

    func_dict_path = Path(__file__).parent / args.pdf_file_path
    print(f"Loading PDF parameters from {func_dict_path}...")
    
    hist_array_lep = {}
    with h5py.File(func_dict_path, "r") as f:
        lep_group = f["lep"]
        for key in lep_group.keys():
            hist_array_lep[key] = lep_group[key][:]

    mmc = MMCLepLepLEP2D(
        hist_array_1=hist_array_lep,
        hist_array_2=hist_array_lep,
        sqrt_s=args.sqrt_s,
        phi_grid_points=args.phi_grid_points,
        theta_grid_points=args.theta_grid_points,
        phi_search_range=args.phi_search_range,
        theta_search_range=args.theta_search_range,
        sigma_MET=1.0 
    )

    print(f"Running highly-optimized 2D neutrino prediction for mumu channel...")
    start_time = time.time()
    
    print("Generating Cen Mo's diagnostic PoI plot for Event 0...")
    mmc.plot_diagnostic(vis1_np[:1], vis2_np[:1], true_tau1=true_tau1_np[:1], true_nu1=true_nu1_np[:1], output_dir=args.output_file_path)
    
    nu1, nu2, likelihood = parallel_calculation(
        mmc, vis1_np, vis2_np,
        num_workers=args.num_workers
    )

    print("✅ Completed parallel processing with time elapsed: {:.2f}s".format(time.time() - start_time))

    # --- FIX: SORT NEUTRINOS BY CHARGE (Removes the 'X' shape in Evaluate) ---
    nu_p_px = np.where(charge_a > 0, nu1.px, nu2.px)
    nu_p_py = np.where(charge_a > 0, nu1.py, nu2.py)
    nu_p_pz = np.where(charge_a > 0, nu1.pz, nu2.pz)
    nu_p_E  = np.where(charge_a > 0, nu1.E,  nu2.E)
    nu_p_sorted = vector.array({"px": nu_p_px, "py": nu_p_py, "pz": nu_p_pz, "E": nu_p_E})

    nu_m_px = np.where(charge_a < 0, nu1.px, nu2.px)
    nu_m_py = np.where(charge_a < 0, nu1.py, nu2.py)
    nu_m_pz = np.where(charge_a < 0, nu1.pz, nu2.pz)
    nu_m_E  = np.where(charge_a < 0, nu1.E,  nu2.E)
    nu_m_sorted = vector.array({"px": nu_m_px, "py": nu_m_py, "pz": nu_m_pz, "E": nu_m_E})

    output_path = Path(args.output_file_path) / "mmc_mumu_2d_v2_new.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({
            "nu_p": nu_p_sorted,
            "nu_m": nu_m_sorted,
            "Likelihood": likelihood, 
            "EventID": eventID
        }, f)
    print(f"Results successfully saved to {output_path}")