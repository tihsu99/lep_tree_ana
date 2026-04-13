import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import copy
import vector
import awkward as ak
import tqdm
import h5py
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events,\
                    get_all_p4_from_ak_events, cme, m_tau, rebuild_p4
from utils.plotter import plot_y_vs_x
from quantum.observables_builder import build_observables, get_observable_names, get_mean_and_err_of_mean
from mmc.MMC import MMC

def evaluate_observables(events, dl_name, output_dir):
    # Compare the distributions of the observables built from reconstructed neutrino momenta to those built from truth neutrino momenta
    weights = events['weight'].to_numpy() if 'weight' in events.fields else np.ones(len(events))
    observables = get_observable_names()
    num_observables = len(observables)
    n_rows = int(np.ceil(num_observables / 3))

    fig_distribution, axes_distribution = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows), dpi=300)
    fig_scatter, axes_scatter = plt.subplots(n_rows, 3, figsize=(12, 5*n_rows), dpi=300)

    for i, obs_name in enumerate(observables):
        # get observable values for truth and reco neutrino momenta
        obs_reco = events[f'{obs_name}']
        obs_truth = events[f'truth_{obs_name}']

        # Distribution plot
        ax_dist = axes_distribution[i//3, i%3]
        mean, err_of_mean = get_mean_and_err_of_mean(ak.to_numpy(obs_truth), weights=weights)
        ax_dist.hist(ak.to_numpy(obs_truth), bins=50, range=(-1, 1), histtype='step', density=True, label=f'Truth (μ={mean:.3f}±{err_of_mean:.3f})', linewidth=2, weights=weights)
        mean, err_of_mean = get_mean_and_err_of_mean(ak.to_numpy(obs_reco), weights=weights)
        ax_dist.hist(ak.to_numpy(obs_reco), bins=50, range=(-1, 1), histtype='step', density=True, label=f'Reconstructed (μ={mean:.3f}±{err_of_mean:.3f})', linewidth=2, weights=weights)
        ax_dist.set_xlabel(obs_name)
        ax_dist.set_ylabel('Density')
        ax_dist.set_title(f'{obs_name} Distribution for {dl_name}')
        ax_dist.legend()

        # Scatter plot
        ax_scatter = axes_scatter[i//3, i%3]
        ax_scatter.scatter(ak.to_numpy(obs_truth), ak.to_numpy(obs_reco), alpha=0.8, s=3)
        ax_scatter.plot([-1, 1], [-1, 1], 'r--')
        ax_scatter.set_xlabel(f'Truth {obs_name}')
        ax_scatter.set_ylabel(f'Reconstructed {obs_name}')
        ax_scatter.set_title(f'{obs_name} Reconstruction for {dl_name}')

    fig_distribution.tight_layout()
    fig_distribution.savefig(f"{output_dir}/{dl_name}_observables_distribution.png")
    plt.close(fig_distribution)
    fig_scatter.tight_layout()
    fig_scatter.savefig(f"{output_dir}/{dl_name}_observables_scatter.png")
    plt.close(fig_scatter)

    # plot correlation between A and B observables
    for is_truth in [False, True]:
        fig_corr, axes_corr = plt.subplots(3, 3, figsize=(15, 15), dpi=300)
        for i, axis_a in enumerate(['n', 'r', 'k']):
            for j, axis_b in enumerate(['n', 'r', 'k']):
                obs_a_name = f'{"truth_" if is_truth else ""}cos_theta_A_{axis_a}'
                obs_b_name = f'{"truth_" if is_truth else ""}cos_theta_B_{axis_b}'
                obs_a = events[obs_a_name]
                obs_b = events[obs_b_name]
                ax_corr = axes_corr[i, j]
                ax_corr.scatter(ak.to_numpy(obs_a), ak.to_numpy(obs_b), alpha=0.8, s=3)
                ax_corr.set_xlabel(obs_a_name)
                ax_corr.set_ylabel(obs_b_name)
                ax_corr.set_title(f'Correlation between {obs_a_name} and {obs_b_name} for {dl_name}')
        fig_corr.tight_layout()
        fig_corr.savefig(f"{output_dir}/{dl_name}_{'truth_' if is_truth else ''}observables_correlation.png")
        plt.close(fig_corr)


def evaluate_reconstruction(events, dl_name, output_dir):
    """
        Evaluate the neutrino reconstruction performance by comparing the reconstructed neutrino 4-momenta to the truth neutrino 4-momenta for events passing the selection.
            - events: ak.Array of events that contain the following fields:
                - GenPart_pdgId: need to have +-15 for truth tau
                - lead_a/b_visible_p4, lead_a/b_missing_p4: for reconstructed visible and missing 4-momenta
    """
    def get_p4(events, prefix):
        return vector.zip({
            "px": events[f'{prefix}_p4'].x,
            "py": events[f'{prefix}_p4'].y,
            "pz": events[f'{prefix}_p4'].z,
            "E": events[f'{prefix}_p4'].t,
        })

    tau_p4 = get_p4_from_ak_events(events, (events['GenPart_pdgId'] == 15), prefix='GenPart_vector')
    antitau_p4 = get_p4_from_ak_events(events, (events['GenPart_pdgId'] == -15), prefix='GenPart_vector')
    # get reconstructed vis and mis p4
    reco_vis_negative_p4 = get_p4(events, 'lead_b_visible')
    reco_vis_positive_p4 = get_p4(events, 'lead_a_visible')
    reco_mis_negativep4_array = get_p4(events, 'lead_b_missing')
    reco_mis_positivep4_array = get_p4(events, 'lead_a_missing')

    # compute truth missing p4 by subtracting visible from tau p4
    miss_positive_p4 = antitau_p4 - reco_vis_positive_p4
    miss_negative_p4 = tau_p4 - reco_vis_negative_p4



    # plot comparison of reconstructed vs truth missing objects
    fig_scatter, axes_scatter = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
    fig_diff, axes_diff = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
    fig_diff_rel, axes_diff_rel = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
    fig_distribution, axes_distribution = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
    for i, var in enumerate(['px', 'py', 'pz', 'E']):
        plot_range = [-cme/2, cme/2]
        if var == 'E':
            plot_range = [0, cme/2]
        #################
        # Neutrino
        #################
        # Neutrino Scatter Plot
        ax_nu = axes_scatter[i, 0]
        truth_ary = getattr(miss_negative_p4, var)
        reco_ary = getattr(reco_mis_negativep4_array, var)
        ax_nu.scatter(ak.to_numpy(truth_ary), ak.to_numpy(reco_ary), alpha=0.8, label=f'Reconstructed {var}', s=3)
        ax_nu.plot(plot_range, plot_range, 'r--', label='Ideal')
        ax_nu.set_xlabel(f'Truth Neutrino {var} (GeV)')
        ax_nu.set_ylabel(f'Reconstructed Neutrino {var} (GeV)')
        ax_nu.set_title(f'Neutrino {var} Reconstruction for {dl_name}')
        ax_nu.legend()

        # Neutrino Difference Plot
        ax_reco_mis_negativediff = axes_diff[i, 0]
        delta_ary = ak.to_numpy(reco_ary - truth_ary)
        plot_y_vs_x(x=ak.to_numpy(truth_ary), y=delta_ary, ax=ax_reco_mis_negativediff, band='68')
        ax_reco_mis_negativediff.set_xlabel(f'Truth Neutrino {var} [GeV]')
        ax_reco_mis_negativediff.set_ylabel(f'Error of Reconstruction')
        ax_reco_mis_negativediff.set_title(f'Neutrino {var} Reconstruction Error vs Truth for {dl_name}')
        ax_reco_mis_negativediff.set_ylim(-3, 3)

        # Neutrino Relative Difference Plot
        rel_err_ary = (delta_ary) / ak.to_numpy(truth_ary)
        ax_reco_mis_negativediff_rel = axes_diff_rel[i, 0]
        plot_y_vs_x(x=ak.to_numpy(truth_ary), y=rel_err_ary, ax=ax_reco_mis_negativediff_rel, band='68')
        ax_reco_mis_negativediff_rel.set_xlabel(f'Truth Neutrino {var} [GeV]')
        ax_reco_mis_negativediff_rel.set_ylabel(f'Relative Error of Reconstruction')
        ax_reco_mis_negativediff_rel.set_title(f'Neutrino {var} Relative Reconstruction Error vs Truth for {dl_name}')
        ax_reco_mis_negativediff_rel.set_ylim(-0.1, 0.1)

        # Neutrino Distribution Plot
        ax_reco_mis_negativedist = axes_distribution[i, 0]
        ax_reco_mis_negativedist.hist(ak.to_numpy(truth_ary), bins=30, range=plot_range, label='Truth', density=True, histtype='step', linewidth=2)
        ax_reco_mis_negativedist.hist(ak.to_numpy(reco_ary), bins=30, range=plot_range, label='Reconstructed', density=True, histtype='step', linewidth=2)
        ax_reco_mis_negativedist.set_xlabel(f'Neutrino {var} (GeV)')
        ax_reco_mis_negativedist.set_ylabel('Normalized Counts')
        ax_reco_mis_negativedist.set_title(f'Neutrino {var} Distribution for {dl_name}')
        ax_reco_mis_negativedist.legend()


        #################
        # Anti-Neutrino
        #################
        # Anti-Neutrino Scatter Plot
        ax_anti_nu = axes_scatter[i, 1]
        truth_ary_anti = getattr(miss_positive_p4, var)
        reco_ary_anti = getattr(reco_mis_positivep4_array, var)
        ax_anti_nu.scatter(ak.to_numpy(truth_ary_anti), ak.to_numpy(reco_ary_anti), alpha=0.8, label=f'Reconstructed {var}', s=3)
        ax_anti_nu.plot(plot_range, plot_range, 'r--', label='Ideal')
        ax_anti_nu.set_xlabel(f'Truth Anti-Neutrino {var} (GeV)')
        ax_anti_nu.set_ylabel(f'Reconstructed Anti-Neutrino {var} (GeV)')
        ax_anti_nu.set_title(f'Anti-Neutrino {var} Reconstruction for {dl_name}')
        ax_anti_nu.legend()
        plt.tight_layout()

        # Anti-Neutrino Difference Plot
        ax_reco_mis_positivediff = axes_diff[i, 1]
        delta_ary_anti = reco_ary_anti - truth_ary_anti
        plot_y_vs_x(x=ak.to_numpy(truth_ary_anti), y=delta_ary_anti, ax=ax_reco_mis_positivediff, band='68')
        ax_reco_mis_positivediff.set_xlabel(f'Truth Anti-Neutrino {var} [GeV]')
        ax_reco_mis_positivediff.set_ylabel(f'Error of Reconstruction')
        ax_reco_mis_positivediff.set_title(f'Anti-Neutrino {var} Reconstruction Error vs Truth for {dl_name}')
        ax_reco_mis_positivediff.set_ylim(-3, 3)
        plt.tight_layout()

        # Anti-Neutrino Relative Difference Plot
        rel_err_ary_anti = ak.to_numpy(delta_ary_anti) / ak.to_numpy(truth_ary_anti)
        ax_reco_mis_positivediff_rel = axes_diff_rel[i, 1]
        plot_y_vs_x(x=ak.to_numpy(truth_ary_anti), y=rel_err_ary_anti, ax=ax_reco_mis_positivediff_rel, band='68')
        ax_reco_mis_positivediff_rel.set_xlabel(f'Truth Anti-Neutrino {var} [GeV]')
        ax_reco_mis_positivediff_rel.set_ylabel(f'Relative Error of Reconstruction')
        ax_reco_mis_positivediff_rel.set_title(f'Anti-Neutrino {var} Relative Reconstruction Error vs Truth for {dl_name}')
        ax_reco_mis_positivediff_rel.set_ylim(-0.1, 0.1)
        plt.tight_layout()

        # Anti-Neutrino Distribution Plot
        ax_reco_mis_positivedist = axes_distribution[i, 1]
        ax_reco_mis_positivedist.hist(ak.to_numpy(truth_ary_anti), bins=30, range=plot_range, label='Truth', density=True, histtype='step', linewidth=2)
        ax_reco_mis_positivedist.hist(ak.to_numpy(reco_ary_anti), bins=30, range=plot_range, label='Reconstructed', density=True, histtype='step', linewidth=2)
        ax_reco_mis_positivedist.set_xlabel(f'Anti-Neutrino {var} (GeV)')
        ax_reco_mis_positivedist.set_ylabel('Normalized Counts')
        ax_reco_mis_positivedist.set_title(f'Anti-Neutrino {var} Distribution for {dl_name}')
        ax_reco_mis_positivedist.legend()
        plt.tight_layout()

    fig_scatter.savefig(f"{output_dir}/{dl_name}_neutrino_momentum_reconstruction_scatter.png")
    plt.close(fig_scatter)
    fig_diff.savefig(f"{output_dir}/{dl_name}_neutrino_momentum_reconstruction_difference.png")
    plt.close(fig_diff)
    fig_diff_rel.savefig(f"{output_dir}/{dl_name}_neutrino_momentum_reconstruction_relative_difference.png")
    plt.close(fig_diff_rel)
    fig_distribution.savefig(f"{output_dir}/{dl_name}_neutrino_momentum_reconstruction_distribution.png")
    plt.close(fig_distribution)

    # plot dR between reconstructed tau+ and tau-
    tau_plus_p4 = reco_vis_positive_p4 + reco_mis_positivep4_array
    tau_minus_p4 = reco_vis_negative_p4 + reco_mis_negativep4_array
    dr_tau_tau = tau_plus_p4.deltaR(tau_minus_p4)
    fig, ax = plt.subplots(dpi=300)
    ax.hist(ak.to_numpy(dr_tau_tau), bins=50, range=(0, 5), histtype='step', density=True)
    ax.set_xlabel('Delta R between Reconstructed Tau+ and Tau-')
    ax.set_ylabel('Normalized Counts')
    ax.set_title(f'Delta R Distribution for {dl_name}')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{dl_name}_deltaR_tau_tau.png'))
    plt.close(fig)

    # plot dR between reconstructed neutrino and visible tau
    dr_reco_mis_negativevis = reco_mis_negativep4_array.deltaR(reco_vis_negative_p4)
    dr_reco_mis_positivevis = reco_mis_positivep4_array.deltaR(reco_vis_positive_p4)
    fig, ax = plt.subplots(dpi=300)
    bins = np.linspace(0, 0.5, 101)
    ax.hist(ak.to_numpy(dr_reco_mis_negativevis), bins=bins, histtype='step', density=True, label='Neutrino vs Visible Tau-')
    ax.hist(ak.to_numpy(dr_reco_mis_positivevis), bins=bins, histtype='step', density=True, label='Anti-Neutrino vs Visible Tau+')
    ax.set_xlabel('Delta R between Reconstructed Neutrino and Visible Tau')
    ax.set_ylabel('Normalized Counts')
    ax.set_title(f'Delta R Distribution for {dl_name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{dl_name}_deltaR_reco_mis_negativevis.png'))
    plt.close(fig)


def _det3(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    return (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )


def _eps_normal(v1, v2, v3):
    """
    Build n_raw^mu = epsilon^{mu nu rho sigma} v1_nu v2_rho v3_sigma
    for arrays of 4-vectors with shape (num_events,).

    Returns a vector array of shape (num_events,).
    """
    t1, x1, y1, z1 = v1.E, v1.px, v1.py, v1.pz
    t2, x2, y2, z2 = v2.E, v2.px, v2.py, v2.pz
    t3, x3, y3, z3 = v3.E, v3.px, v3.py, v3.pz

    n0 = _det3(x1, y1, z1, x2, y2, z2, x3, y3, z3)
    n1 = -_det3(t1, y1, z1, t2, y2, z2, t3, y3, z3)
    n2 = _det3(t1, x1, z1, t2, x2, z2, t3, x3, z3)
    n3 = -_det3(t1, x1, y1, t2, x2, y2, t3, x3, y3)

    return vector.zip({
        "px": n1,
        "py": n2,
        "pz": n3,
        "E": n0,
    })


def compute_neutrino_momenta(
        vis1_p4: vector.Array, # (num_events,)
        vis2_p4: vector.Array, # (num_events,)
        m_miss1_grid: np.ndarray = None, # (num_events, num_grid_points)
        m_miss2_grid: np.ndarray = None, # (num_events, num_grid_points)
        choose: str = "larger_E1",
):
    """
    Reconstruct neutrino 4-momenta in:
        tau+ tau- -> (vis1 + missing1) + (vis2 + missing2)
        Assumptions:
        - No other visible particles in the event besides vis1 and vis2

    Returns:
        missing1_p4, missing2_p4, flag_valid (all in shape (num_events, num_grid_points))

    flag_valid meaning:
    0: no real physical solution
    1: exactly one physical solution (passes E>0)
    2: two physical solutions (function returns one of them deterministically)

    ============================================================================
    Steps:
    1) Define initial 4-momentum P = (cme,0,0,0). Assume p4 of neutrinos are k1, k2
    2) Define v1, v2, v3 as visible tau1, visible tau2, and Q = k1 + k2 = P - v1 - v2
    3) Define a1=v1.dot(k1), a2=v2.dot(k2), b=v2.dot(k1), c=v3.dot(k1). a, b and c can be expressed
    in terms of known quantities (m_tau, v1, v2, Q, mk1, mk2):
            a = 0.5 * (m_tau**2 - v1.dot(v1) - mk1**2))  # from tau mass constraint
            a2 = 0.5 * (m_tau**2 - v2.dot(v2) - mk2**2))  # from tau mass constraint
            b = v2.dot(Q) - a2                      # from dot with v2
            c = 0.5 * (Q.dot(Q) + mk1**2 - mk2**2)  # from Q mass constraint
    4) Build 3-vector vec(d)=vec(a,b,c).
    Build 3x3 matrix G:
            G_ij = vi.dot(vj)   for i,j=1,2,3
    5) Find a four-vector n orthogonal to v1, v2, Q and that n.dot(n) = -1
    6) Parametrize k1 as:
            k1 = k0 + beta * n, where k0 = sum_i alpha_i * v_i
        we have: v_j.dot(k1) = sum_i alpha_i * v_j.dot(v_i) = d_j
        so that:   
            vec(alpha) = G^-1.dot(vec(d))
    7) Impose k1.dot(k1)=mk1**2 to solve for beta:
            mk1**2 = k1.dot(k1) = k0^2 - beta^2
        so that:
            beta = sqrt(k0^2 - mk1**2)
    8) Finally k2 = Q - k1
    ============================================================================
    """
    num_events = len(vis1_p4)

    if m_miss1_grid is None:
        m_miss1_grid = np.zeros((num_events, 1))
    if m_miss2_grid is None:
        m_miss2_grid = np.zeros((num_events, 1))
    num_grid_points = m_miss1_grid.shape[1]

    # ----------------------------
    # Define visible vectors and initial state
    # ----------------------------
    v1 = vis1_p4
    v2 = vis2_p4
    m_tau = 1.777 

    P = vector.zip({
        "px": np.zeros(num_events),
        "py": np.zeros(num_events),
        "pz": np.zeros(num_events),
        "E":  np.full(num_events, cme),
    })

    Q = P - v1 - v2   # shape (num_events,)

    # ----------------------------
    # Precompute event-level invariants
    # ----------------------------
    v1_sq = np.asarray(v1.dot(v1))[:, None]      # (N,1)
    v2_sq = np.asarray(v2.dot(v2))[:, None]      # (N,1)
    Q_sq  = np.asarray(Q.dot(Q))[:, None]        # (N,1)
    v2_dot_Q = np.asarray(v2.dot(Q))[:, None]    # (N,1)

    m1_sq = m_miss1_grid**2
    m2_sq = m_miss2_grid**2
    m_tau_sq = m_tau**2

    # ----------------------------
    # Compute a, b, c for each grid point
    # ----------------------------
    a = 0.5 * (m_tau_sq - v1_sq - m1_sq)
    a2 = 0.5 * (m_tau_sq - v2_sq - m2_sq)
    b = v2_dot_Q - a2
    c = 0.5 * (Q_sq + m1_sq - m2_sq)

    # d[e,g,:] = [a, b, c]
    d = np.stack([a, b, c], axis=-1)   # shape (N, G, 3)

    # ----------------------------
    # Build Gram matrix G[e,i,j] = vi·vj
    # with basis (v1, v2, Q)
    # ----------------------------
    G = np.empty((num_events, 3, 3), dtype=np.float64)

    v1v1 = np.asarray(v1.dot(v1))
    v1v2 = np.asarray(v1.dot(v2))
    v1Q  = np.asarray(v1.dot(Q))
    v2v2 = np.asarray(v2.dot(v2))
    v2Q  = np.asarray(v2.dot(Q))
    QQ   = np.asarray(Q.dot(Q))

    G[:, 0, 0] = v1v1
    G[:, 0, 1] = v1v2
    G[:, 0, 2] = v1Q
    G[:, 1, 0] = v1v2
    G[:, 1, 1] = v2v2
    G[:, 1, 2] = v2Q
    G[:, 2, 0] = v1Q
    G[:, 2, 1] = v2Q
    G[:, 2, 2] = QQ

    # Invert event-by-event
    detG = np.linalg.det(G)
    invertible = np.abs(detG) > 1e-12

    Ginv = np.full_like(G, np.nan)
    if np.any(invertible):
        Ginv[invertible] = np.linalg.inv(G[invertible])

    # ----------------------------
    # alpha[e,g,i] = sum_j Ginv[e,i,j] d[e,g,j]
    # ----------------------------
    alpha = np.einsum("eij,egj->egi", Ginv, d)   # (N, G, 3)

    # ----------------------------
    # Build k0 = alpha1*v1 + alpha2*v2 + alpha3*Q
    # ----------------------------
    a1 = alpha[:, :, 0]
    a2 = alpha[:, :, 1]
    a3 = alpha[:, :, 2]

    k0_px = a1 * np.asarray(v1.px)[:, None] + a2 * np.asarray(v2.px)[:, None] + a3 * np.asarray(Q.px)[:, None]
    k0_py = a1 * np.asarray(v1.py)[:, None] + a2 * np.asarray(v2.py)[:, None] + a3 * np.asarray(Q.py)[:, None]
    k0_pz = a1 * np.asarray(v1.pz)[:, None] + a2 * np.asarray(v2.pz)[:, None] + a3 * np.asarray(Q.pz)[:, None]
    k0_E  = a1 * np.asarray(v1.E )[:, None] + a2 * np.asarray(v2.E )[:, None] + a3 * np.asarray(Q.E )[:, None]

    # k0^2
    k0_sq = k0_E**2 - k0_px**2 - k0_py**2 - k0_pz**2

    # For general invisible mass: beta^2 = k0^2 - m1^2
    beta2 = k0_sq - m1_sq

    # ----------------------------
    # Build normal n orthogonal to (v1, v2, Q), normalize to n^2 = -1
    # ----------------------------
    n_raw = _eps_normal(v1, v2, Q)  # shape (N,)

    n_raw_sq = np.asarray(n_raw.dot(n_raw))   # should be < 0 for spacelike normal
    good_normal = n_raw_sq < -1e-12

    norm = np.full(num_events, np.nan, dtype=np.float64)
    norm[good_normal] = np.sqrt(-n_raw_sq[good_normal])

    nx = np.asarray(n_raw.px) / norm
    ny = np.asarray(n_raw.py) / norm
    nz = np.asarray(n_raw.pz) / norm
    nE = np.asarray(n_raw.E ) / norm

    # Broadcast n to (N,G)
    nx = nx[:, None]
    ny = ny[:, None]
    nz = nz[:, None]
    nE = nE[:, None]

    # ----------------------------
    # Real solutions require beta2 >= 0
    # ----------------------------
    tol = 1e-10
    beta2_clipped = np.where(beta2 > 0.0, beta2, 0.0)
    beta = np.sqrt(beta2_clipped)

    real_solution = beta2 >= -tol
    good_event_basis = invertible[:, None] & good_normal[:, None]

    # ----------------------------
    # Two candidate solutions
    # ----------------------------
    k1p_px = k0_px + beta * nx
    k1p_py = k0_py + beta * ny
    k1p_pz = k0_pz + beta * nz
    k1p_E  = k0_E  + beta * nE

    k1m_px = k0_px - beta * nx
    k1m_py = k0_py - beta * ny
    k1m_pz = k0_pz - beta * nz
    k1m_E  = k0_E  - beta * nE

    Qpx = np.asarray(Q.px)[:, None]
    Qpy = np.asarray(Q.py)[:, None]
    Qpz = np.asarray(Q.pz)[:, None]
    QE  = np.asarray(Q.E )[:, None]

    k2p_px = Qpx - k1p_px
    k2p_py = Qpy - k1p_py
    k2p_pz = Qpz - k1p_pz
    k2p_E  = QE  - k1p_E

    k2m_px = Qpx - k1m_px
    k2m_py = Qpy - k1m_py
    k2m_pz = Qpz - k1m_pz
    k2m_E  = QE  - k1m_E

    # ----------------------------
    # Physical checks
    # ----------------------------
    k1p_sq = k1p_E**2 - k1p_px**2 - k1p_py**2 - k1p_pz**2
    k2p_sq = k2p_E**2 - k2p_px**2 - k2p_py**2 - k2p_pz**2
    k1m_sq = k1m_E**2 - k1m_px**2 - k1m_py**2 - k1m_pz**2
    k2m_sq = k2m_E**2 - k2m_px**2 - k2m_py**2 - k2m_pz**2

    mass_tol = 1e-6

    valid_plus = (
        good_event_basis
        & real_solution
        & (k1p_E > 0.0)
        & (k2p_E > 0.0)
        & (np.abs(k1p_sq - m1_sq) < mass_tol)
        & (np.abs(k2p_sq - m2_sq) < mass_tol)
    )

    valid_minus = (
        good_event_basis
        & real_solution
        & (k1m_E > 0.0)
        & (k2m_E > 0.0)
        & (np.abs(k1m_sq - m1_sq) < mass_tol)
        & (np.abs(k2m_sq - m2_sq) < mass_tol)
    )

    # ----------------------------
    # Flag
    # ----------------------------
    n_valid = valid_plus.astype(np.int8) + valid_minus.astype(np.int8)
    flag_valid = np.where(n_valid == 0, 0, np.where(n_valid == 1, 1, 2)).astype(np.int8)

    # ----------------------------
    # Deterministic choice when both exist
    # ----------------------------
    if choose == "larger_E1":
        choose_plus_when_both = k1p_E >= k1m_E
    elif choose == "smaller_abs_beta":
        # same |beta| for +/- by construction, so this does not separate them;
        # kept here only for interface completeness
        choose_plus_when_both = np.ones_like(k1p_E, dtype=bool)
    else:
        raise ValueError(f"Unknown choose mode: {choose}")

    choose_plus = (
        (valid_plus & ~valid_minus)
        | (valid_plus & valid_minus & choose_plus_when_both)
    )
    choose_minus = (
        (valid_minus & ~valid_plus)
        | (valid_plus & valid_minus & ~choose_plus_when_both)
    )

    # ----------------------------
    # Fill chosen solution; NaN for invalid
    # ----------------------------
    miss1_px = np.full_like(k0_px, np.nan)
    miss1_py = np.full_like(k0_py, np.nan)
    miss1_pz = np.full_like(k0_pz, np.nan)
    miss1_E  = np.full_like(k0_E , np.nan)

    miss2_px = np.full_like(k0_px, np.nan)
    miss2_py = np.full_like(k0_py, np.nan)
    miss2_pz = np.full_like(k0_pz, np.nan)
    miss2_E  = np.full_like(k0_E , np.nan)

    miss1_px[choose_plus] = k1p_px[choose_plus]
    miss1_py[choose_plus] = k1p_py[choose_plus]
    miss1_pz[choose_plus] = k1p_pz[choose_plus]
    miss1_E [choose_plus] = k1p_E [choose_plus]

    miss2_px[choose_plus] = k2p_px[choose_plus]
    miss2_py[choose_plus] = k2p_py[choose_plus]
    miss2_pz[choose_plus] = k2p_pz[choose_plus]
    miss2_E [choose_plus] = k2p_E [choose_plus]

    miss1_px[choose_minus] = k1m_px[choose_minus]
    miss1_py[choose_minus] = k1m_py[choose_minus]
    miss1_pz[choose_minus] = k1m_pz[choose_minus]
    miss1_E [choose_minus] = k1m_E [choose_minus]

    miss2_px[choose_minus] = k2m_px[choose_minus]
    miss2_py[choose_minus] = k2m_py[choose_minus]
    miss2_pz[choose_minus] = k2m_pz[choose_minus]
    miss2_E [choose_minus] = k2m_E [choose_minus]

    miss1_p4 = vector.zip({
        "px": miss1_px,
        "py": miss1_py,
        "pz": miss1_pz,
        "E":  miss1_E,
    })

    miss2_p4 = vector.zip({
        "px": miss2_px,
        "py": miss2_py,
        "pz": miss2_pz,
        "E":  miss2_E,
    })

    if m_miss1_grid.shape[1] == 1 and m_miss2_grid.shape[1] == 1:
        # Squeeze out grid dimension if it was not used
        miss1_p4 = ak.firsts(miss1_p4)
        miss2_p4 = ak.firsts(miss2_p4)
        flag_valid = flag_valid[:, 0]

    return miss1_p4, miss2_p4, flag_valid

class NeutrinoReconstructionProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        super().__init__(config)
        self.config = config
        self.output_dir_name = self.config.get('output_dir_name', 'neutrino_reconstruction')
        self.output_dir = f"{output_dir}/{self.output_dir_name}/"
        self.dl_to_load = self.config.get('dl_to_load', []) 
        self.regions = self.config.get('regions', [])
        os.makedirs(self.output_dir, exist_ok=True)
        self.fields_to_add = [
            'lead_a_missing_p4', 'lead_b_missing_p4', 
            'reco_tau_a_p4', 'reco_tau_b_p4',
            'flags_valid', 'mmc_likelihood' 
        ] + get_observable_names()

        # Instantiate the new MMC Engine Wrapper
        self.mmc_regions = self.config.get('mmc_regions', [])
        if len(self.mmc_regions) > 0:
            self.mmc_engine = MMC(self.config)

    def run(self, dl_dict):
        requested = self.dl_to_load
        dl_to_load = list(dl_dict.keys()) if len(requested) == 0 else requested
        for region_name in self.regions:
            for dl_name in dl_to_load:
                if dl_name not in dl_dict:
                    print(f"[WARN] Requested dataloader '{dl_name}' not found. Skip.")
                    continue
                dl = dl_dict[dl_name]
                cur_output_dir = f"{self.output_dir}/{region_name}/"
                output_file = f"{cur_output_dir}/{dl_name}_reconstructed_neutrinos.parquet"

                solution_loaded = False
                if os.path.exists(output_file):
                    events = ak.from_parquet(output_file)
                    solution_loaded = True
                    for key in self.fields_to_add:
                        if key not in events.fields or len(events[key]) != len(dl.data[region_name]):
                            print(f"Field {key} not found or length mismatch in loaded file for {dl_name} in region {region_name}. Recomputing neutrino reconstruction.")
                            solution_loaded = False
                            break
                        if key.endswith('_p4'):
                            events[key] = rebuild_p4(events[key])

                if not solution_loaded:
                    if not os.path.exists(cur_output_dir):
                        os.makedirs(cur_output_dir, exist_ok=True)
                    if region_name not in dl.data:
                        raise ValueError(f"Region {region_name} not found in dataloader {dl_name}. Available regions: {list(dl.data.keys())}")

                    events = dl.data.get(region_name)
                    print(f"Processing {dl_name} for region {region_name} with {len(events)} events")

                    if len(events) == 0:
                        print(f"  -> No events in {dl_name}/{region_name}. Skipping reconstruction and saving empty output.")
                        ak.to_parquet(events, output_file, compression='snappy')
                        continue

                    reco_vis_positive_p4 = events['lead_a_visible_p4']
                    reco_vis_negative_p4 = events['lead_b_visible_p4']

                    # -------------------------------------------------
                    # Reconstruct neutrino momenta using either MMC or algebraic method, depending on the region
                    # -------------------------------------------------
                    reco_mis_negativep4_array, reco_mis_positivep4_array, flags_valid_array, mmc_likelihood = None, None, None, None
                    if region_name in self.mmc_regions:
                        print(f"  -> Routing to MMC Engine for {region_name}...")
                        reco_mis_negativep4_array, reco_mis_positivep4_array, flags_valid_array, mmc_likelihood = self.mmc_engine.calculate(
                            vis1_p4=reco_vis_negative_p4,
                            vis2_p4=reco_vis_positive_p4,
                            region_name=region_name,
                            events=events
                        )
                    else:
                        print(f"  -> Routing to Algebraic Neutrino Reconstruction for {region_name}...")
                        reco_mis_negativep4_array, reco_mis_positivep4_array, flags_valid_array = compute_neutrino_momenta(
                            vis1_p4=reco_vis_negative_p4,
                            vis2_p4=reco_vis_positive_p4,
                        )
                        mmc_likelihood = np.zeros(len(events))  # Placeholder likelihood for non-MMC reconstruction

                    events[f'lead_a_missing_p4'] = reco_mis_positivep4_array
                    events[f'lead_b_missing_p4'] = reco_mis_negativep4_array
                    events['flags_valid'] = flags_valid_array
                    events['mmc_likelihood'] = mmc_likelihood 

                    for key in ['a', 'b']:
                        events[f'reco_tau_{key}_p4'] = events[f'lead_{key}_visible_p4'] + events[f'lead_{key}_missing_p4']
                    
                    # derive the observables for QI study
                    observables = build_observables(tau_a_p4=events['reco_tau_a_p4'], tau_b_p4=events['reco_tau_b_p4'], vis_a_p4=events['lead_a_visible_p4'], vis_b_p4=events['lead_b_visible_p4'])
                    for obs_name, obs_values in observables.items():
                        events[obs_name] = obs_values

                    if 'ztautau' in dl_name.lower():
                        output_dir_for_plots = f"{cur_output_dir}/plots/"
                        os.makedirs(output_dir_for_plots, exist_ok=True)
                        evaluate_reconstruction(events, dl_name, output_dir_for_plots)

                        # compare truth vs reconstructed observables
                        output_dir_for_comparison_plots = f"{cur_output_dir}/QI_observables/"
                        os.makedirs(output_dir_for_comparison_plots, exist_ok=True)
                        valid_events = events[events['flags_valid'] > 0]
                        evaluate_observables(valid_events, dl_name, output_dir_for_comparison_plots)

                    ak.to_parquet(events, output_file, compression='snappy')
                    print(f"Saved reconstructed neutrino data to {output_file} for {dl_name}")

                for key in self.fields_to_add:
                    if key in events.fields:
                        dl.data[region_name][key] = events[key]
                    else:
                        print(f"[WARN] Missing field '{key}' for {dl_name}/{region_name}; skip write-back for this field.")

    def finalize(self):
        pass