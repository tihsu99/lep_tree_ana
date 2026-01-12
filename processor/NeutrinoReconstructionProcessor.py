import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import copy
import vector
import awkward as ak
import tqdm
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme, m_tau
from utils.plotter import plot_y_vs_x

def decode_event_category(cat: int, order_sensitive=False):
    """
    Decode event category integer to human-readable string.
    """
    particle_category_map = {
        0: 'NoneTau',
        1: 'Pi',
        2: 'Rho',
        3: 'Lepton',
        4: 'Others'
    }

    tau_plus_cat = cat // 10
    tau_minus_cat = cat % 10

    if order_sensitive:
        return f"{particle_category_map.get(tau_minus_cat)}_{particle_category_map.get(tau_plus_cat)}"
    else:
        first, second = sorted([tau_plus_cat, tau_minus_cat])
        return f"{particle_category_map.get(first)}_{particle_category_map.get(second)}"


def calculate_neutrino_p4(
        tau1_vis_p4,
        tau2_vis_p4, 
        cme=cme, 
        tol_mass2=1e-6,      # GeV^2 tolerance for massless check
        tol_beta2=1e-10      # allow tiny negative due to numerics
    ):
    """
    Reconstruct neutrino 4-momenta in:
      tau+ tau- -> (pi- nu) (pi+ anti-nu)

    Inputs:
      tau1_vis_p4, tau2_vis_p4: vector package 4-vectors (t,x,y,z) with t=E
      cme: sqrt(s) in GeV (assumed total initial 4-momentum is (cme,0,0,0))

    Returns:
      nu_p4, anti_nu_p4, flag_valid

    flag_valid meaning:
      0: no real physical solution
      1: exactly one physical solution (passes E>0)
      2: two physical solutions (function returns one of them deterministically)

    ============================================================================
    Steps:
    1) Define initial 4-momentum P = (cme,0,0,0). Assume p4 of neutrinos are k1, k2
    2) Define v1, v2, v3 as visible tau1, visible tau2, and Q = k1 + k2 = P - v1 - v2
    3) Define a=v1.dot(k1), b=v2.dot(k1), c=v3.dot(k1). a, b and c can be expressed
       in terms of known quantities (m_tau, v1, v2, Q):
            a = 0.5 * (m_tau**2 - v1.dot(v1))
            b = v2.dot(Q) - a
            c = 0.5 * Q.dot(Q)
    4) Build 3-vector vec(d)=vec(a,b,c).
       Build 3x3 matrix G:
            G_ij = vi.dot(vj)   for i,j=1,2,3
    5) Find a four-vector n orthogonal to v1, v2, Q and that n.dot(n) = -1
    6) Parametrize k1 as:
            k1 = vec(alpha).dot(vec(v)) + beta * n
        we have: v_j.dot(k1) = sum_i alpha_i * v_j.dot(v_i) = d_j
        so that:   
            vec(alpha) = G^-1.dot(vec(d))
    7) Impose k1.dot(k1)=0 to solve for beta:
            0 = k1.dot(k1) = (sum_i alpha_i v_i)^2 - beta^2
        so that:
            beta = sqrt((sum_i alpha_i v_i)^2)
    8) Finally k2 = Q - k1
    ============================================================================
    """
    failed_solution = vector.obj(E=0.0, px=0.0, py=0.0, pz=0.0)

    # Initial state in CM
    P = vector.obj(E=float(cme), px=0.0, py=0.0, pz=0.0)

    p1 = tau1_vis_p4
    p2 = tau2_vis_p4

    # Q = k1 + k2
    Q = P - p1 - p2

    # Constraints
    a = 0.5 * (m_tau * m_tau - p1.dot(p1))
    b = p2.dot(Q) - a
    c = 0.5 * Q.dot(Q)

    v1, v2, v3 = p1, p2, Q

    # Build G matrix G_ij = vi.dot(vj)
    G = np.array([
        [v1.dot(v1), v1.dot(v2), v1.dot(v3)],
        [v2.dot(v1), v2.dot(v2), v2.dot(v3)],
        [v3.dot(v1), v3.dot(v2), v3.dot(v3)],
    ])
    d = np.array([a, b, c])

    # solve for alpha which satisfies G * alpha = d
    try:
        alpha = np.linalg.solve(G, d)
    except np.linalg.LinAlgError:
        print("Warning: encountered singular matrix in neutrino reconstruction.")
        # singular matrix, no solution
        return failed_solution, failed_solution, 0

    # solve for beta
    tmp = alpha[0] * v1 + alpha[1] * v2 + alpha[2] * v3
    beta2 = tmp.dot(tmp)
    # if beta2 < -tol_beta2:
    #     print("Warning: no real solution for neutrino reconstruction (beta^2 < 0).")
    #     # no real solution
    #     return failed_solution, failed_solution, 0

    beta = np.sqrt(max(beta2, 0.0))

    # build n orthogonal to v1, v2, v3
    n0 = np.linalg.det(np.array([
        [v1.x, v1.y, v1.z],
        [v2.x, v2.y, v2.z],
        [v3.x, v3.y, v3.z],
    ]))
    n1 = -np.linalg.det(np.array([
        [v1.t, v1.y, v1.z],
        [v2.t, v2.y, v2.z],
        [v3.t, v3.y, v3.z],
    ]))
    n2 = np.linalg.det(np.array([
        [v1.t, v1.x, v1.z],
        [v2.t, v2.x, v2.z],
        [v3.t, v3.x, v3.z],
    ]))
    n3 = -np.linalg.det(np.array([
        [v1.t, v1.x, v1.y],
        [v2.t, v2.x, v2.y],
        [v3.t, v3.x, v3.y],
    ]))
    n = vector.obj(E=n0, px=n1, py=n2, pz=n3)
    n = n / np.sqrt(-n.dot(n))  # normalize to -1

    # two possible solutions for k1
    k1_sol1 = alpha[0] * v1 + alpha[1] * v2 + alpha[2] * v3 + beta * n
    k1_sol2 = alpha[0] * v1 + alpha[1] * v2 + alpha[2] * v3 - beta * n

    # corresponding k2
    k2_sol1 = Q - k1_sol1
    k2_sol2 = Q - k1_sol2

    # check validity of solutions (E>0)
    def is_physical(k1, k2):
        if not(k1.t > 0 and k2.t > 0):
            return False
        # # check massless within tolerance
        # if abs(k1.dot(k1)) > tol_mass2 or abs(k2.dot(k2)) > tol_mass2:
        #     return False
        return True

    valid1 = is_physical(k1_sol1, k2_sol1)
    valid2 = is_physical(k1_sol2, k2_sol2)

    if valid1 and valid2:
        # return first solution deterministically
        return k1_sol1, k2_sol1, 2
    elif valid1:
        return k1_sol1, k2_sol1, 1
    elif valid2:
        return k1_sol2, k2_sol2, 1
    else:
        print("Warning: no physical solution for neutrino reconstruction (E>0 and massless).")
        return failed_solution, failed_solution, 0



class NeutrinoReconstructionProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        """
        Processor to make control plots for data/MC comparison.
        """
        super().__init__(config)
        self.config = config
        if config and 'output_dir_name' in config:
            output_dir = f"{output_dir}/{config['output_dir_name']}"
        else:
            output_dir = f"{output_dir}/neutrino_reconstruction/"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dl_dict):
        # only load Ztautau samples for now
        dl_to_load = [key for key, dl in dl_dict.items() if ('Ztautau' in key and (not dl.is_data))]
        for key in dl_to_load:
            dl = dl_dict[key]
            events = dl.data.get(dl.region_of_interest)
            print(f"Number of events in {key} before neutrino reconstruction: {len(events)}")

            # plot truth category
            truth_categories = ak.to_numpy(events.event_category)
            unique, counts = np.unique(truth_categories, return_counts=True)
            category_dict = {
                decode_event_category(k): v for k, v in zip(unique, counts)
            }
            print(f"Truth category distribution for {key}: {category_dict}")
            plt.figure(figsize=(8,6), dpi=300)
            plt.bar(category_dict.keys(), category_dict.values())
            plt.xlabel('Truth Category')
            plt.xticks(rotation=45)
            plt.ylabel('Counts')
            plt.title(f'Truth Category Distribution for {key}')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{key}_truth_category_distribution.png")
            plt.close()

            # Concentrate on PiPi category for now
            pi_pi_events = events[events.event_category == 11]
            # pi_pi_events = pi_pi_events[:500]  # limit to first 500 events for speed
            events_of_interest = pi_pi_events
            print(f"Number of PiPi events in {key}: {len(pi_pi_events)}")

            # Get truth neutrino p4
            flag_truth_nu = (events_of_interest['GenPart_pdgId'] == 16)
            flag_truth_anti_nu = (events_of_interest['GenPart_pdgId'] == -16)
            truth_nu_p4 = get_p4_from_ak_events(events_of_interest, flag_truth_nu, prefix='GenPart_vector')
            truth_anti_nu_p4 = get_p4_from_ak_events(events_of_interest, flag_truth_anti_nu, prefix='GenPart_vector')

            # Get visible tau p4
            flag_pi_plus = (events_of_interest['Part_charge'] == 1) & (abs(events_of_interest['Part_pdgId']) == 41)
            flag_pi_minus = (events_of_interest['Part_charge'] == -1) & (abs(events_of_interest['Part_pdgId']) == 41)
            reco_pi_plus_p4 = get_p4_from_ak_events(events_of_interest, flag_pi_plus, prefix='Part_fourMomentum')
            reco_pi_minus_p4 = get_p4_from_ak_events(events_of_interest, flag_pi_minus, prefix='Part_fourMomentum')

            # Reconstruct neutrinos
            nu_p4_list = {'px': [], 'py': [], 'pz': [], 'E': []}
            anti_nu_p4_list = {'px': [], 'py': [], 'pz': [], 'E': []}
            flags_valid = []

            for i in tqdm.tqdm(range(len(events_of_interest)), desc=f"Reconstructing neutrinos for {key}"):
                nu_p4, anti_nu_p4, flag_valid = calculate_neutrino_p4(
                    tau1_vis_p4=reco_pi_minus_p4[i],
                    tau2_vis_p4=reco_pi_plus_p4[i],
                )
                nu_p4_list['px'].append(nu_p4.px)
                nu_p4_list['py'].append(nu_p4.py)
                nu_p4_list['pz'].append(nu_p4.pz)
                nu_p4_list['E'].append(nu_p4.E)
                anti_nu_p4_list['px'].append(anti_nu_p4.px)
                anti_nu_p4_list['py'].append(anti_nu_p4.py)
                anti_nu_p4_list['pz'].append(anti_nu_p4.pz)
                anti_nu_p4_list['E'].append(anti_nu_p4.E)
                flags_valid.append(flag_valid)
            nu_p4_array = vector.zip(nu_p4_list)
            anti_nu_p4_array = vector.zip(anti_nu_p4_list)
            flags_valid_array = np.array(flags_valid)
            
            # plot comparison of reconstructed vs truth neutrinos
            fig_scatter, axes_scatter = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
            fig_diff, axes_diff = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
            fig_diff_rel, axes_diff_rel = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
            fig_distribution, axes_distribution = plt.subplots(4, 2, figsize=(12, 5*4), dpi=300)
            for i, var in enumerate(['px', 'py', 'pz', 'E']):
                plot_range = [-cme/2, cme/2]
                if var == 'E':
                    plot_range = [0, cme]
                #################
                # Neutrino
                #################
                # Neutrino Scatter Plot
                ax_nu = axes_scatter[i, 0]
                truth_ary = getattr(truth_nu_p4, var)
                reco_ary = getattr(nu_p4_array, var)
                ax_nu.scatter(ak.to_numpy(truth_ary), ak.to_numpy(reco_ary), alpha=0.8, label=f'Reconstructed {var}', s=3)
                ax_nu.plot(plot_range, plot_range, 'r--', label='Ideal')
                ax_nu.set_xlabel(f'Truth Neutrino {var} (GeV)')
                ax_nu.set_ylabel(f'Reconstructed Neutrino {var} (GeV)')
                ax_nu.set_title(f'Neutrino {var} Reconstruction for {key}')
                ax_nu.legend()

                # Neutrino Difference Plot
                ax_nu_diff = axes_diff[i, 0]
                delta_ary = ak.to_numpy(reco_ary - truth_ary)
                plot_y_vs_x(x=ak.to_numpy(truth_ary), y=delta_ary, ax=ax_nu_diff, band='68')
                ax_nu_diff.set_xlabel(f'Truth Neutrino {var} [GeV]')
                ax_nu_diff.set_ylabel(f'Error of Reconstruction')
                ax_nu_diff.set_title(f'Neutrino {var} Reconstruction Error vs Truth for {key}')
                ax_nu_diff.set_ylim(-3, 3)

                # Neutrino Relative Difference Plot
                rel_err_ary = (delta_ary) / ak.to_numpy(truth_ary)
                ax_nu_diff_rel = axes_diff_rel[i, 0]
                plot_y_vs_x(x=ak.to_numpy(truth_ary), y=rel_err_ary, ax=ax_nu_diff_rel, band='68')
                ax_nu_diff_rel.set_xlabel(f'Truth Neutrino {var} [GeV]')
                ax_nu_diff_rel.set_ylabel(f'Relative Error of Reconstruction')
                ax_nu_diff_rel.set_title(f'Neutrino {var} Relative Reconstruction Error vs Truth for {key}')
                ax_nu_diff_rel.set_ylim(-0.1, 0.1)

                # Neutrino Distribution Plot
                ax_nu_dist = axes_distribution[i, 0]
                ax_nu_dist.hist(ak.to_numpy(truth_ary), bins=30, range=plot_range, label='Truth', density=True, histtype='step', linewidth=2)
                ax_nu_dist.hist(ak.to_numpy(reco_ary), bins=30, range=plot_range, label='Reconstructed', density=True, histtype='step', linewidth=2)
                ax_nu_dist.set_xlabel(f'Neutrino {var} (GeV)')
                ax_nu_dist.set_ylabel('Normalized Counts')
                ax_nu_dist.set_title(f'Neutrino {var} Distribution for {key}')
                ax_nu_dist.legend()


                #################
                # Anti-Neutrino
                #################
                # Anti-Neutrino Scatter Plot
                ax_anti_nu = axes_scatter[i, 1]
                truth_ary_anti = getattr(truth_anti_nu_p4, var)
                reco_ary_anti = getattr(anti_nu_p4_array, var)
                ax_anti_nu.scatter(ak.to_numpy(truth_ary_anti), ak.to_numpy(reco_ary_anti), alpha=0.8, label=f'Reconstructed {var}', s=3)
                ax_anti_nu.plot(plot_range, plot_range, 'r--', label='Ideal')
                ax_anti_nu.set_xlabel(f'Truth Anti-Neutrino {var} (GeV)')
                ax_anti_nu.set_ylabel(f'Reconstructed Anti-Neutrino {var} (GeV)')
                ax_anti_nu.set_title(f'Anti-Neutrino {var} Reconstruction for {key}')
                ax_anti_nu.legend()
                plt.tight_layout()

                # Anti-Neutrino Difference Plot
                ax_anti_nu_diff = axes_diff[i, 1]
                delta_ary_anti = reco_ary_anti - truth_ary_anti
                plot_y_vs_x(x=ak.to_numpy(truth_ary_anti), y=delta_ary_anti, ax=ax_anti_nu_diff, band='68')
                ax_anti_nu_diff.set_xlabel(f'Truth Anti-Neutrino {var} [GeV]')
                ax_anti_nu_diff.set_ylabel(f'Error of Reconstruction')
                ax_anti_nu_diff.set_title(f'Anti-Neutrino {var} Reconstruction Error vs Truth for {key}')
                ax_anti_nu_diff.set_ylim(-3, 3)
                plt.tight_layout()

                # Anti-Neutrino Relative Difference Plot
                rel_err_ary_anti = ak.to_numpy(delta_ary_anti) / ak.to_numpy(truth_ary_anti)
                ax_anti_nu_diff_rel = axes_diff_rel[i, 1]
                plot_y_vs_x(x=ak.to_numpy(truth_ary_anti), y=rel_err_ary_anti, ax=ax_anti_nu_diff_rel, band='68')
                ax_anti_nu_diff_rel.set_xlabel(f'Truth Anti-Neutrino {var} [GeV]')
                ax_anti_nu_diff_rel.set_ylabel(f'Relative Error of Reconstruction')
                ax_anti_nu_diff_rel.set_title(f'Anti-Neutrino {var} Relative Reconstruction Error vs Truth for {key}')
                ax_anti_nu_diff_rel.set_ylim(-0.1, 0.1)
                plt.tight_layout()

                # Anti-Neutrino Distribution Plot
                ax_anti_nu_dist = axes_distribution[i, 1]
                ax_anti_nu_dist.hist(ak.to_numpy(truth_ary_anti), bins=30, range=plot_range, label='Truth', density=True, histtype='step', linewidth=2)
                ax_anti_nu_dist.hist(ak.to_numpy(reco_ary_anti), bins=30, range=plot_range, label='Reconstructed', density=True, histtype='step', linewidth=2)
                ax_anti_nu_dist.set_xlabel(f'Anti-Neutrino {var} (GeV)')
                ax_anti_nu_dist.set_ylabel('Normalized Counts')
                ax_anti_nu_dist.set_title(f'Anti-Neutrino {var} Distribution for {key}')
                ax_anti_nu_dist.legend()
                plt.tight_layout()

            fig_scatter.savefig(f"{self.output_dir}/{key}_neutrino_momentum_reconstruction_scatter.png")
            plt.close(fig_scatter)
            fig_diff.savefig(f"{self.output_dir}/{key}_neutrino_momentum_reconstruction_difference.png")
            plt.close(fig_diff)
            fig_diff_rel.savefig(f"{self.output_dir}/{key}_neutrino_momentum_reconstruction_relative_difference.png")
            plt.close(fig_diff_rel)
            fig_distribution.savefig(f"{self.output_dir}/{key}_neutrino_momentum_reconstruction_distribution.png")
            plt.close(fig_distribution)
            


    def finalize(self):
        pass