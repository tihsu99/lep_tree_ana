# Adapted from https://gitlab.cern.ch/yulei/ztautauml
import pandas as pd
import awkward as ak
import vector
import numpy as np
from itertools import product

from pandas import DataFrame
from scipy.linalg import sqrtm, eig
from collections import namedtuple

from utils.common_functions import get_p4_from_ak_events, get_sum_p4_from_ak_events

Hist = namedtuple('Hist', ['bin_edges', 'values', 'errors'])
ValueWithUncertainty = namedtuple('ValueWithUncertainty', ['value', 'err_up', 'err_down'])
AnalyzingPowerAry = np.array([0, 1, 0.41, -0.33, -0.34, 0]) # order: [notTauDecay, pi, rho, el, mu, others]
NominalBCValues = {
    'B_An': 0.0,
    'B_Ar': 0.0,
    'B_Ak': 0.1444,
    'B_Bn': 0.0,
    'B_Br': 0.0,
    'B_Bk': 0.1474,
    'C_nn': 0.8446,
    'C_rr': -0.7971,
    'C_kk': 1.0345,
    'C_nr': 0.0,
    'C_nk': 0.0,
    'C_rk': 0.0,
    'C_rn': 0.0,
    'C_kn': 0.0,
    'C_kr': 0.0,
}

def get_analyzing_power_ary():
    # order: [notTauDecay, pi, rho, el, mu, others]
    return AnalyzingPowerAry


def reweight_correlation(ary_observable, ary_spin_analyzing_power, weight, element_name, variation):
    assert len(ary_observable) == len(weight)
    assert len(ary_observable) == len(ary_spin_analyzing_power)
    assert element_name in NominalBCValues, f"Element name {element_name} not found in nominal BC values"
    original_value = NominalBCValues[element_name]
    target_value = original_value + variation
    scale_factor = (1 + target_value * ary_spin_analyzing_power * ary_observable) / (1 + original_value * ary_spin_analyzing_power * ary_observable)
    new_weight = weight * scale_factor
    return new_weight


def shift_SDM_element(events, element_name, variation):
    observable_name, analyzing_power = None, None
    if element_name.startswith('C_'):
        observable_name = f'truth_cos_theta_A_{element_name[-2]}_times_cos_theta_B_{element_name[-1]}'
        analyzing_power = ak.to_numpy(events[f'analyzing_power_a'])*(-1) * ak.to_numpy(events[f'analyzing_power_b'])
    elif element_name.startswith('B_'):
        target_object = element_name[-2].upper()  # 'A' or 'B'
        axis = element_name[-1].lower()  # 'n', 'r', or 'k'
        observable_name = f'truth_cos_theta_{target_object}_{axis}'
        sign = -1 if target_object == 'A' else 1
        analyzing_power = ak.to_numpy(events[f'analyzing_power_{target_object.lower()}'] * sign)
    else:
        raise ValueError(f"Unknown SDM element name: {element_name}")
    
    new_weight = reweight_correlation(
        ary_observable = ak.to_numpy(events[observable_name]),
        ary_spin_analyzing_power = analyzing_power,
        weight = ak.to_numpy(events['weight_nominal']),
        element_name = element_name,
        variation = variation
    )
    return new_weight



def get_mean_and_err_of_mean(x, weights=None, err=None):
    weights = weights if weights is not None else np.ones_like(x)
    if weights.sum() == 0:
        return 0, 0
    err = err if err is not None else weights**0.5
    mean = np.average(x, weights=weights)
    err_of_mean = np.sqrt(np.sum( ((x - mean) / np.sum(weights) * err)**2))
    return mean, err_of_mean

def get_observable_names():
    """
    Get the names of the observables that we will build
    """
    observable_names = ['theta_cm', 'mtautau']
    observable_names += [f'cos_theta_A_{axis}' for axis in ['n', 'r', 'k']]
    observable_names += [f'cos_theta_B_{axis}' for axis in ['n', 'r', 'k']]
    for axis_a, axis_b in product(['n', 'r', 'k'], repeat=2):
        observable_names.append(f'cos_theta_A_{axis_a}_times_cos_theta_B_{axis_b}')
    return observable_names

def helicity_basis(particle: vector.Vector):
    """
    Helicity basis: https://arxiv.org/pdf/2305.07075
    Returns the helicity basis for a given particle
    """

    k_hat = particle.to_pxpypz().unit()

    # Define beam direction
    p_hat = vector.Vector(x=0, y=0, z=1)
    y_p = p_hat.dot(k_hat)
    r_p = np.sqrt(1 - y_p ** 2)

    r_hat = 1 / r_p * (p_hat - y_p * k_hat)
    r_hat = r_hat.unit()

    n_hat = r_hat.cross(k_hat)
    n_hat = n_hat.unit()

    return {"k": k_hat, "r": r_hat, "n": n_hat}


def build_observables(tau_a_p4, tau_b_p4, vis_a_p4, vis_b_p4):
    """
    Build the following observables for each event:
        - cos_theta_A_n, cos_theta_A_r, cos_theta_A_k
        - cos_theta_B_n, cos_theta_B_r, cos_theta_B_k
        where A and B are the two visible particles, and n, r, k are the helicity basis vectors defined in the rest frame of the tau a.
    """
    cm_p4 = tau_a_p4 + tau_b_p4
    boost_to_cm = -cm_p4.to_beta3()

    # boost all relevant 4-vectors to the CM frame
    tau_a_p4_cm = tau_a_p4.boost(boost_to_cm)
    tau_b_p4_cm = tau_b_p4.boost(boost_to_cm)
    vis_a_p4_cm = vis_a_p4.boost(boost_to_cm)
    vis_b_p4_cm = vis_b_p4.boost(boost_to_cm)

    # define helicity basis in cm frame using tau a momentum
    helicity_basis_a = helicity_basis(tau_a_p4_cm)

    # boost visible momenta to tau a rest frame
    boost_to_a_rest = -tau_a_p4_cm.to_beta3()
    vis_a_p4_a_rest = vis_a_p4_cm.boost(boost_to_a_rest)

    # boost visible momenta to tau b rest frame
    boost_to_b_rest = -tau_b_p4_cm.to_beta3()
    vis_b_p4_b_rest = vis_b_p4_cm.boost(boost_to_b_rest)

    # Boost the common CM-frame helicity basis into each tau rest frame separately.
    # The A and B observables must use the basis vectors evaluated in their own rest frames.
    helicity_basis_a_rest = {}
    helicity_basis_b_rest = {}
    for axis in ['n', 'r', 'k']:
        axis_vector = helicity_basis_a[axis]
        helicity_basis_tmp = vector.zip({
            "x": axis_vector.x,
            "y": axis_vector.y,
            "z": axis_vector.z,
            "t": np.zeros_like(axis_vector.x),
        })
        helicity_basis_a_rest[axis] = helicity_basis_tmp.boost(boost_to_a_rest).to_pxpypz().unit()
        helicity_basis_b_rest[axis] = helicity_basis_tmp.boost(boost_to_b_rest).to_pxpypz().unit()


    observables = {}
    observables['theta_cm'] = np.arccos(abs(tau_a_p4_cm.costheta)) * 2 / np.pi
    observables['mtautau'] = cm_p4.mass
    for axis in ['n', 'r', 'k']:
        observables[f'cos_theta_A_{axis}'] = vis_a_p4_a_rest.to_pxpypz().unit().dot(helicity_basis_a_rest[axis])
        observables[f'cos_theta_B_{axis}'] = vis_b_p4_b_rest.to_pxpypz().unit().dot(helicity_basis_b_rest[axis])
    
    # product observables
    for axis_a, axis_b in product(['n', 'r', 'k'], repeat=2):
        observables[f'cos_theta_A_{axis_a}_times_cos_theta_B_{axis_b}'] = observables[f'cos_theta_A_{axis_a}'] * observables[f'cos_theta_B_{axis_b}']

    return observables


def compute_full_density_matrix(C: dict[str, float], B: dict[str, float]) -> np.array:
    # Define Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]])  # σ1
    pauli_y = np.array([[0, -1j], [1j, 0]])  # σ2
    pauli_z = np.array([[1, 0], [0, -1]])  # σ3
    identity_2 = np.eye(2)

    # Define Kronecker product function for simplicity
    def kron(a, b):
        return np.kron(a, b)

    # Construct the density matrix ρ
    rho = (1 / 4) * (
            np.eye(4) +
            B['Bk'] * kron(pauli_x, identity_2) +
            B['Br'] * kron(pauli_y, identity_2) +
            B['Bn'] * kron(pauli_z, identity_2) +
            B['Ak'] * kron(identity_2, pauli_x) +
            B['Ar'] * kron(identity_2, pauli_y) +
            B['An'] * kron(identity_2, pauli_z) +
            C['kk'] * kron(pauli_x, pauli_x) +
            C['rr'] * kron(pauli_y, pauli_y) +
            C['nn'] * kron(pauli_z, pauli_z) +
            C['kr'] * kron(pauli_x, pauli_y) +
            C['kn'] * kron(pauli_x, pauli_z) +
            C['rn'] * kron(pauli_y, pauli_z) +
            C['rk'] * kron(pauli_y, pauli_x) +
            C['nr'] * kron(pauli_z, pauli_x) +
            C['nk'] * kron(pauli_z, pauli_y)
    )

    pauli_y_tensor = kron(pauli_y, pauli_y)
    rho_conjugate = np.conjugate(rho)
    rho_tilde = pauli_y_tensor @ rho_conjugate @ pauli_y_tensor
    # Compute the square root of ρ
    sqrt_rho = sqrtm(rho)
    # Compute the R matrix
    R = sqrt_rho @ rho_tilde @ sqrt_rho

    # Compute the eigenvalues of rho
    eigenvalues, _ = eig(R)
    eigenvalues = np.real(eigenvalues)
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]

    return eigenvalues


def calculate_polarization_from_hist(h_observable: Hist, analyzing_power: float):
    """
    Calculate the polarization using the formula:
        P = 3 * <cos(theta)> / analyzing_power
    where <cos(theta)> is the mean of the observable distribution, and analyzing_power is the analyzing power of the decay mode.
    """
    bin_centers = 0.5 * (h_observable.bin_edges[:-1] + h_observable.bin_edges[1:])
    mean, err_of_mean = get_mean_and_err_of_mean(bin_centers, weights=h_observable.values, err=h_observable.errors)
    polarization = 3 * mean / analyzing_power
    err_of_polarization = 3 * err_of_mean / np.abs(analyzing_power)
    return polarization, err_of_polarization

def calculate_spin_correlation_from_hist(h_observable: Hist, analyzing_power_a: float, analyzing_power_b: float):
    """
    Calculate the correlation using the formula:
        C = 9 * <cos(theta_A) * cos(theta_B)>
    where <cos(theta_A) * cos(theta_B)> is the mean of the product observable distribution.
    """
    bin_centers = 0.5 * (h_observable.bin_edges[:-1] + h_observable.bin_edges[1:])
    mean, err_of_mean = get_mean_and_err_of_mean(bin_centers, weights=h_observable.values, err=h_observable.errors)
    correlation = 9 * mean / (analyzing_power_a * analyzing_power_b)
    err_of_correlation = 9 * err_of_mean / np.abs(analyzing_power_a * analyzing_power_b)
    return correlation, err_of_correlation

def evaluate_quantum_results_with_uncertainties(BC_matrices) -> dict:
    """Compute eigenvalues and uncertainties using first-order perturbation theory with precomputed up/down values."""
    quantum_results = {}

    def compute_eigenvalues_from_BC_matrices(input_BC_matrices: dict) -> np.array:
        """Compute eigenvalues from a given set of BC_matrices."""
        C = {key.replace("C_", ""): value for key, value in input_BC_matrices.items() if 'C_' in key}
        B = {key.replace("B_", ""): value for key, value in input_BC_matrices.items() if 'B_' in key}
        return compute_full_density_matrix(C=C, B=B)

    BC_matrices_nominal, BC_matrices_up, BC_matrices_down = {}, {}, {}
    for key, value in BC_matrices.items():
        BC_matrices_nominal[key] = value.value
        BC_matrices_up[key] = value.value + value.err_up
        BC_matrices_down[key] = value.value - value.err_down

    # Compute nominal eigenvalues
    nominal_eigenvalues = compute_eigenvalues_from_BC_matrices(BC_matrices_nominal)

    # Initialize uncertainty storage for asymmetric uncertainties
    eigenvalue_uncertainties_up = np.zeros_like(nominal_eigenvalues)
    eigenvalue_uncertainties_down = np.zeros_like(nominal_eigenvalues)

    # Compute uncertainties
    for param in BC_matrices.keys():
        if param not in BC_matrices_up or param not in BC_matrices_down:
            continue  # Skip if no up/down value is provided for this parameter

        # Compute eigenvalues for up and down perturbed BC_matrices
        eigenvalues_up = compute_eigenvalues_from_BC_matrices({**BC_matrices_nominal, param: BC_matrices_up[param]})
        eigenvalues_down = compute_eigenvalues_from_BC_matrices({**BC_matrices_nominal, param: BC_matrices_down[param]})

        # Accumulate squared uncertainties
        eigenvalue_uncertainties_up += np.abs(eigenvalues_up - nominal_eigenvalues) ** 2
        eigenvalue_uncertainties_down += np.abs(eigenvalues_down - nominal_eigenvalues) ** 2

    # Final uncertainties (square root of accumulated squares)
    eigenvalue_uncertainties_up = np.sqrt(eigenvalue_uncertainties_up)
    eigenvalue_uncertainties_down = np.sqrt(eigenvalue_uncertainties_down)

    # Compute Concurrence
    concurrence_nominal = max(0, nominal_eigenvalues[0] - sum(nominal_eigenvalues[1:]))
    concurrence_uncertainty_up = np.sqrt(sum(eigenvalue_uncertainties_up ** 2))
    concurrence_uncertainty_down = np.sqrt(sum(eigenvalue_uncertainties_down ** 2))

    quantum_results['Concurrence'] = ValueWithUncertainty(value=concurrence_nominal, err_up=concurrence_uncertainty_up, err_down=concurrence_uncertainty_down)

    # Compute Cij terms with asymmetric uncertainties
    for i, j in [('kk', 'nn'), ('kk', 'rr'), ('nn', 'rr')]:
        value_sum = np.abs(BC_matrices_nominal[f'C_{i}'] + BC_matrices_nominal[f'C_{j}']) - np.sqrt(2)
        value_diff = np.abs(BC_matrices_nominal[f'C_{i}'] - BC_matrices_nominal[f'C_{j}']) - np.sqrt(2)

        uncertainty_up = np.sqrt(
            (BC_matrices_up[f'C_{i}'] - BC_matrices_nominal[f'C_{i}']) ** 2 + (BC_matrices_up[f'C_{j}'] - BC_matrices_nominal[f'C_{j}']) ** 2
        )
        uncertainty_down = np.sqrt(
            (BC_matrices_nominal[f'C_{i}'] - BC_matrices_down[f'C_{i}']) ** 2 + (BC_matrices_nominal[f'C_{j}'] - BC_matrices_down[f'C_{j}']) ** 2
        )

        quantum_results[f'C{i} + C{j}'] = ValueWithUncertainty(value=value_sum, err_up=uncertainty_up, err_down=uncertainty_down)
        quantum_results[f'C{i} - C{j}'] = ValueWithUncertainty(value=value_diff, err_up=uncertainty_up, err_down=uncertainty_down)

    return quantum_results


def derive_results(obs_hist_dict, analyzing_power_a, analyzing_power_b):
    """
    Derive the B and C matrices from the histograms of the observables, then compute the quantum results with uncertainties.
    """
    BC_matrices = {}
    # calculate polarization
    for axis in ['n', 'r', 'k']:
        for particle in ['A', 'B']:
            obs_name = f'cos_theta_{particle}_{axis}'
            analyzing_power = analyzing_power_a if particle == 'A' else analyzing_power_b
            if obs_name in obs_hist_dict:
                polarization, err_of_polarization = calculate_polarization_from_hist(obs_hist_dict[obs_name], analyzing_power)
                BC_matrices[f'B_{particle}{axis}'] = ValueWithUncertainty(value=polarization, err_up=err_of_polarization, err_down=err_of_polarization)

    # calculate spin correlation
    for axis_a, axis_b in product(['n', 'r', 'k'], repeat=2):
        obs_name = f'cos_theta_A_{axis_a}_times_cos_theta_B_{axis_b}'
        if obs_name in obs_hist_dict:
            correlation, err_of_correlation = calculate_spin_correlation_from_hist(obs_hist_dict[obs_name], analyzing_power_a, analyzing_power_b)
            BC_matrices[f'C_{axis_a}{axis_b}'] = ValueWithUncertainty(value=correlation, err_up=err_of_correlation, err_down=err_of_correlation)

    # compute quantum results with uncertainties
    quantum_results = evaluate_quantum_results_with_uncertainties(BC_matrices)
    
    return BC_matrices, quantum_results


if __name__ == "__main__":
    import os
    # plot observables using truth Z->tautau events
    event_categores = [11]
    events_name = 'pipi'
    event_categores = [13, 14, 31, 41]
    events_name = 'pilep'
    input_file = '/eos/user/c/cmo/project/ZtautauLep/tree_ana/run/20260328-testhadhad/Ztautau/filtered___raw.parquet'
    output_dir = f'example_plots/{events_name}/'
    output_dir = f'test/{events_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    events = ak.from_parquet(input_file)
    events = events[np.isin(events['event_category'], event_categores)]

    # build tau and visible 4-momenta
    truth_pdgId = events['GenPart_pdgId']
    mis_pdg_ID = np.array([-12, -14, 16])

    tau_a_p4 = get_p4_from_ak_events(events, flag=(truth_pdgId == -15), prefix='GenPart_vector')
    mis_a_flag = np.zeros_like(truth_pdgId, dtype=bool)
    for pdg_id in mis_pdg_ID:
        mis_a_flag = mis_a_flag | (truth_pdgId == -pdg_id)
    mis_a_flag = mis_a_flag & (events['GenPart_status'] == 1)
    # mis_a_p4 = get_p4_from_ak_events(events, flag=mis_a_flag, prefix='GenPart_vector')
    mis_a_p4 = get_sum_p4_from_ak_events(events, flag=mis_a_flag, prefix='GenPart_vector')

    tau_b_p4 = get_p4_from_ak_events(events, flag=(truth_pdgId == 15), prefix='GenPart_vector')
    mis_b_flag = np.zeros_like(truth_pdgId, dtype=bool)
    for pdg_id in mis_pdg_ID:
        mis_b_flag = mis_b_flag | (truth_pdgId == pdg_id)
    mis_b_flag = mis_b_flag & (events['GenPart_status'] == 1)
    mis_b_p4 = get_sum_p4_from_ak_events(events, flag=mis_b_flag, prefix='GenPart_vector')

    vis_a_p4 = tau_a_p4 - mis_a_p4
    vis_b_p4 = tau_b_p4 - mis_b_p4

    observables = build_observables(tau_a_p4, tau_b_p4, vis_a_p4, vis_b_p4)


    # plot the observables
    import matplotlib.pyplot as plt
    for obs_name, obs_values in observables.items():
        mean, err_of_mean = get_mean_and_err_of_mean(obs_values)
        print(f"{obs_name}: mean = {mean:.4f} ± {err_of_mean:.4f}")
        plt.figure(figsize=(8, 6), dpi=300)
        plt.hist(obs_values, bins=50, range=(-1, 1), histtype='step', density=True, linewidth=2, label=f'{obs_name}: mean={mean:.3f}±{err_of_mean:.3f}')
        plt.xlabel(obs_name)
        plt.ylabel('Density')
        plt.title(f'{obs_name} distribution for {events_name} events')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.ylim(0, None)
        plt.legend()
        plt.savefig(f'{output_dir}/{obs_name}.png')
