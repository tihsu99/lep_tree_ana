# Adapted from https://gitlab.cern.ch/yulei/ztautauml
import pandas as pd
import awkward as ak
import vector
import numpy as np
from itertools import product

from pandas import DataFrame
from scipy.linalg import sqrtm, eig

from utils.common_functions import get_p4_from_ak_events

def get_observable_names():
    """
    Get the names of the observables that we will build
    """
    observable_names = []
    observable_names += [f'cos_theta_A_{axis}' for axis in ['n', 'r', 'k']]
    observable_names += [f'cos_theta_B_{axis}' for axis in ['n', 'r', 'k']]
    for axis_a, axis_b in product(['n', 'r', 'k'], repeat=2):
        observable_names.append(f'cos_theta_A_{axis_a}_times_cos_theta_B_{axis_b}')
    observable_names.append('cos_AB')
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

    # boost helicity basis to tau a rest frame. This actually should be the same as pre-boost helicity basis since they are all perpendicular or parallel to the boost direction, but we do it for consistency.
    for axis in ['n', 'r', 'k']:
        axis_vector = helicity_basis_a[axis]
        helicity_basis_a_tmp = vector.zip({
            "x": axis_vector.x,
            "y": axis_vector.y,
            "z": axis_vector.z,
            "t": np.zeros_like(axis_vector.x),
        })
        helicity_basis_a_a_rest = helicity_basis_a_tmp.boost(boost_to_a_rest)
        helicity_basis_a[axis] = helicity_basis_a_a_rest.to_pxpypz().unit()

        helicity_basis_a_b_rest = helicity_basis_a_tmp.boost(boost_to_b_rest)
        helicity_basis_a[axis] = helicity_basis_a_b_rest.to_pxpypz().unit()


    observables = {}
    for axis in ['n', 'r', 'k']:
        observables[f'cos_theta_A_{axis}'] = vis_a_p4_a_rest.to_pxpypz().unit().dot(helicity_basis_a[axis])
        observables[f'cos_theta_B_{axis}'] = vis_b_p4_b_rest.to_pxpypz().unit().dot(helicity_basis_a[axis])
    
    # product observables
    for axis_a, axis_b in product(['n', 'r', 'k'], repeat=2):
        observables[f'cos_theta_A_{axis_a}_times_cos_theta_B_{axis_b}'] = observables[f'cos_theta_A_{axis_a}'] * observables[f'cos_theta_B_{axis_b}']

    observables['cos_AB'] = vis_a_p4_a_rest.to_pxpypz().unit().dot(vis_b_p4_b_rest.to_pxpypz().unit())

    return observables



if __name__ == "__main__":
    import os
    # plot observables using truth Z->tautau events
    event_categores = [11]
    events_name = 'pipi'
    input_file = '/eos/user/c/cmo/project/ZtautauLep/tree_ana/run/20260323-pipi/Ztautau/filtered___raw.parquet'
    output_dir = f'example_plots/{events_name}/'
    output_dir = f'test/{events_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    events = ak.from_parquet(input_file)
    events = events[np.isin(events['event_category'], event_categores)]

    # build tau and visible 4-momenta
    truth_pdgId = events['GenPart_pdgId']
    tau_a_p4 = get_p4_from_ak_events(events, flag=(truth_pdgId == -15), prefix='GenPart_vector')
    mis_a_p4 = get_p4_from_ak_events(events, flag=(truth_pdgId == -16), prefix='GenPart_vector')
    tau_b_p4 = get_p4_from_ak_events(events, flag=(truth_pdgId == 15), prefix='GenPart_vector')
    mis_b_p4 = get_p4_from_ak_events(events, flag=(truth_pdgId == 16), prefix='GenPart_vector')
    vis_a_p4 = tau_a_p4 - mis_a_p4
    vis_b_p4 = tau_b_p4 - mis_b_p4

    observables = build_observables(tau_a_p4, tau_b_p4, vis_a_p4, vis_b_p4)


    # plot the observables
    import matplotlib.pyplot as plt
    for obs_name, obs_values in observables.items():
        plt.figure(figsize=(8, 6), dpi=300)
        plt.hist(obs_values, bins=50, range=(-1, 1), histtype='step', density=True, linewidth=2)
        plt.xlabel(obs_name)
        plt.ylabel('Density')
        plt.title(f'{obs_name} distribution for {events_name} events')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.ylim(0, None)
        plt.savefig(f'{output_dir}/{obs_name}.png')

