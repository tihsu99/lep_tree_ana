import numpy as np
import pandas as pd
import uproot as ur
import matplotlib.pyplot as plt
import logging
import vector
import glob
import os
import awkward as ak
import copy
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events,\
            get_all_p4_from_ak_events, cme, rebuild_p4, deltaR_nearby
from quantum.observables_builder import build_observables, get_analyzing_power_ary


def define_recon_level_variables(events: ak.Array):
    # Particle p4 
    events['Part_p4'] = vector.zip(
        {
            "px": events['Part_fourMomentum_fCoordinates_fX'],
            "py": events['Part_fourMomentum_fCoordinates_fY'],
            "pz": events['Part_fourMomentum_fCoordinates_fZ'],
            "E": events['Part_fourMomentum_fCoordinates_fT'],
        }
    )

    """
        Define hemisphere specific variables
    """

    # Split event into hemisphere a and b according to thrust
    truthst = vector.zip({"x": events['thrust_x'], "y": events['thrust_y'], "z": events['thrust_z']})
    part_hemisphere_id = ak.where(events['Part_p4'].to_Vector3D().dot(truthst) > 0, 1, -1)

    # find the leading charged particle in each hemisphere
    part_idx_all = ak.local_index(events['Part_pdgId'])
    flags_hemisphere = {
        'a': part_hemisphere_id == 1,
        'b': part_hemisphere_id == -1,
    }
    lead_particle_flags = {'a': None, 'b': None}
    lead_particle_charges = {'a': None, 'b': None}
    for hemisphere in ['a', 'b']:
        mask = flags_hemisphere[hemisphere] & (events['Part_charge'] != 0)
        p4 = events['Part_p4'][mask]
        idx_particle = part_idx_all[mask]
        idx_sorted = ak.argsort(p4.p, axis=1, ascending=False)
        particle_idx_sorted = idx_particle[idx_sorted]
        leading_particle_idx = ak.firsts(particle_idx_sorted)
        flag_is_leading_particle = (part_idx_all == leading_particle_idx)
        lead_particle_flags[hemisphere] = flag_is_leading_particle
        lead_particle_charges[hemisphere] = ak.fill_none(ak.firsts(events['Part_charge'][flag_is_leading_particle]), 0)

    # label events that have opposite charge leading particles in two hemispheres
    events['is_leading_OS'] = (lead_particle_charges['a']+ lead_particle_charges['b'] == 0) & (lead_particle_charges['a'] != 0) & (lead_particle_charges['b'] != 0)
    events['is_leading_SS'] = (lead_particle_charges['a'] - lead_particle_charges['b'] == 0) & (lead_particle_charges['a'] != 0) & (lead_particle_charges['b'] != 0)
    events['valid_leadings'] = (events['is_leading_OS'] | events['is_leading_SS'])

    # redefine hemisphere id based on charge of leading particle in each hemisphere, if exists
    # after this, charge of a is positive and charge of b is negative for OS events.
    switch_hemisphere = events['is_leading_OS'] & (lead_particle_charges['a'] < 0) & (lead_particle_charges['b'] > 0)
    switch_hemisphere_broadcasted = ak.broadcast_arrays(switch_hemisphere, part_hemisphere_id)[0]
    events['Part_in_hemisphere_a'] = flags_hemisphere['a'] ^ switch_hemisphere
    events['Part_in_hemisphere_b'] = flags_hemisphere['b'] ^ switch_hemisphere
    events['is_lead_a'] = lead_particle_flags['a'] ^ switch_hemisphere_broadcasted
    events['is_lead_b'] = lead_particle_flags['b'] ^ switch_hemisphere_broadcasted

    # store p4, pdgId and other info of leading particle in each hemisphere
    for hemisphere in ['a', 'b']:
        lead_flag = events[f'is_lead_{hemisphere}']
        events[f'lead_{hemisphere}_valid'] = ak.any(lead_flag, axis=1)
        lead_px = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fX'][lead_flag]), 0)
        lead_py = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fY'][lead_flag]), 0)
        lead_pz = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fZ'][lead_flag]), 0)
        lead_E = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fT'][lead_flag]), 0)
        events[f'lead_{hemisphere}_p4'] = vector.zip({"px": lead_px, "py": lead_py, "pz": lead_pz, "E": lead_E,})
        events[f'lead_{hemisphere}_pdgId'] = ak.fill_none(ak.firsts(events['Part_pdgId'][lead_flag]), -999)
        events[f'lead_{hemisphere}_charge'] = ak.fill_none(ak.firsts(events['Part_charge'][lead_flag]), 0)
        events[f'lead_{hemisphere}_hpcTotalShowerEnergy'] = ak.fill_none(ak.firsts(events['Part_hpcTotalShowerEnergy'][lead_flag]), -999)
        events[f'lead_{hemisphere}_z0'] = ak.fill_none(ak.firsts(events['Trac_impParToVertexZ'][lead_flag]), -999)
        events[f'lead_{hemisphere}_d0'] = ak.fill_none(ak.firsts(events['Trac_impParToVertexRPhi'][lead_flag]), -999)

        # calculate dR between leading particle and all other particles
        events[f'Part_dR_to_lead_{hemisphere}'] = events[f'lead_{hemisphere}_p4'].deltaR(events['Part_p4'])
        events[f'Part_angle_to_lead_{hemisphere}'] = events[f'lead_{hemisphere}_p4'].deltaangle(events['Part_p4']) * 180 / np.pi

        # find leading pions/electrons/muons and their nearby photons in each hemisphere
        events[f'lead_{hemisphere}_is_pion'] = ak.fill_none(ak.any(lead_flag & (abs(events['Part_pdgId']) == 41) & (events['Part_charge'] != 0), axis=1), False)
        events[f'lead_{hemisphere}_is_electron'] = ak.fill_none(ak.any(lead_flag & (abs(events['Part_pdgId']) == 2) & (events['Part_charge'] != 0), axis=1), False)
        events[f'lead_{hemisphere}_is_muon'] = ak.fill_none(ak.any(lead_flag & (abs(events['Part_pdgId']) == 6) & (events['Part_charge'] != 0), axis=1), False)
        events[f'lead_{hemisphere}_is_lepton'] = ak.fill_none(ak.any(lead_flag & (abs(events['Part_pdgId']) == 2) & (events['Part_charge'] != 0), axis=1), False)

        events[f'is_photon_near_lead_{hemisphere}'] = (events['Part_pdgId'] == 21) & (events['Part_charge'] == 0) & (events[f'Part_dR_to_lead_{hemisphere}'] < deltaR_nearby)
        events[f'num_photon_near_lead_{hemisphere}'] = ak.fill_none(ak.sum(events[f'is_photon_near_lead_{hemisphere}'], axis=1), 0)

        # calculate visible p4 for each hemisphere
        sum_nearby_photon_p4 = get_sum_p4_from_ak_events(events, events[f'is_photon_near_lead_{hemisphere}']==1)
        events[f'lead_{hemisphere}_visible_p4'] = events[f'lead_{hemisphere}_p4'] + sum_nearby_photon_p4

        sum_hemisphere_photon_p4 = get_sum_p4_from_ak_events(events, (events['Part_pdgId'] == 21) & (events['Part_charge'] == 0) & events[f'Part_in_hemisphere_{hemisphere}'])
        events[f'hemisphere_{hemisphere}_visible_p4'] = events[f'lead_{hemisphere}_p4'] + sum_hemisphere_photon_p4


    """
        Define event level variables
    """
    events['nprong'] = ak.sum(events['Part_charge'] != 0, axis=1)
    events['charged_E'] = ak.sum(events['Part_fourMomentum_fCoordinates_fT'][events['Part_charge'] != 0], axis=1) + ak.sum(events['Part_hpcTotalShowerEnergy'][events['Part_charge'] != 0], axis=1)

    # missing p4 and missing pt
    mask_for_missing_p4 = (events['Part_charge'] != 0) | (events['Part_pdgId'] == 21)
    events['missing_p4'] = vector.zip({
        "px": -ak.sum(events['Part_fourMomentum_fCoordinates_fX'][mask_for_missing_p4], axis=1),
        "py": -ak.sum(events['Part_fourMomentum_fCoordinates_fY'][mask_for_missing_p4], axis=1),
        "pz": -ak.sum(events['Part_fourMomentum_fCoordinates_fZ'][mask_for_missing_p4], axis=1),
        "E": cme - ak.sum(events['Part_fourMomentum_fCoordinates_fT'][mask_for_missing_p4], axis=1),
    })
    events['missing_pt'] = events['missing_p4'].pt

    # isolation angle: minimum angle between charged particles in different hemispheres
    pairs = ak.cartesian({
        'a': events['Part_p4'][events['Part_in_hemisphere_a'] & (events['Part_charge'] != 0)],
        'b': events['Part_p4'][events['Part_in_hemisphere_b'] & (events['Part_charge'] != 0)],
    })
    isolation_angle = pairs['a'].deltaangle(pairs['b']) * 180 / np.pi
    events['isolation_angle'] = ak.fill_none(ak.min(isolation_angle, axis=1), -999)

    # Prad, Erad for events with charged leading particle in both hemispheres
    flag_valid = events['valid_leadings']
    events['P_rad'] = ak.where(flag_valid, (events['lead_a_p4'].p**2 + events['lead_b_p4'].p**2)**0.5 / (cme/2), -999)
    E_rad = ak.zeros_like(events['P_rad'])
    for hemisphere in ['a', 'b']:
        flag_nearby_hpc_energy = events[f'Part_angle_to_lead_{hemisphere}'] < 30
        E_rad = E_rad + ak.sum(events[f'Part_hpcTotalShowerEnergy'][flag_nearby_hpc_energy], axis=-1)**2
    events['E_rad'] = ak.where(flag_valid, E_rad**0.5 / (cme/2), -999)
    return events


def define_truth_level_variables(events: ak.Array, is_Ztautau=False):
    if not is_Ztautau:
        print("Currently only truth level variable definition for Ztautau samples is implemented. Returning events without modification.")
        return events


    genpart_p4 = get_all_p4_from_ak_events(events, ak.ones_like(events['GenPart_pdgId'], dtype=bool), 'GenPart_vector')
    for charge, hemisphere_id in [(1, 'a'), (-1, 'b')]:
        # -15 - tau+ - a, 15 - tau- - b
        tau_mask = (events['GenPart_pdgId'] == -15 * charge)
        tau_p4 = get_p4_from_ak_events(events, tau_mask, 'GenPart_vector')
        events[f'truth_tau_{hemisphere_id}_p4'] = tau_p4

        dR_to_tau = tau_p4.deltaR(genpart_p4)
        events[f'GenPart_is_final_state_near_tau_{hemisphere_id}'] = (dR_to_tau < deltaR_nearby) & (events['GenPart_status'] == 1)
        events[f'GenPart_is_photon_near_tau_{hemisphere_id}'] = events[f'GenPart_is_final_state_near_tau_{hemisphere_id}'] & (events['GenPart_pdgId'] == 22)
        events[f'GenPart_is_pion_near_tau_{hemisphere_id}'] = events[f'GenPart_is_final_state_near_tau_{hemisphere_id}'] & (abs(events['GenPart_pdgId']) == 211)
        events[f'GenPart_is_lepton_near_tau_{hemisphere_id}'] = events[f'GenPart_is_final_state_near_tau_{hemisphere_id}'] & ((abs(events['GenPart_pdgId']) == 11) | (abs(events['GenPart_pdgId']) == 13))
        events[f'truth_num_photon_near_tau_{hemisphere_id}'] = ak.sum(events[f'GenPart_is_photon_near_tau_{hemisphere_id}'], axis=1)
        events[f'truth_num_pion_near_tau_{hemisphere_id}'] = ak.sum(events[f'GenPart_is_pion_near_tau_{hemisphere_id}'], axis=1)

        # find corresponding neutrinos from tau decay
        missing_pdgId = np.array([12, 14, -16]) * charge
        mask_is_missing_part = ak.zeros_like(events['GenPart_pdgId'], dtype=bool)
        for pdgId in missing_pdgId:
            mask_is_missing_part = mask_is_missing_part | (events['GenPart_pdgId'] == pdgId)
        mask_is_missing_part = mask_is_missing_part & (events['GenPart_status'] == 1)
        events[f'truth_missing_{hemisphere_id}_p4'] = get_sum_p4_from_ak_events(events, mask_is_missing_part, 'GenPart_vector')
        events[f'truth_visible_{hemisphere_id}_p4'] = tau_p4 - events[f'truth_missing_{hemisphere_id}_p4']


    # build QI observables
    observables = build_observables(tau_a_p4=events['truth_tau_a_p4'], tau_b_p4=events['truth_tau_b_p4'],
                                    vis_a_p4=events['truth_visible_a_p4'], vis_b_p4=events['truth_visible_b_p4'])
    for obs_name, obs_value in observables.items():
        events[f'truth_{obs_name}'] = obs_value

    # truth-level selection for QI region
    events['truth_QI_region'] = events['truth_theta_cm'] * 2 / np.pi > 0.6

    # analyzing power
    # non-tau, pion, rho, ele, mu, other 
    analyzing_power_ary = get_analyzing_power_ary()
    event_category = events['event_category']

    pos_power = analyzing_power_ary[event_category // 10]
    neg_power = analyzing_power_ary[event_category % 10]
    events['analyzing_power_a'] = pos_power
    events['analyzing_power_b'] = neg_power
    events['analyzing_power'] = pos_power * neg_power

    return events





