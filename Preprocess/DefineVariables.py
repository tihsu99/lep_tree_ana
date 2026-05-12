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
from quantum.observables_builder import build_observables, get_observable_names, get_bc_name_from_variable_name
import quantum.unfold as unfold
import utils.tau_decay as tau_decay
from utils.tau_decay import get_analyzing_powers_from_event_categories, get_analyzing_powers_from_event_category
from processor.NeutrinoReconstructionProcessor import compute_neutrino_momenta
from mmc.MMC import MMC


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
        events[f'lead_{hemisphere}_idx'] = leading_particle_idx
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
        lead_p = events[f'lead_{hemisphere}_p4'].p + 1e-10
        events[f'lead_{hemisphere}_E_over_p'] = ak.fill_none(ak.firsts(events['Part_hpcTotalShowerEnergy'][lead_flag]) / lead_p, -999)

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

    # ---------------------------------------------------------
    # Leptonic Specific Aliases & Hardware Variables
    # ---------------------------------------------------------
    events['n_charged'] = events['nprong']
    
    n_charged_a = ak.sum(flags_hemisphere['a'] & (events['Part_charge'] != 0), axis=1)
    n_charged_b = ak.sum(flags_hemisphere['b'] & (events['Part_charge'] != 0), axis=1)
    events['pass_1_vs_1'] = (n_charged_a == 1) & (n_charged_b == 1)

    events['lead_a_raw_hpc_E'] = events['lead_a_hpcTotalShowerEnergy']
    events['lead_b_raw_hpc_E'] = events['lead_b_hpcTotalShowerEnergy']

    # Pre-calculate neutral energy mapping
    is_neutral = (events['Part_charge'] == 0)
    neutral_hpc_E = ak.where(is_neutral, events['Part_hpcTotalShowerEnergy'], 0.0)

    # Single unified loop for both Electron and Muon variables
    for hemi in ['a', 'b']:
        lead_flag = events[f'is_lead_{hemi}']
        lead_idx = events[f'lead_{hemi}_idx']
        
        # --- Muon Variables ---
        if 'Muid_partIdx' in events.fields:
            muid_mask = (events['Muid_partIdx'] == lead_idx)
            events[f'lead_{hemi}_raw_muon_tag'] = ak.fill_none(ak.firsts(events['Muid_tag'][muid_mask]), 0)
            events[f'lead_{hemi}_raw_muon_hits'] = ak.fill_none(ak.firsts(events['Muid_hitPattern'][muid_mask]), 0)
        else:
            events[f'lead_{hemi}_raw_muon_tag'] = ak.zeros_like(events['nprong'])
            events[f'lead_{hemi}_raw_muon_hits'] = ak.zeros_like(events['nprong'])
            
        # --- Electron Variables ---
        d_angle = np.abs(events[f'lead_{hemi}_p4'].deltaangle(events['Part_p4'])) * (180.0 / np.pi)
        in_cone = d_angle < 18.0
        events[f'neutral_cone18_{hemi}'] = ak.sum(neutral_hpc_E[in_cone], axis=1)

        events[f'lead_{hemi}_raw_wires'] = ak.fill_none(ak.firsts(events['Dedx_nrWires'][lead_flag]), -999)
        events[f'lead_{hemi}_raw_hcal'] = ak.fill_none(ak.firsts(events['Part_hacTowerHitPattern'][lead_flag]), -999)
        
        raw_shower_phi = ak.fill_none(ak.firsts(events['Part_hpcShowerPhi'][lead_flag]), -999.0)
        raw_track_phi = events[f'lead_{hemi}_p4'].phi

        best_phi_rad = ak.where(raw_shower_phi > -900.0, raw_shower_phi, raw_track_phi)
        best_phi_deg = best_phi_rad * (180.0 / np.pi)
        best_dist_mod = (best_phi_deg % 360.0) % 15.0
        
        events[f'lead_{hemi}_best_crack'] = ak.where(best_dist_mod < 7.5, best_dist_mod, 15.0 - best_dist_mod)

        if 'Elid_partIdx' in events.fields:
            elid_mask = (events['Elid_partIdx'] == lead_idx)
            events[f'lead_{hemi}_elid'] = ak.fill_none(ak.firsts(events['Elid_tag'][elid_mask]), 0)
        else:
            events[f'lead_{hemi}_elid'] = ak.zeros_like(events['nprong'])

        events[f'lead_{hemi}_hpc_E'] = ak.fill_none(ak.firsts(events['Part_hpcTotalShowerEnergy'][lead_flag]), 0.0)

        # =========================================================
        # REDEFINE 'is_electron' AND 'is_muon' WITH STRICT N-1 CUTS
        # =========================================================
        base_is_e = ak.fill_none(ak.any(lead_flag & (abs(events['Part_pdgId']) == 2) & (events['Part_charge'] != 0), axis=1), False)
        base_is_mu = ak.fill_none(ak.any(lead_flag & (abs(events['Part_pdgId']) == 6) & (events['Part_charge'] != 0), axis=1), False)
        
        # --- FROM YOUR apply_tautau_to_ee SCRIPT ---
        strict_e_hardware = (
            (events[f'lead_{hemi}_raw_wires'] >= 38) &               # pass_cut_ele_wires
            (events[f'lead_{hemi}_raw_muon_hits'] == 0) &            # pass_cut_ele_muon_veto
            ((events[f'lead_{hemi}_raw_hcal'] >> 1) == 0) &          # pass_cut_ele_hcal_veto
            (events[f'lead_{hemi}_best_crack'] > 1.0) &              # pass_cut_ele_best_crack
            (events[f'lead_{hemi}_hpc_E'] < 40.0) &                  # pass_cut_ele_pid
            (events[f'neutral_cone18_{hemi}'] < 4.0) &               # pass_cut_neutral
            (events[f'lead_{hemi}_elid'] >= 3) &                     # pass_cut_official_elid
            (events[f'lead_{hemi}_raw_muon_tag'] == 0)               # pass_cut_official_muon
        )

        # --- FROM YOUR apply_tautau_to_mumu SCRIPT ---
        strict_mu_hardware = (
            (events[f'lead_{hemi}_raw_muon_tag'] >= 7) &             # pass_muid_strict (Bitmask >= 7)
            (events[f'lead_{hemi}_raw_muon_hits'] >= 2) &            # pass_muon_hits (>= 2 hits in chambers)
            (events[f'lead_{hemi}_hpc_E'] < 2.0)                     # pass_cut_mip (MIP signature < 2.0 GeV)
        )
        
        # Overwrite the flags in the events array
        events[f'lead_{hemi}_is_electron'] = base_is_e & strict_e_hardware
        events[f'lead_{hemi}_is_muon'] = base_is_mu & strict_mu_hardware
        
        # Redefine is_lepton to include only these strictly vetted particles
        events[f'lead_{hemi}_is_lepton'] = events[f'lead_{hemi}_is_electron'] | events[f'lead_{hemi}_is_muon']

    return events


def define_signal_exclusive_variables(events: ak.Array):
    """
        Truth-level variables for Z->tautau events
    """
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
    events['truth_QI_region'] = (events['truth_theta_cm'] > 0.6) & (events['truth_mtautau'] > 80)

    # analyzing power
    # non-tau, pion, rho, ele, mu, other
    pos_power, neg_power = get_analyzing_powers_from_event_categories(events['event_category'])
    events['analyzing_power_a'] = pos_power
    events['analyzing_power_b'] = neg_power
    events['analyzing_power'] = pos_power * neg_power

    # reweight events to get expected distributions for QI observables at truth level (remove bias from tau decay and selection)
    mask_reweight = events['truth_QI_region'] & (events['analyzing_power'] != 0)
    bins_reweighting = np.linspace(-1, 1, 101)
    bins_centers = 0.5 * (bins_reweighting[:-1] + bins_reweighting[1:])

    weight_original = ak.to_numpy(events['weight'])
    for obs_name in [obs for obs in get_observable_names() if 'cos' in obs]:
        reweight_sf = np.ones_like(obs_values)

        bc_name = get_bc_name_from_variable_name(obs_name)
        nominal_bc_value = tau_decay.NOMINAL_BC_VALUES[bc_name]

        # retrieve observable values and specify events to be reweighted
        obs_values = ak.to_numpy(events[f'truth_{obs_name}'])
        obs_binned = unfold.bin_variable(observable_values=obs_values, bins=bins_reweighting)
        mask_obs = mask_reweight & (obs_values > -1) & (obs_values < 1)

        # calculate sf channel-wise
        for c_pos in range(1, 5):
            for c_neg in range(1, 5):
                channel = 10 * c_pos + c_neg
                ap_pos, ap_neg = get_analyzing_powers_from_event_category(channel)
                mask_channel = mask_obs & (events['event_category'] == channel)

                sum_weights_per_bin, _ = np.histogram(obs_values[mask_channel], bins=bins_reweighting, weights=weight_original[mask_channel])
                sum_weights_total = np.sum(sum_weights_per_bin)

                # get target distribution
                slope = nominal_bc_value
                extra_factor = 1
                if bc_name.startswith('B_A'):
                    slope = ap_pos
                elif bc_name.startswith('B_B'):
                    slope = ap_neg
                elif bc_name.startswith('C_'):
                    slope = ap_pos * ap_neg
                    extra_factor = -1 * np.log(np.abs(bins_centers)) 
                target_distribution = 0.5 * (1 + slope * bins_centers) * extra_factor
                # drop norm effect
                target_distribution = target_distribution / np.sum(target_distribution) * sum_weights_total

                # calculate sf per bin and apply to events in the bin
                sf_per_bin = np.ones_like(sum_weights_per_bin)
                nonzero_mask = sum_weights_per_bin > 0
                sf_per_bin[nonzero_mask] = target_distribution[nonzero_mask] / sum_weights_per_bin[nonzero_mask]
                for i in range(len(sf_per_bin)):
                    mask_bin = mask_channel & (obs_binned == i)
                    reweight_sf[mask_bin] = sf_per_bin[i]
        events[f'{obs_name}_reweight_sf'] = reweight_sf
                
    return events


def define_region_specific_variables(events: ak.Array):
    """
        Recon-level neutrino reconstruction and QI observables
    """
    reco_vis_positive_p4 = events['lead_a_visible_p4']
    reco_vis_negative_p4 = events['lead_b_visible_p4']
    num_events = len(events)
    nu_pos_px, nu_pos_py, nu_pos_pz, nu_pos_E = np.zeros(num_events), np.zeros(num_events), np.zeros(num_events), np.zeros(num_events)
    nu_neg_px, nu_neg_py, nu_neg_pz, nu_neg_E = np.zeros(num_events), np.zeros(num_events), np.zeros(num_events), np.zeros(num_events)
    flags_valid_array = np.zeros(num_events, dtype=bool)
    mmc_likelihood = np.zeros(num_events)

    mask_do_mmc = np.zeros(num_events, dtype=bool)
    # MMC for certain regions
    mmc_regions = ['ee', 'mumu', 'emu']
    mmc_engine = MMC({'mmc_regions': mmc_regions, 'mmc_workers': 200})
    for region in mmc_regions:
        mask_region = events[f'{region}_cut']
        mask_do_mmc = mask_do_mmc | mask_region
        tmp_v1_p4, tmp_v2_p4, tmp_flag_valid, tmp_mmc_likelihood = mmc_engine.calculate(
            vis_a_p4=reco_vis_positive_p4[mask_region],
            vis_b_p4=reco_vis_negative_p4[mask_region],
            region_name=region,
            events=events[mask_region]
        )
        nu_pos_px[mask_region], nu_pos_py[mask_region], nu_pos_pz[mask_region], nu_pos_E[mask_region] = tmp_v1_p4.px, tmp_v1_p4.py, tmp_v1_p4.pz, tmp_v1_p4.E
        nu_neg_px[mask_region], nu_neg_py[mask_region], nu_neg_pz[mask_region], nu_neg_E[mask_region] = tmp_v2_p4.px, tmp_v2_p4.py, tmp_v2_p4.pz, tmp_v2_p4.E
        flags_valid_array[mask_region] = tmp_flag_valid
        mmc_likelihood[mask_region] = tmp_mmc_likelihood

    # Analytical solution for the rest of the events (baseline region - MMC regions)
    mask_analytical = (~mask_do_mmc) & events['baseline_cut']
    if np.any(mask_analytical):
        tmp_v1_p4, tmp_v2_p4, flags_valid_array_analytical = compute_neutrino_momenta(
            vis1_p4=reco_vis_positive_p4[mask_analytical],
            vis2_p4=reco_vis_negative_p4[mask_analytical],
        )
        nu_pos_px[mask_analytical], nu_pos_py[mask_analytical], nu_pos_pz[mask_analytical], nu_pos_E[mask_analytical] = tmp_v1_p4.px, tmp_v1_p4.py, tmp_v1_p4.pz, tmp_v1_p4.E
        nu_neg_px[mask_analytical], nu_neg_py[mask_analytical], nu_neg_pz[mask_analytical], nu_neg_E[mask_analytical] = tmp_v2_p4.px, tmp_v2_p4.py, tmp_v2_p4.pz, tmp_v2_p4.E
        flags_valid_array[mask_analytical] = flags_valid_array_analytical
        mmc_likelihood[mask_analytical] = np.ones_like(nu_neg_px[mask_analytical])  

    events[f'lead_a_missing_p4'] = vector.zip({"px": nu_pos_px, "py": nu_pos_py, "pz": nu_pos_pz, "E": nu_pos_E})
    events[f'lead_b_missing_p4'] = vector.zip({"px": nu_neg_px, "py": nu_neg_py, "pz": nu_neg_pz, "E": nu_neg_E})
    events['flags_valid'] = flags_valid_array
    events['mmc_likelihood'] = mmc_likelihood

    for key in ['a', 'b']:
        events[f'reco_tau_{key}_p4'] = events[f'lead_{key}_visible_p4'] + events[f'lead_{key}_missing_p4']

    # derive the observables for QI study
    observables = build_observables(tau_a_p4=events['reco_tau_a_p4'], tau_b_p4=events['reco_tau_b_p4'], vis_a_p4=events['lead_a_visible_p4'], vis_b_p4=events['lead_b_visible_p4'])
    for obs_name, obs_values in observables.items():
        events[obs_name] = obs_values

    return events
