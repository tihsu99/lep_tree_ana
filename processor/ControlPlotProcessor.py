import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme
from utils.plotter import do_control_plot
from quantum.observables_builder import get_observable_names

    

def make_control_plots_tautau(dl_dict, luminosity, normalize, output_dir, region_name="tautau", log_scale=True):
    # isolation angle
    def get_isolation_angle(events):
        isolation_angle = ak.to_numpy(events['isolation_angle'], allow_missing=False)
        return isolation_angle

    bin_edges = np.linspace(160, 180, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_isolation_angle,
        bin_edges=bin_edges,
        x_label='Isolation Angle [deg]',
        title='Control Plot: Isolation Angle',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_isolation_angle.png")

    # Erad
    def get_erad(events):
        erad = ak.to_numpy(events['E_rad'], allow_missing=False)
        return erad
    bin_edges = np.linspace(0, 0.8, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_erad,
        bin_edges=bin_edges,
        x_label='E_rad',
        title='Control Plot: E_rad',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_erad.png")

    # p_rad
    def get_prad(events):
        prad = ak.to_numpy(events['P_rad'], allow_missing=False)
        return prad
    bin_edges = np.linspace(0, 1, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_prad,
        bin_edges=bin_edges,
        x_label='P_rad',
        title='Control Plot: P_rad',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_prad.png")

    # charged_E
    def get_charged_E(events):
        charged_E = ak.to_numpy(events['charged_E'], allow_missing=False)
        return charged_E
    bin_edges = np.linspace(0, cme, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_charged_E,
        bin_edges=bin_edges,
        x_label='charged_E',
        title='Control Plot: charged_E',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_chargedE.png")

    ###################################################
    # Common event variables
    ###################################################

    # E/p of lead part in each hemisphere
    def get_lead_part_E_over_p(events):
        mask_is_lead_a = events['is_lead_a'] == 1
        mask_is_lead_b = events['is_lead_b'] == 1
        lead_a_E = ak.to_numpy(ak.firsts(events['Part_hpcTotalShowerEnergy'][mask_is_lead_a]), allow_missing=False)
        lead_b_E = ak.to_numpy(ak.firsts(events['Part_hpcTotalShowerEnergy'][mask_is_lead_b]), allow_missing=False)
        lead_a_p = ak.to_numpy(ak.firsts(events['Part_p4'][mask_is_lead_a].p), allow_missing=False)
        lead_b_p = ak.to_numpy(ak.firsts(events['Part_p4'][mask_is_lead_b].p), allow_missing=False)
        # lead_a_p = ak.to_numpy(events['lead_a_p4'].p, allow_missing=False)
        # lead_b_p = ak.to_numpy(events['lead_b_p4'].p, allow_missing=False)
        lead_a_E_over_p = lead_a_E / (lead_a_p + 1e-10) # avoid division by zero
        lead_b_E_over_p = lead_b_E / (lead_b_p + 1e-10)
        lead_parts_E_over_p = np.concatenate([lead_a_E_over_p, lead_b_E_over_p])
        weights = np.concatenate([events['weight'], events['weight']])
        return lead_parts_E_over_p, weights
    bin_edges = np.linspace(0, 2, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_lead_part_E_over_p,
        bin_edges=bin_edges,
        x_label='Lead Part E/p',
        title='Control Plot: Lead Part E/p',
        luminosity=luminosity, normalize=normalize,
        log_scale=True,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_parts_E_over_p.png")

    # hpcNumLayers of lead part in each hemisphere
    def get_lead_part_hpcNumLayers(events):
        lead_a_hpcNumLayers = ak.to_numpy(ak.firsts(events['Part_hpcNumLayers'][events['is_lead_a'] == 1]), allow_missing=False)
        lead_b_hpcNumLayers = ak.to_numpy(ak.firsts(events['Part_hpcNumLayers'][events['is_lead_b'] == 1]), allow_missing=False)
        lead_parts_hpcNumLayers = np.concatenate([lead_a_hpcNumLayers, lead_b_hpcNumLayers])
        weights = np.concatenate([events['weight'], events['weight']])
        return lead_parts_hpcNumLayers, weights
    bin_edges = np.linspace(-0.5, 20.5, 22)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_lead_part_hpcNumLayers,
        bin_edges=bin_edges,
        x_label='Lead Part hpcNumLayers',
        title='Control Plot: Lead Part hpcNumLayers',
        luminosity=luminosity, normalize=normalize,
        log_scale=True,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_parts_hpcNumLayers.png")

    # HT
    def get_ht(events):
        flag = ak.ones_like(events['Part_charge'], dtype=bool)
        p4_all = get_all_p4_from_ak_events(events, flag)
        ht = ak.sum(p4_all.pt, axis=-1)
        ht = ak.to_numpy(ht, allow_missing=False)
        return ht
    bin_edges = np.linspace(0, 100, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_ht,
        bin_edges=bin_edges,
        x_label='Ht [GeV]',
        title='Control Plot: HT',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_ht.png")

    # missing pT
    def get_missing_pt(events):
        missing_pt = ak.to_numpy(events['missing_pt'], allow_missing=False)
        return missing_pt
    bin_edges = np.linspace(0, 100, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_missing_pt,
        bin_edges=bin_edges,
        x_label='Missing pT [GeV]',
        title='Control Plot: Missing pT',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_missing_pt.png")


    # nprong
    def get_nprong(events):
        nprong = ak.to_numpy(events['nprong'], allow_missing=False)
        return nprong

    bin_edges = np.arange(2, 8, 1)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_nprong,
        bin_edges=bin_edges,
        x_label='nprong',
        title='Control Plot: nprong',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_nprong.png")


    # number of neutral particles 
    def get_n_neutral(events):
        neutral_mask = events['Part_charge'] == 0
        n_neutral = ak.to_numpy(ak.sum(neutral_mask, axis=-1), allow_missing=False)
        return n_neutral
    bin_edges = np.linspace(0, 10, 11)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_n_neutral,
        bin_edges=bin_edges,
        x_label='Number of Neutral Particles',
        title='Control Plot: Number of Neutral Particles',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_n_neutral.png")
    
    # -log10(1-thrust)
    def get_neglog1mthrust(events):
        thrust_magnitude = events['thrust_Mag']
        neglog1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
        return neglog1mthrust
    bin_edges = np.linspace(0, 10, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_neglog1mthrust,
        bin_edges=bin_edges,
        x_label=r'-log10(1 - thrust)',
        title=r'Control Plot: -log10(1 - thrust)',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_thrust_neglog1mthrust.png")


    ################################################
    # lead part in each hemisphere
    ################################################

    # Four momentum of lead part in each hemisphere
    for var in ['pt', 'theta', 'phi', 'E']:
        def get_lead_part_var(events):
            lead_a_var = ak.to_numpy( getattr(events['lead_a_p4'], var), allow_missing=False)
            lead_b_var = ak.to_numpy( getattr(events['lead_b_p4'], var), allow_missing=False)
            lead_parts_var = np.concatenate([lead_a_var, lead_b_var])
            weights = np.concatenate([events['weight'], events['weight']])
            return lead_parts_var, weights
        if var == 'pt':
            x_label = 'Lead Part pT [GeV]'
            title = 'Control Plot: Lead Part pT'
            bin_edges = np.linspace(0, 50, 21)
        elif var == 'theta':
            x_label = 'Lead Part Theta [rad]'
            title = 'Control Plot: Lead Part Theta'
            bin_edges = np.linspace(0, np.pi, 21)
        elif var == 'phi':
            x_label = 'Lead Part Phi [rad]'
            title = 'Control Plot: Lead Part Phi'
            bin_edges = np.linspace(-np.pi, np.pi, 21)
        elif var == 'E':
            x_label = 'Lead Part E [GeV]'
            title = 'Control Plot: Lead Part E'
            bin_edges = np.linspace(0, 50, 21)

        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            region_name=region_name,
            func_get_variable=get_lead_part_var,
            bin_edges=bin_edges,
            x_label=x_label,
            title=title,
            luminosity=luminosity, normalize=normalize,
            log_scale=log_scale,
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_plot_lead_parts_{var}.png")

    # four momentum of lead part pair
    for var in ['pt', 'theta', 'phi', 'E', 'M']:
        def get_lead_part_pair_var(events):
            lead_pair_p4 = events['lead_a_p4'] + events['lead_b_p4']
            lead_pair_var = ak.to_numpy( getattr(lead_pair_p4, var), allow_missing=False)
            return lead_pair_var
        if var == 'pt':
            x_label = 'Lead Part Pair pT [GeV]'
            title = 'Control Plot: Lead Part Pair pT'
            bin_edges = np.linspace(0, 100, 21)
        elif var == 'theta':
            x_label = 'Lead Part Pair Theta [rad]'
            title = 'Control Plot: Lead Part Pair Theta'
            bin_edges = np.linspace(0, np.pi, 21)
        elif var == 'phi':
            x_label = 'Lead Part Pair Phi [rad]'
            title = 'Control Plot: Lead Part Pair Phi'
            bin_edges = np.linspace(-np.pi, np.pi, 21)
        elif var == 'E':
            x_label = 'Lead Part Pair E [GeV]'
            title = 'Control Plot: Lead Part Pair E'
            bin_edges = np.linspace(0, 100, 21)
        elif var == 'M':
            x_label = 'Lead Part Pair Invariant Mass [GeV]'
            title = 'Control Plot: Lead Part Pair Invariant Mass'
            bin_edges = np.linspace(0, 100, 21)

        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            region_name=region_name,
            func_get_variable=get_lead_part_pair_var,
            bin_edges=bin_edges,
            x_label=x_label,
            title=title,
            luminosity=luminosity, normalize=normalize,
            log_scale=log_scale,
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_plot_lead_part_pair_{var}.png")


    # dR between lead parts in two hemispheres
    def get_lead_part_pair_angle(events):
        lead_a_p4 = events['lead_a_p4']
        lead_b_p4 = events['lead_b_p4']
        angle = lead_a_p4.deltaangle(lead_b_p4)
        angle = ak.to_numpy(angle, allow_missing=False)
        return angle
        # dR = lead_a_p4.deltaR(lead_b_p4)
        # dR = ak.to_numpy(dR, allow_missing=False)
        # return dR
    bin_edges = np.linspace(160/180*np.pi, 180/180*np.pi, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_lead_part_pair_angle,
        bin_edges=bin_edges,
        x_label='Angle between Lead Parts',
        title='Control Plot: Angle between Lead Parts',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_part_pair_angle.png")


    # invariant mass of lead visible p4
    def get_lead_visible_mass(events):
        if len(events) == 0:
            return np.array([])
        mass_list = []
        for key in ['a', 'b']:
            lead_vis_p4 = events[f'lead_{key}_visible_p4']
            mass = lead_vis_p4.mass
            mass_list.append(ak.to_numpy(mass, allow_missing=False))
        mass_all = ak.concatenate(mass_list, axis=-1)
        mass_all = ak.to_numpy(mass_all, allow_missing=False)
        return mass_all, np.concatenate([events['weight'], events['weight']])
    bin_edges = np.linspace(0, 2, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_lead_visible_mass,
        bin_edges=bin_edges,
        x_label='Invariant Mass of Lead Visible System [GeV]',
        title='Control Plot: Invariant Mass of Lead Visible System',
        luminosity=luminosity, normalize=normalize, log_scale=True,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_visible_mass.png")


def make_control_plots_pion(dl_dict, luminosity, normalize, output_dir, region_name="pion", log_scale=True):
    # plot pdg id for charged particles
    # for better visualization, first map others, pi, el, mu to 0, 1, 2, 3, then plot histogram with x-ticks showing the mapping
    map_pdgId = {
        41: 1,  # pi
        2: 2,   # el
        6: 3,   # mu
    }
    def get_charged_pdgId(events):
        charged_mask = events['Part_charge'] != 0
        charged_pdgId = ak.to_numpy(ak.flatten(events['Part_pdgId'][charged_mask]), allow_missing=False)
        charged_pdgId = np.abs(charged_pdgId)  
        # map pdgId for better visualization
        mask_others = np.ones_like(charged_pdgId, dtype=bool)
        for pdgId in map_pdgId.keys():
            mask_others = mask_others & (charged_pdgId != pdgId)
            charged_pdgId[charged_pdgId == pdgId] = map_pdgId[pdgId]
        charged_pdgId[mask_others] = 0
        broad_casted_weights = ak.broadcast_arrays(events['weight'], events['Part_pdgId'])[0][charged_mask]
        return charged_pdgId, broad_casted_weights

    bin_edges = np.linspace(-0.5, 3.5, 5)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_charged_pdgId,
        bin_edges=bin_edges,
        x_label='Charged Particle PDG ID',
        title='Control Plot: Charged Particle PDG ID',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Other', 'Pi', 'El', 'Mu'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_charged_pdgId.png")

    # number of photons in the event
    def get_n_photons(events):
        photon_mask = np.abs(events['Part_pdgId']) == 21
        n_photons = ak.to_numpy(ak.sum(photon_mask, axis=-1), allow_missing=False)
        return n_photons
    
    bin_edges = np.linspace(0, 10, 11)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_n_photons,
        bin_edges=bin_edges,
        x_label='Number of Photons',
        title='Control Plot: Number of Photons',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_n_photons.png")


    # number of pions in the event
    def get_n_pions(events):
        pion_mask = np.abs(events['Part_pdgId']) == 41
        n_pions = ak.to_numpy(ak.sum(pion_mask, axis=-1), allow_missing=False)
        return n_pions
    bin_edges = np.linspace(0, 8, 9)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_n_pions,
        bin_edges=bin_edges,
        x_label='Number of Pions',
        title='Control Plot: Number of Pions',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_n_pions.png")

    # number of leptons
    def get_n_leptons(events):
        lepton_mask = (np.abs(events['Part_pdgId']) == 2) | (np.abs(events['Part_pdgId']) == 6)
        n_leptons = ak.to_numpy(ak.sum(lepton_mask, axis=-1), allow_missing=False)
        return n_leptons
    bin_edges = np.linspace(0, 8, 9)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_n_leptons,
        bin_edges=bin_edges,
        x_label='Number of Leptons',
        title='Control Plot: Number of Leptons',
        luminosity=luminosity, normalize=normalize, log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_n_leptons.png")

    # plot total charge
    def get_total_charge(events):
        total_charge = ak.to_numpy(ak.sum(events['Part_charge'], axis=-1), allow_missing=False)
        return total_charge
    bin_edges = np.linspace(-2.5, 2.5, 6)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_total_charge,
        bin_edges=bin_edges,
        x_label='Total Charge',
        title='Control Plot: Total Charge',
        luminosity=luminosity, normalize=normalize, log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_total_charge.png")

    # plot hpcNumLayers for lead pion
    def get_lead_pion_hpcNumLayers(events):
        if len(events) == 0:
            return np.array([])
        hpcNumLayers_list = []
        for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
            # tmp_events = events[events[f'lead_{hemisphere_id}_is_pion'] == 1]
            tmp_events = events
            lead_pion_hpcNumLayers = ak.to_numpy(ak.firsts(tmp_events['Part_hpcNumLayers'][tmp_events[f'is_lead_{hemisphere_id}'] == 1]), allow_missing=False)
            lead_pion_hpcNumLayers = lead_pion_hpcNumLayers.flatten()
            hpcNumLayers_list.append(lead_pion_hpcNumLayers)
        hpcNumLayers_all = np.concatenate(hpcNumLayers_list)
        weights = np.concatenate([events['weight'], events['weight']])
        return hpcNumLayers_all, weights
    bin_edges = np.linspace(0, 11, 12)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_lead_pion_hpcNumLayers,
        bin_edges=bin_edges,
        x_label='Lead Pion hpcNumLayers',
        title='Control Plot: Lead Pion hpcNumLayers',
        luminosity=luminosity, normalize=normalize, log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_pion_hpcNumLayers.png")

    # plot pion-photon pair features
    def get_lead_pion_photon_pair_dR(events):
        if len(events) == 0:
            return np.array([])
        dr_list = []
        weight_list = []
        for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
            tmp_events = events[events[f'num_photon_near_lead_{hemisphere_id}'] > 0]
            pion_p4 = tmp_events[f'lead_{hemisphere_id}_p4']
            photon_mask = tmp_events[f'is_photon_near_lead_{hemisphere_id}'] == 1
            photon_p4 = tmp_events['Part_p4'][photon_mask]
            dr = pion_p4.deltaR(photon_p4)
            dr_list.append(ak.to_numpy(ak.flatten(dr, axis=-1), allow_missing=False))
            broad_casted_weights = ak.broadcast_arrays(tmp_events['weight'], tmp_events['Part_p4'])[0]
            weight = broad_casted_weights[photon_mask]
            weight_list.append(ak.to_numpy(ak.flatten(weight, axis=-1), allow_missing=False))
        dr_all = ak.concatenate(dr_list, axis=-1)
        dr_all = ak.to_numpy(dr_all, allow_missing=False)
        weights = np.concatenate(weight_list)
        return dr_all, weights
    bin_edges = np.linspace(0, 0.3, 51)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_lead_pion_photon_pair_dR,
        bin_edges=bin_edges,
        x_label='dR between Lead Pion and nearby Photons',
        title='Control Plot: dR between Lead Pion and nearby Photons',
        luminosity=luminosity, normalize=normalize, log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_pion_photon_pair_dR.png")

    def get_num_photon_near_lead_pion(events):
        if len(events) == 0:
            return np.array([])
        num_list = []
        for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
            # tmp_events = events[events[f'lead_{hemisphere_id}_is_pion'] == 1] 
            tmp_events = events
            photon_mask = tmp_events[f'is_photon_near_lead_{hemisphere_id}'] == 1
            num_photons = ak.to_numpy(ak.sum(photon_mask, axis=-1), allow_missing=False)
            num_list.append(num_photons)
        num_all = ak.concatenate(num_list, axis=-1)
        num_all = ak.to_numpy(num_all, allow_missing=False)
        return num_all, np.concatenate([events['weight'], events['weight']])
    bin_edges = np.linspace(0, 10, 11)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_num_photon_near_lead_pion,
        bin_edges=bin_edges,
        x_label='Number of Photons near Lead Pion',
        title='Control Plot: Number of Photons near Lead Pion',
        luminosity=luminosity, normalize=normalize, log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_pion_num_photon_nearby.png")

    # invariant mass of lead pion and nearby photon
    def get_lead_pion_nearby_photon_pair_mass(events):
        if len(events) == 0:
            return np.array([])
        mass_list = []
        weight_list = []
        for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
            tmp_events = events[events[f'num_photon_near_lead_{hemisphere_id}'] > 0]
            pion_p4 = tmp_events[f'lead_{hemisphere_id}_p4']
            photon_mask = tmp_events[f'is_photon_near_lead_{hemisphere_id}'] == 1
            photon_p4 = tmp_events['Part_p4'][photon_mask]
            sum_photon_p4 = get_sum_p4_from_ak_events(tmp_events, photon_mask)
            system_p4 = pion_p4 + sum_photon_p4
            pair_mass = system_p4.mass
            mass_list.append(ak.to_numpy(ak.flatten(pair_mass, axis=-1), allow_missing=False))
            weight_list.append(ak.to_numpy(tmp_events['weight'], allow_missing=False))
        mass_all = ak.concatenate(mass_list, axis=-1)
        mass_all = ak.to_numpy(mass_all, allow_missing=False)
        return mass_all, np.concatenate(weight_list)
    bin_edges = np.linspace(0, 2, 21)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        region_name=region_name,
        func_get_variable=get_lead_pion_nearby_photon_pair_mass,
        bin_edges=bin_edges,
        x_label='Invariant Mass of Lead Pion and Nearby Photons [GeV]',
        title='Control Plot: Invariant Mass of Lead Pion and Nearby Photons',
        luminosity=luminosity, normalize=normalize, log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_pion_nearby_photon_pair_mass.png")


def make_control_plots_vb0(dl_dict, luminosity, normalize, output_dir, region_name="pion", log_scale=True):
    """
        plot some post-nutrino reco variables
    """
    # check if missing mass reco variable is available
    if not (all(['lead_a_missing_p4' in dl.data[region_name].fields for dl in dl_dict.values()])):
        print("Missing neutrino reco variables, skip control plots.")
    else:
        # plot reconstructed tau p4
        for var in ['pt', 'theta', 'phi', 'E', 'M', 'pz']:
            def get_reco_tau(events):
                reco_tau_var_list = []
                flag_valid = events['flags_valid'] > 0
                weight = events['weight'][flag_valid]
                for key in ['a', 'b']:
                    # reco_tau_p4 = events[f'reco_tau_{key}_p4']
                    reco_tau_var = ak.to_numpy(getattr(events[f'reco_tau_{key}_p4'], var)[flag_valid], allow_missing=False)
                    reco_tau_var_list.append(reco_tau_var)
                reco_tau_var_all = np.concatenate(reco_tau_var_list, axis=-1)
                weights = np.concatenate([weight, weight])
                return reco_tau_var_all, weights
            if var == 'pt':
                x_label = 'Reconstructed Tau pT [GeV]'
                title = 'Control Plot: Reconstructed Tau pT'
                bin_edges = np.linspace(0, 50, 21)
            elif var == 'theta':
                x_label = 'Reconstructed Tau Theta [rad]'
                title = 'Control Plot: Reconstructed Tau Theta'
                bin_edges = np.linspace(0, np.pi, 21)
            elif var == 'phi':
                x_label = 'Reconstructed Tau Phi [rad]'
                title = 'Control Plot: Reconstructed Tau Phi'
                bin_edges = np.linspace(-np.pi, np.pi, 21)
            elif var == 'E':
                x_label = 'Reconstructed Tau E [GeV]'
                title = 'Control Plot: Reconstructed Tau E'
                bin_edges = np.linspace(0, 50, 21)
            elif var == 'M':
                x_label = 'Reconstructed Tau Mass [GeV]'
                title = 'Control Plot: Reconstructed Tau Mass'
                bin_edges = np.linspace(0, 50, 21)
            elif var == 'pz':
                x_label = 'Reconstructed Tau pz [GeV]'
                title = 'Control Plot: Reconstructed Tau pz'
                bin_edges = np.linspace(-50, 50, 21)
            fig, ax, ax_ratio = do_control_plot(
                dl_dict,
                region_name=region_name,
                func_get_variable=get_reco_tau,
                bin_edges=bin_edges,
                x_label=x_label,
                title=title,
                luminosity=luminosity, normalize=normalize, log_scale=log_scale,
            )
            plt.tight_layout()
            plt.savefig(f"{output_dir}/control_plot_reco_tau_{var}.png")
        
        # plot ditau system
        for var in ['pt', 'theta', 'phi', 'E', 'M', 'pz']:
            def get_reco_ditau(events):
                reco_ditau_var_list = []
                flag_valid = events['flags_valid'] > 0
                weight = events['weight'][flag_valid]
                reco_tau_a_p4 = events[f'reco_tau_a_p4'][flag_valid]
                reco_tau_b_p4 = events[f'reco_tau_b_p4'][flag_valid]
                reco_ditau_p4 = reco_tau_a_p4 + reco_tau_b_p4
                reco_ditau_var = ak.to_numpy(getattr(reco_ditau_p4, var), allow_missing=False)
                return reco_ditau_var, weight
            if var == 'pt':
                x_label = 'Reconstructed Ditau pT [GeV]'
                title = 'Control Plot: Reconstructed Ditau pT'
                bin_edges = np.linspace(0, 100, 21)
            elif var == 'theta':
                x_label = 'Reconstructed Ditau Theta [rad]'
                title = 'Control Plot: Reconstructed Ditau Theta'
                bin_edges = np.linspace(0, np.pi, 21)
            elif var == 'phi':
                x_label = 'Reconstructed Ditau Phi [rad]'
                title = 'Control Plot: Reconstructed Ditau Phi'
                bin_edges = np.linspace(-np.pi, np.pi, 21)
            elif var == 'E':
                x_label = 'Reconstructed Ditau E [GeV]'
                title = 'Control Plot: Reconstructed Ditau E'
                bin_edges = np.linspace(0, 100, 21)
            elif var == 'M':
                x_label = 'Reconstructed Ditau Mass [GeV]'
                title = 'Control Plot: Reconstructed Ditau Mass'
                bin_edges = np.linspace(0, 100, 21)
            elif var == 'pz':
                x_label = 'Reconstructed Ditau pz [GeV]'
                title = 'Control Plot: Reconstructed Ditau pz'
                bin_edges = np.linspace(-100, 100, 21)
            fig, ax, ax_ratio = do_control_plot(
                dl_dict,
                region_name=region_name,
                func_get_variable=get_reco_ditau,
                bin_edges=bin_edges,
                x_label=x_label,
                title=title,
                luminosity=luminosity, normalize=normalize, log_scale=log_scale,
            )
            plt.tight_layout()
            plt.savefig(f"{output_dir}/control_plot_reco_ditau_{var}.png")

        # # residual missing pt, E, and mass after accounting all visible particles and the lead neutrino reco
        # scale = 20
        # for var in ['pt', 'E', 'M', 'pz']:
        #     def get_residual_missing(events):
        #         residual_px = events['missing_p4'].px - events['lead_a_missing_p4'].px - events['lead_b_missing_p4'].px
        #         residual_py = events['missing_p4'].py - events['lead_a_missing_p4'].py - events['lead_b_missing_p4'].py
        #         residual_pz = events['missing_p4'].pz - events['lead_a_missing_p4'].pz - events['lead_b_missing_p4'].pz
        #         residual_E = events['missing_p4'].E - events['lead_a_missing_p4'].E - events['lead_b_missing_p4'].E
        #         if var == 'pt':
        #             residual_missing_var = np.sqrt(residual_px**2 + residual_py**2)
        #         elif var == 'E':
        #             residual_missing_var = residual_E
        #         elif var == 'M':
        #             mass2 = residual_E**2 - residual_px**2 - residual_py**2 - residual_pz**2
        #             residual_missing_var = np.where(mass2 >= 0, np.sqrt(mass2), -np.sqrt(-mass2)) # handle negative mass^2 due to reco imperfections
        #         elif var == 'pz':
        #             residual_missing_var = residual_pz
        #         residual_missing_var = ak.fill_none(residual_missing_var, 0) # replace None with 0
        #         return ak.to_numpy(residual_missing_var, allow_missing=False)

        #     if var == 'pt':
        #         x_label = 'Residual Missing pT [GeV]'
        #         title = 'Control Plot: Residual Missing pT'
        #         bin_edges = np.linspace(0, scale, 51)
        #     elif var == 'E':
        #         x_label = 'Residual Missing E [GeV]'
        #         title = 'Control Plot: Residual Missing E'
        #         bin_edges = np.linspace(-scale, scale, 51)
        #     elif var == 'M':
        #         x_label = 'Residual Missing Mass [GeV]'
        #         title = 'Control Plot: Residual Missing Mass'
        #         bin_edges = np.linspace(-scale, scale, 51)
        #     elif var == 'pz':
        #         x_label = 'Residual Missing pz [GeV]'
        #         title = 'Control Plot: Residual Missing pz'
        #         bin_edges = np.linspace(-scale, scale, 51)
        #     fig, ax, ax_ratio = do_control_plot(
        #         dl_dict,
        #         region_name=region_name,
        #         func_get_variable=get_residual_missing,
        #         bin_edges=bin_edges,
        #         x_label=x_label,
        #         title=title,
        #         luminosity=luminosity, normalize=normalize, log_scale=log_scale,
        #     )
        #     plt.tight_layout()
        #     plt.savefig(f"{output_dir}/control_plot_residual_missing_{var}.png")

        # dR between lead visible system and reconstructed missing momentum
        def get_dR_lead_visible_missing(events):
            dR_list = []
            events = events[events['flags_valid'] > 0]
            for key in ['a', 'b']:
                lead_vis_p4 = events[f'lead_{key}_visible_p4']
                missing_p4 = events[f'lead_{key}_missing_p4']
                dR = lead_vis_p4.deltaR(missing_p4)
                dR_list.append(ak.to_numpy(dR, allow_missing=False))
            dR_all = ak.concatenate(dR_list, axis=-1)
            dR_all = ak.to_numpy(dR_all, allow_missing=False)
            return dR_all, np.concatenate([events['weight'], events['weight']])
        bin_edges = np.linspace(0, 0.4, 51)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            region_name=region_name,
            func_get_variable=get_dR_lead_visible_missing,
            bin_edges=bin_edges,
            x_label='dR between Lead Visible System and Missing Momentum',
            title='Control Plot: dR between Lead Visible System and Missing Momentum',
            luminosity=luminosity, normalize=normalize, log_scale=log_scale,
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_plot_dR_lead_visible_missing.png")



class ControlPlotProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        """
        Processor to make control plots for data/MC comparison.
        """
        super().__init__(config)
        self.config = config
        if 'output_dir_name' in config:
            output_dir = f"{output_dir}/{config['output_dir_name']}"
        else:
            output_dir = f"{output_dir}/control_plots/"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.luminosity = config.get('luminosity', None) 
        self.normalize = False
        self.regions = config.get('regions', ['hadhad'])
        self.verbosity = config.get('verbosity', 1)
        # self.normalize = True

    def run(self, dl_dict):
        # plot QI observables for hadhad region: verbose level 0
        if self.verbosity >= 0:
            for region in self.regions:
                print(f"Processing {region} region: plotting quantum observables")
                output_dir = f"{self.output_dir}/{region}/"
                os.makedirs(output_dir, exist_ok=True)

                make_control_plots_vb0(
                    dl_dict,
                    luminosity=self.luminosity,
                    normalize=self.normalize,
                    output_dir=output_dir,
                    region_name=region,
                    log_scale=False,
                )

        # common control plots: verbose level 1
        if self.verbosity >= 1:
            for region in self.regions:
                output_dir_tautau = f"{self.output_dir}/{region}/"
                os.makedirs(output_dir_tautau, exist_ok=True)
                make_control_plots_tautau(
                    dl_dict,
                    luminosity=self.luminosity,
                    normalize=self.normalize,
                    output_dir=output_dir_tautau,
                    region_name=region,
                    log_scale=False,
                )
            
            for hadhad_region_name in ['hadhad', 'pipi', 'pirho', 'rhopi']:
                if hadhad_region_name in self.regions:
                    print(f"Processing {hadhad_region_name} region")
                    output_dir_hadhad = f"{self.output_dir}/{hadhad_region_name}/"
                    os.makedirs(output_dir_hadhad, exist_ok=True)
                    make_control_plots_pion(
                        dl_dict,
                        luminosity=self.luminosity,
                        normalize=self.normalize,
                        output_dir=output_dir_hadhad,
                        region_name=hadhad_region_name,
                        log_scale=False,
                    )



    def finalize(self):
        pass