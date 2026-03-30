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
            get_all_p4_from_ak_events, cme, rebuild_p4
from quantum.observables_builder import build_observables, get_mean_and_err_of_mean

log = logging.getLogger(__name__)

def filter_hadhad_region(events: ak.Array, filter_log_dict: dict):
    filter_log_dict['hadhad region initial'] = filter_log_dict.get('hadhad region initial', 0) + len(events)

    recpart_pdgid = events['Part_pdgId']
    recpart_abspdgid = abs(recpart_pdgid)
    recpart_charge = events['Part_charge']

    # require exactly two pions with opposite charge
    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)
    flag_is_pion = (recpart_abspdgid == 41)
    pass_filter = (ak.sum(flag_is_pion, axis=1) == 2) & pass_filter
    charge_ary_of_pions = recpart_charge[flag_is_pion]
    flag_opposite_charge = (ak.sum(charge_ary_of_pions, axis=1) == 0)
    pass_filter = flag_opposite_charge & pass_filter
    filter_log_dict['2 pions with opposite charge'] = filter_log_dict.get('2 pions with opposite charge', 0) + ak.sum(pass_filter)

    # num_photon_near_leading_pion <= 3
    for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
        num_photons = ak.sum(events[f'is_photon_near_lead_{hemisphere_id}'], axis=1)
        pass_filter = (num_photons <= 3) & pass_filter
    filter_log_dict[f'number of photons near leading pion <= 3 in hemisphere {hemisphere_id}'] = filter_log_dict.get(f'number of photons near leading pion <= 3 in hemisphere {hemisphere_id}', 0) + ak.sum(pass_filter)

    events_hadhad = events[pass_filter]

    return events_hadhad, filter_log_dict

def filter_inclusive_tautau_loose(events: ak.Array, filter_log_dict: dict):
    filter_log_dict['inclusive tautau initial'] = filter_log_dict.get('inclusive tautau initial', 0) + len(events)

    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

    events['nprong'] = ak.sum((events['Part_charge'] != 0), axis=1)
    pass_filter = (events['nprong'] >= 2) & (events['nprong'] <= 6) & pass_filter
    events = events[pass_filter]
    pass_filter = pass_filter[pass_filter]

    filter_log_dict['Chaarge multiplicity in [2, 6]'] = filter_log_dict.get('Chaarge multiplicity in [2, 6]', 0) + len(events)

    # define hemisphere by the sign of dot product between particle momentum and thrust vector
    events['Part_p'] = events['Part_p4'].p
    events['truthst_vector'] = vector.zip({
        "x": events['thrust_x'],
        "y": events['thrust_y'],
        "z": events['thrust_z'],
    })
    # events['Part_hemisphere'] = ak.where(events['Part_p4'].to_Vector3D().dot(events['truthst_vector']) > 0, 1, -1)
    hemisphere_id = ak.where(events['Part_p4'].to_Vector3D().dot(events['truthst_vector']) > 0, 1, -1)

    # find the charge, p4 and pdgId of leading particle in each hemisphere
    idx_all = ak.local_index(events['Part_pdgId'])
    lead_particle_flags = {1: None, -1: None}
    lead_particle_charges = {1: None, -1: None}
    for hemisphere in [1, -1]:
        mask = (hemisphere_id == hemisphere) & (events['Part_charge'] != 0)
        p4 = events['Part_p4'][mask]
        idx_particle = idx_all[mask]
        idx_sorted = ak.argsort(p4.p, axis=1, ascending=False)
        particle_idx_sorted = idx_particle[idx_sorted]
        leading_particle_idx = ak.firsts(particle_idx_sorted)
        flag_is_leading_particle = (idx_all == leading_particle_idx)
        lead_particle_flags[hemisphere] = flag_is_leading_particle
        lead_particle_charges[hemisphere] = events['Part_charge'][flag_is_leading_particle][...,0]
    
    # redefine hemisphere id based on charge of leading particle in each hemisphere, if exists
    switch_hemisphere = (lead_particle_charges[1] < 0) & (lead_particle_charges[-1] > 0)
    events['Part_hemisphere'] = ak.where(switch_hemisphere, -hemisphere_id, hemisphere_id)
    # now lead_a is the leading charged particle in the hemisphere with positive leading particle, and lead_b is the leading charged particle in the hemisphere with negative leading particle
    events['is_lead_a'] = lead_particle_flags[1] & (events['Part_charge'] > 0) | lead_particle_flags[-1] & (events['Part_charge'] > 0)
    events['is_lead_b'] = lead_particle_flags[1] & (events['Part_charge'] < 0) | lead_particle_flags[-1] & (events['Part_charge'] < 0)
    
    # store p4, pdgId and other info of leading particle in each hemisphere
    for hemisphere in ['a', 'b']:
        lead_flag = events[f'is_lead_{hemisphere}']
        events[f'lead_{hemisphere}_valid'] = ak.any(lead_flag, axis=1)
        lead_px = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fX'][lead_flag]), 0)
        lead_py = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fY'][lead_flag]), 0)
        lead_pz = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fZ'][lead_flag]), 0)
        lead_E = ak.fill_none(ak.firsts(events['Part_fourMomentum_fCoordinates_fT'][lead_flag]), 0)
        events[f'lead_{hemisphere}_p4'] = vector.zip({
            "px": lead_px,
            "py": lead_py,
            "pz": lead_pz,
            "E": lead_E,
        })
        events[f'lead_{hemisphere}_pdgId'] = ak.fill_none(ak.firsts(events['Part_pdgId'][lead_flag]), 0)
        events[f'lead_{hemisphere}_charge'] = ak.fill_none(ak.firsts(events['Part_charge'][lead_flag]), 0)
        events[f'lead_{hemisphere}_hpcTotalShowerEnergy'] = ak.fill_none(ak.firsts(events['Part_hpcTotalShowerEnergy'][lead_flag]), 0)
        events[f'lead_{hemisphere}_z0'] = ak.fill_none(ak.firsts(events['Trac_impParToVertexZ'][lead_flag]), -999)
        events[f'lead_{hemisphere}_d0'] = ak.fill_none(ak.firsts(events['Trac_impParToVertexRPhi'][lead_flag]), -999)
        events[f'lead_{hemisphere}_is_pion'] = ak.any(lead_flag & (abs(events['Part_pdgId']) == 41), axis=1)

        # match photons near the leading particle in each hemisphere by dR
        dR_threshold = 0.3
        lead_p4 = events[f'lead_{hemisphere}_p4']
        part_p4 = events['Part_p4']
        dR_to_lead = lead_p4.deltaR(part_p4)
        events[f'Part_dR_to_lead_{hemisphere}'] = dR_to_lead

        photon_mask = (events['Part_pdgId'] == 21) & (events['Part_charge'] == 0)
        hemisphere_id = 1 if hemisphere == 'a' else -1
        nearby_photon_mask = (dR_to_lead < dR_threshold) & (photon_mask) & (events['Part_hemisphere'] == hemisphere_id)
        events[f'is_photon_near_lead_{hemisphere}'] = nearby_photon_mask
        events[f'has_pion_photon_pair_{hemisphere}'] = ak.any(nearby_photon_mask, axis=1) & events[f'lead_{hemisphere}_is_pion'] 


    pass_filter = events['lead_a_valid'] & events['lead_b_valid'] & pass_filter
    filter_log_dict['leading charged particle in each hemisphere'] = filter_log_dict.get('leading charged particle in each hemisphere', 0) + ak.sum(pass_filter)

    cut_lead_a = (np.abs(events['lead_a_p4'].costheta) > 0.035) & (np.abs(events['lead_a_p4'].costheta) < 0.731)
    cut_lead_b = (np.abs(events['lead_b_p4'].costheta) > 0.035) & (np.abs(events['lead_b_p4'].costheta) < 0.731)
    pass_filter = (cut_lead_a | cut_lead_b) & pass_filter

    pass_filter = (np.abs(events['lead_a_z0']) < 4.5) & (np.abs(events['lead_b_z0']) < 4.5) & pass_filter
    pass_filter = ((np.abs(events['lead_a_d0']) < 0.3) | (np.abs(events['lead_b_d0']) < 0.3)) & pass_filter

    events = events[pass_filter]
    pass_filter = pass_filter[pass_filter]
    filter_log_dict['angular and vertex cuts on leading charged particles'] = filter_log_dict.get('angular and vertex cuts on leading charged particles', 0) + ak.sum(pass_filter)


    # sum charged E
    charged_E = ak.sum(events['Part_fourMomentum_fCoordinates_fT'] * (events['Part_charge'] != 0), axis=-1)
    charged_E = charged_E + ak.sum(events['Part_hpcTotalShowerEnergy'] * (events['Part_charge'] == 0), axis=-1)
    events['charged_E'] = charged_E
    pass_filter = (charged_E > 0.0875 * cme) & pass_filter
    filter_log_dict['charged energy > 0.0875*Ecm'] = filter_log_dict.get('charged energy > 0.0875*Ecm', 0) + ak.sum(pass_filter)


    events['missing_px'] = -ak.sum(events['Part_fourMomentum_fCoordinates_fX'] * (events['Part_charge'] != 0), axis=-1)
    events['missing_py'] = -ak.sum(events['Part_fourMomentum_fCoordinates_fY'] * (events['Part_charge'] != 0), axis=-1)
    # events['missing_pz'] = -ak.sum(events['Part_fourMomentum_fCoordinates_fZ'], axis=-1)
    events['missing_pt'] = np.sqrt(events['missing_px']**2 + events['missing_py']**2)
    pass_filter = (((events['nprong'] == 2) & (events['missing_pt'] > 0.4)) | (events['nprong'] != 2)) & pass_filter
    filter_log_dict['missing pt > 0.4 for 2-prong events'] = filter_log_dict.get('missing pt > 0.4 for 2-prong events', 0) + ak.sum(pass_filter)

    # define isolation angle: minimum angle between any charged particles in different hemisphere
    pairs = ak.cartesian({
        'a': events['Part_p4'][(events['Part_charge'] != 0) & (events['Part_hemisphere'] == 1)], 
        'b': events['Part_p4'][(events['Part_charge'] != 0) & (events['Part_hemisphere'] == -1)]
    }, nested=False, axis=1)
    angle_between_charged = pairs['a'].deltaangle(pairs['b']) * 180 / np.pi
    min_angle_between_charged = ak.min(angle_between_charged, axis=-1)
    events['isolation_angle'] = ak.fill_none(min_angle_between_charged, -1)

    # E_rad
    for hemisphere in [1, -1]:
        hemisphere_id = 'a' if hemisphere == 1 else 'b'
        lead_p4 = events[f'lead_{hemisphere_id}_p4']
        part_p4 = events['Part_p4']
        angle_to_lead = lead_p4.deltaangle(part_p4) * 180 / np.pi
        nearby_part_mask = angle_to_lead < 30
        nearby_hpc_energy = ak.sum(events['Part_hpcTotalShowerEnergy'][nearby_part_mask], axis=-1)
        events[f'{hemisphere_id}_nearby_hpc_energy'] = nearby_hpc_energy

    events['E_rad'] = (events['a_nearby_hpc_energy']**2 + events['b_nearby_hpc_energy']**2)**0.5 / (cme/2)
    # P_rad
    events['P_rad'] = (events['lead_a_p4'].p**2 + events['lead_b_p4'].p**2)**0.5 / (cme/2)

    # isolation angle < 179.5 for 2-prong events
    pass_filter = (((events['nprong'] == 2) & (events['isolation_angle'] < 179.5)) | (events['nprong'] != 2)) & pass_filter
    filter_log_dict['acollinearity for 2-prong events > 0.5 degree'] = filter_log_dict.get('acollinearity for 2-prong events > 0.5 degree', 0) + ak.sum(pass_filter)

    pass_filter = ak.fill_none(pass_filter, False)
    events = events[pass_filter]

    return events, filter_log_dict


def filter_inclusive_tautau_tight(events: ak.Array, filter_log_dict: dict):
    filter_log_dict['inclusive tautau tight initial'] = filter_log_dict.get('inclusive tautau tight initial', 0) + len(events)
    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

    # isolation angle > 160 degree
    pass_filter = (events['isolation_angle'] > 160) & pass_filter
    filter_log_dict['isolation angle > 160 degree'] = filter_log_dict.get('isolation angle > 160 degree', 0) + ak.sum(pass_filter)

    # Erad Prad
    pass_filter = (events['E_rad'] < 0.8) & (events['P_rad'] < 1) & pass_filter
    filter_log_dict['E_rad < 0.8 and P_rad < 1'] = filter_log_dict.get('E_rad < 0.8 and P_rad < 1', 0) + ak.sum(pass_filter)

    pass_filter = ak.fill_none(pass_filter, False)
    events = events[pass_filter]
    return events, filter_log_dict

        
def filter_pion_events(events: ak.Array, filter_log_dict: dict):
    filter_log_dict['pion region initial'] = filter_log_dict.get('pion region initial', 0) + len(events)
    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

    # thrust within range
    thrust_magnitude = events['thrust_Mag']
    neglog1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
    pass_filter = (neglog1mthrust > 2.5) & (neglog1mthrust < 4.5) & pass_filter
    filter_log_dict['thrust within [2.5, 4.5]'] = filter_log_dict.get('thrust within [2.5, 4.5]', 0) + ak.sum(pass_filter)

    # nprong = 2
    pass_filter = (events['nprong'] == 2) & pass_filter
    filter_log_dict['nprong=2'] = filter_log_dict.get('nprong=2', 0) + ak.sum(pass_filter)

    # event contains at least one charged pion
    flag_is_pion = (abs(events['Part_pdgId']) == 41) & (events['Part_charge'] != 0)
    pass_filter = ak.any(flag_is_pion, axis=1) & pass_filter
    filter_log_dict['at least one charged pion'] = filter_log_dict.get('at least one charged pion', 0) + ak.sum(pass_filter)

    # event contains at no charged pdgId=0, 42, or 65 particles (unk, kaon, proton)
    flag_charged_unk_kaon_proton = ak.any(((abs(events['Part_pdgId']) == 0) | (abs(events['Part_pdgId']) == 42) | (abs(events['Part_pdgId']) == 65)) & (events['Part_charge'] != 0), axis=1)
    pass_filter = pass_filter & ~flag_charged_unk_kaon_proton 
    filter_log_dict['no charged unk/kaon/proton'] = filter_log_dict.get('no charged unk/kaon/proton', 0) + ak.sum(pass_filter)

    # finalize
    pass_filter = ak.fill_none(pass_filter, False)
    events = events[pass_filter]
    return events, filter_log_dict

def filter_pilep_events(events: ak.Array, filter_log_dict: dict):
    filter_log_dict['pilep region initial'] = filter_log_dict.get('pilep region initial', 0) + len(events)
    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

    # find leading pion in each hemisphere
    for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
        events[f'lead_{hemisphere_id}_is_e'] = (events['Part_hemisphere'] == hemisphere) & (events['Part_charge'] != 0) & (abs(events['Part_pdgId']) == 2)
        events[f'lead_{hemisphere_id}_is_mu'] = (events['Part_hemisphere'] == hemisphere) & (events['Part_charge'] != 0) & (abs(events['Part_pdgId']) == 6)
        events[f'has_lead_{hemisphere_id}_e'] = ak.any(events[f'lead_{hemisphere_id}_is_e'], axis=1)
        events[f'has_lead_{hemisphere_id}_mu'] = ak.any(events[f'lead_{hemisphere_id}_is_mu'], axis=1)
        events[f'has_lead_{hemisphere_id}_lepton'] = events[f'has_lead_{hemisphere_id}_e'] | events[f'has_lead_{hemisphere_id}_mu']
    
    pass_filter = (events['has_lead_a_lepton'] | events['has_lead_b_lepton']) & pass_filter
    filter_log_dict['at least one leading lepton (e or mu) in either hemisphere'] = filter_log_dict.get('at least one leading lepton (e or mu) in either hemisphere', 0) + ak.sum(pass_filter)

    # number of photons near leading pion == 0
    pass_filter = ( (events['lead_a_is_pion'] * ak.sum(events['is_photon_near_lead_a'], axis=1)) + (events['lead_b_is_pion'] * ak.sum(events['is_photon_near_lead_b'], axis=1)) == 0 ) & pass_filter
    filter_log_dict['no photons near leading pion'] = filter_log_dict.get('no photons near leading pion', 0) + ak.sum(pass_filter)

    # # number of photons <= 1
    # pass_filter = (ak.sum((events['Part_pdgId'] == 21), axis=1) <= 1) & pass_filter
    # print(f"Filter efficiency for pilep region after photon multiplicity cut: {ak.sum(pass_filter) / len(events):.4f}")

    events = events[pass_filter]
    return events, filter_log_dict


def filter_event(events: ak.Array, filter_log_dict: dict, input_region='raw', regions_to_load=[], is_Ztautau=False):
    # redefine Part_p4
    events['Part_p4'] = vector.zip(
        {
            "px": events['Part_fourMomentum_fCoordinates_fX'],
            "py": events['Part_fourMomentum_fCoordinates_fY'],
            "pz": events['Part_fourMomentum_fCoordinates_fZ'],
            "E": events['Part_fourMomentum_fCoordinates_fT'],
        }
    )
    # redefine the lead_a/b_p4
    if f'lead_a_p4' in events.fields and f'lead_b_p4' in events.fields:
        for part in ['a', 'b']:
            events[f'lead_{part}_p4'] = vector.zip(
                {
                    "px": events[f'lead_{part}_p4'].x,
                    "py": events[f'lead_{part}_p4'].y,
                    "pz": events[f'lead_{part}_p4'].z,
                    "E": events[f'lead_{part}_p4'].t,
                }
            )
    
    # define some truth-level variables if not already defined
    dR_threshold = 0.3
    if (not 'GenPart_is_final_state_near_tau_a' in events.fields) and is_Ztautau:
        genpart_p4 = get_all_p4_from_ak_events(events, ak.ones_like(events['GenPart_pdgId'], dtype=bool), 'GenPart_vector')
        for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
            # -15 - tau+ - a, 15 - tau- - b
            tau_mask = (events['GenPart_pdgId'] == -15 * hemisphere)
            tau_p4 = get_p4_from_ak_events(events, tau_mask, 'GenPart_vector')
            dR_to_tau = tau_p4.deltaR(genpart_p4)
            events[f'GenPart_is_final_state_near_tau_{hemisphere_id}'] = (dR_to_tau < dR_threshold) & (events['GenPart_status'] == 1)
            events[f'GenPart_is_photon_near_tau_{hemisphere_id}'] = events[f'GenPart_is_final_state_near_tau_{hemisphere_id}'] & (events['GenPart_pdgId'] == 22)
            events[f'GenPart_is_pion_near_tau_{hemisphere_id}'] = events[f'GenPart_is_final_state_near_tau_{hemisphere_id}'] & (abs(events['GenPart_pdgId']) == 211)
            events[f'truth_num_photon_near_tau_{hemisphere_id}'] = ak.sum(events[f'GenPart_is_photon_near_tau_{hemisphere_id}'], axis=1)
            events[f'truth_num_pion_near_tau_{hemisphere_id}'] = ak.sum(events[f'GenPart_is_pion_near_tau_{hemisphere_id}'], axis=1)

    filtered_events_dict = {
        input_region: events
    }

    load_all = (len(regions_to_load) == 0)
    # select inclusive tautau loose from raw events.
    if load_all or ('inclusive_tautau_loose' in regions_to_load): 
        if not 'raw' in filtered_events_dict:
            log.error("Raw events not found in filtered_events_dict. Cannot apply filter_inclusive_tautau_loose.")
            return {}, filter_log_dict
        filtered_events, filter_log_dict = filter_inclusive_tautau_loose(filtered_events_dict['raw'], filter_log_dict)
        filtered_events_dict['inclusive_tautau_loose'] = filtered_events

    # select inclusive tautau tight from inclusive tautau loose
    if load_all or ('tautau' in regions_to_load):
        if not 'inclusive_tautau_loose' in filtered_events_dict:
            log.error("Inclusive tautau loose events not found in filtered_events_dict. Cannot apply filter_inclusive_tautau_tight.")
            return {}, filter_log_dict
        filtered_events, filter_log_dict = filter_inclusive_tautau_tight(filtered_events_dict['inclusive_tautau_loose'], filter_log_dict)
        filtered_events_dict['tautau'] = filtered_events

    # select pion region from inclusive tautau region. 
    if load_all or ('pion' in regions_to_load):
        if not 'tautau' in filtered_events_dict:
            log.error("Inclusive tautau events not found in filtered_events_dict. Cannot apply filter_pion_events.")
            return {}, filter_log_dict
        filtered_events, filter_log_dict = filter_pion_events(filtered_events_dict['tautau'], filter_log_dict)
        filtered_events_dict['pion'] = filtered_events

    # select pilep regions from pion region
    if load_all or ('pilep' in regions_to_load):
        if not 'pion' in filtered_events_dict:
            log.error("Pion events not found in filtered_events_dict. Cannot apply filter_pilep_events.")
            return {}, filter_log_dict
        filtered_events, filter_log_dict = filter_pilep_events(filtered_events_dict['pion'], filter_log_dict)
        filtered_events_dict['pilep'] = filtered_events

        # further define piele and pimu regions inside pilep region
        # piele region: events with at least one leading electron in either hemisphere
        pilep_events = filtered_events_dict['pilep']
        piele_events = pilep_events[pilep_events['has_lead_a_e'] | pilep_events['has_lead_b_e']]
        filter_log_dict['piele region'] = filter_log_dict.get('piele region', 0) + len(piele_events)
        filtered_events_dict['piele'] = piele_events

        # pimu region: events with at least one leading muon in either hemisphere
        pimu_events = pilep_events[pilep_events['has_lead_a_mu'] | pilep_events['has_lead_b_mu']]
        filter_log_dict['pimu region'] = filter_log_dict.get('pimu region', 0) + len(pimu_events)
        filtered_events_dict['pimu'] = pimu_events

    if load_all or ('hadhad' in regions_to_load):
        if not 'pion' in filtered_events_dict:
            log.error("Pion events not found in filtered_events_dict. Cannot apply filter_hadhad_region.")
            return {}, filter_log_dict
        filtered_events, filter_log_dict = filter_hadhad_region(filtered_events_dict['pion'], filter_log_dict)
        filtered_events_dict['hadhad'] = filtered_events

    if (not is_Ztautau) and ('raw' in filtered_events_dict):
        del filtered_events_dict['raw']

    return filtered_events_dict, filter_log_dict


class DataLoader:
    def __init__(self, config, output_dir):
        self.config = config
        # load all config into member variables
        for key, value in config.items():
            setattr(self, key, value)
        self.norm_factor = config.get("norm_factor", 1.0)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.name = self.config.get("name", "")
        self.tree_name = self.config.get("tree_name", "t")
        self.input_files = self.config.get("input_files", [])
        self.region_of_interest = self.config.get("region_of_interest", "pipi")
        self.load_regions = self.config.get("load_regions", [self.region_of_interest])
        self.load_regions = list(set(self.load_regions)) # remove duplicates
        self.is_data = self.config.get("is_data", False)
        self.initial_total_num_events = 0
        self.luminosity = self.config.get("luminosity", 0)
        if self.luminosity == 0 and not self.is_data:
            log.warning("Luminosity is set to 0 for MC sample. Please set luminosity in config for proper normalization.")

        self.is_Ztautau = "Ztautau" in self.name

        if not self.input_files:
            raise ValueError("Input files must be specified.")
        elif isinstance(self.input_files, str):
            self.input_files = glob.glob(self.input_files)
        else:
            all_files = []
            for pattern in self.input_files:
                all_files.extend(glob.glob(pattern))
            # sort files for consistency
            all_files = sorted(all_files)
            self.input_files = all_files

        self.data = {}
        self.filter_results = {
            'initial_total_num_events': 0,
        }

        _data_loaded = False
        if len(glob.glob(self.output_dir + f"/filtered___{self.region_of_interest}.parquet")) > 0:
            log.info(f"Loading existing filtered data from {self.output_dir}")
            for region in self.load_regions:
                file = self.output_dir + "/filtered___" + region + ".parquet"
                if os.path.exists(file):
                    self.data[region] = ak.from_parquet(file)
                    if len(self.data[region]) == 0:
                        log.warning(f"Filtered data for region {region} is empty. This may be due to previous filtering steps removing all events. Creating empty array for this region.")
                        empty_events = next(iter(self.data.values())) # get the structure of events from any existing region
                        filter_events = ak.zeros_like(empty_events['evtNumber'], dtype=bool)
                        self.data[region] = empty_events[filter_events]
                    if self.initial_total_num_events == 0:
                        self.initial_total_num_events = self.data[region]['initial_total_num_events'][0]
                    _data_loaded = True
                else:
                    log.warning(f"Filtered data file for region {region} does not exist. Re-loading and filtering data from input files.")
                    _data_loaded = False
                    break

        if not _data_loaded:
            self.load_data()
            self.save_data()
            keys = list(self.data.keys())
            for key in keys:
                if key not in self.load_regions:
                    del self.data[key]
            _data_loaded = True

        log.info(f"DataLoader initialization complete. Loaded {len(all_files)} files.")

    

    def load_data(self) -> pd.DataFrame:
        # if filtered___tautau_loose is already loaded, start from there
        region_init = "inclusive_tautau_loose"
        # only load cutflow log before the "region_init_cutflow_name"
        # region_init_cutflow_name = "inclusive tautau tight initial"
        region_init_cutflow_name = "isolation angle > 160 degree"

        if os.path.exists(self.output_dir + f"/filtered___{region_init}.parquet"):
            log.info(f"Loading existing raw data from {self.output_dir}/filtered___{region_init}.parquet")
            self.data[region_init] = ak.from_parquet(self.output_dir + f"/filtered___{region_init}.parquet")
            self.initial_total_num_events = self.data[region_init]['initial_total_num_events'][0]
            # load existing cutflow log if exists
            if os.path.exists(self.output_dir + f"/cutflow_{self.name}.txt"):
                log.info(f"Loading existing cutflow log from {self.output_dir}/cutflow_{self.name}.txt")
                with open(self.output_dir + f"/cutflow_{self.name}.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines[1:]: # skip header
                        parts = line.split()
                        if len(parts) >= 4:
                            cut_name = ' '.join(parts[:-3]) 
                            if cut_name != region_init_cutflow_name:
                                num_events = int(parts[-3])
                                self.filter_results[cut_name] = num_events
                            else:
                                break
            else:
                log.warning(f"Cutflow log does not exist. Will create new cutflow log after filtering.")
                self.filter_results['initial_total_num_events'] = self.initial_total_num_events

            # skip existing regions
            existing_regions = []
            for existing_files in glob.glob(self.output_dir + f"/filtered___*.parquet"):
                existing_region = os.path.basename(existing_files).split("___")[1].split(".parquet")[0]
                existing_regions.append(existing_region)
                if existing_region != region_init and existing_region in self.load_regions:
                    self.data[existing_region] = ak.from_parquet(existing_files)

            regions_to_load = list(set(self.load_regions) - set(existing_regions))
            log.info(f"regions to load and filter from {region_init}: {regions_to_load}")
            filtered_events, self.filter_results = filter_event(self.data[region_init], self.filter_results, input_region=region_init, regions_to_load=regions_to_load, is_Ztautau=self.is_Ztautau)
            for key, evt in filtered_events.items():
                self.data[key] = evt
                self.data[key]['initial_total_num_events'] = self.initial_total_num_events
        else:
            log.info("Loading data from input files.")
            # Identify branches to load
            f = ur.open(self.input_files[0])
            tree = f[self.tree_name]

            common_evt_branches = ["Event_evtNumber", "Event_totalChargedEnergy", "Event_totalEMEnergy", "Event_totalHadronicEnergy", "thrust_Mag", "thrust_x", "thrust_y", "thrust_z", "nGoodPart", 
                "event_category"
            ]
            gen_part_branches = ["pdgId", "status", "vector_fCoordinates_fX", "vector_fCoordinates_fY", "vector_fCoordinates_fZ", "vector_fCoordinates_fT"]
            gen_part_branches = [f"GenPart_{b}" for b in gen_part_branches]
            
            part_branches = [
                "charge", "pdgId", "fourMomentum_fCoordinates_fX", "fourMomentum_fCoordinates_fY", "fourMomentum_fCoordinates_fZ", "fourMomentum_fCoordinates_fT", "isGood", "vtxIdx", 
                "hpcShowerEnergy", "hpcShowerTheta", "hpcShowerPhi", "hpcParticleCode", "hpcNumLayers", "hpcLayerHitPattern", "hpcNumAssociatedShowers", "hpcTotalShowerEnergy", 
                "hacShowerEnergy", "hacShowerTheta", "hacShowerPhi", "hacParticleCode", "hacNumTowers", "hacTowerHitPattern", "hacNumAssociatedShowers", "hacTotalShowerEnergy", 
                "sticShowerEnergy", "sticShowerTheta", "sticShowerPhi", "sticNumTowers", "sticChargedTag", "sticSiliconVertexPos", 
                "lock",
            ]
            part_branches = [f'Part_{b}' for b in part_branches]
            id_branches = [
                "Elid_partIdx", "Elid_tag", "Elid_gammaConversion",
                "Muid_partIdx", "Muid_tag",
                "Haid_pionRich", "Haidn_pionTag", "Haidr_pionTag", "Haide_pionTag", "Haidc_pionTag"
            ]
            track_branches = [ f'Trac_{b}' for b in 
                [
                    "originVtxIdx", "impParToVertexRPhi", "impParToVertexZ", "impParRPhi", "impParZ",
                ]
            ]

            part_branches = part_branches + id_branches + track_branches

            vertex_branches = [ f'Vtx_{b}' for b in 
                ["position_fCoordinates_fX", "position_fCoordinates_fY", "position_fCoordinates_fZ",]
            ]

            branches_to_load = common_evt_branches + part_branches + vertex_branches 
            if not self.is_data:
                branches_to_load += gen_part_branches

            # Load data from all files
            initial_total_num_events = 0
            for file in self.input_files:
                log.info(f"Loading data from file: {file}")
                try:
                    f = ur.open(file)
                    tree = f[self.tree_name]
                    # load all events as awkward array 
                    events = tree.arrays(branches_to_load, library="ak")

                    # adjust event index to be unique across files
                    # the original Event_evtNumber starts from 1 for each file
                    if len(events) == 0:
                        continue
                    events['evtNumber'] = events['Event_evtNumber'] + initial_total_num_events
                    events['initial_total_num_events'] = len(events)
                    initial_total_num_events += len(events)

                    # select Part_xxx via isGood flag
                    part_abscosth = abs(events['Part_fourMomentum_fCoordinates_fZ']) / ((events['Part_fourMomentum_fCoordinates_fX'])**2 + (events['Part_fourMomentum_fCoordinates_fY'])**2 + (events['Part_fourMomentum_fCoordinates_fZ'])**2)**0.5
                    flag_not_0pdgid = (events['Part_pdgId'] != 0)
                    events['Part_isGood'] = (events['Part_isGood']==1) & (part_abscosth < 0.732) & (part_abscosth > 0.035) # & flag_not_0pdgid
                    for part_branch in part_branches:
                        if part_branch != 'Part_isGood':
                            events[part_branch] = events[part_branch][events['Part_isGood']] 

                    if not self.is_data:
                        # get truth info of tau pair and tau neutrinos
                        dict_part_pdg = {
                            'tau': 15,
                            'anti_tau': -15,
                            'nu_tau': 16,
                            'anti_nu_tau': -16,
                        }
                        for key, pdgid in dict_part_pdg.items():
                            flag = (events['GenPart_pdgId'] == pdgid)
                            events[f'truth_{key}_px'] = ak.firsts(events['GenPart_vector_fCoordinates_fX'][flag][...,::-1])
                            events[f'truth_{key}_py'] = ak.firsts(events['GenPart_vector_fCoordinates_fY'][flag][...,::-1])
                            events[f'truth_{key}_pz'] = ak.firsts(events['GenPart_vector_fCoordinates_fZ'][flag][...,::-1])
                            events[f'truth_{key}_E'] = ak.firsts(events['GenPart_vector_fCoordinates_fT'][flag][...,::-1])


                    # filter events
                    self.filter_results['initial_total_num_events'] += len(events)
                    events_pass_filter, self.filter_results = filter_event(events, self.filter_results, is_Ztautau=self.is_Ztautau)

                    # record filtered events into self.data
                    for key, evt in events_pass_filter.items():
                        self.data.setdefault(key, []).append(evt)

                except Exception as e:
                    log.error(f"Error reading file {file} or tree {self.tree_name}: {e}")
                    continue

            # Concatenate data from all files
            for key in self.data:
                self.data[key] = ak.concatenate(self.data[key], axis=0)
                self.initial_total_num_events = initial_total_num_events
                self.data[key]['initial_total_num_events'] = initial_total_num_events

        # Log filter results
        if self.filter_results['initial_total_num_events'] > 0:
            with open(self.output_dir + f"/cutflow_{self.name}.txt", "w") as f:
                f.write(f"{'Cut':<40} {'Events':<20} {'Efficiency':<20} {'Relative Efficiency':<20}\n")
                previous_count = self.filter_results['initial_total_num_events']
                for key, value in self.filter_results.items():
                    log.info(f"Filter result - {key}: {value}. Filter efficiency: {value / self.filter_results['initial_total_num_events']:.4f}")
                    efficiency = value / self.filter_results['initial_total_num_events']
                    relative_efficiency = value / previous_count if previous_count > 0 else 1.0
                    f.write(f"{key:<40} {value:<20} {efficiency:<20.4f} {relative_efficiency:<20.4f}\n")
                    previous_count = value

            # plot filter results
            cutflow_labels = list(self.filter_results.keys())
            cutflow_values = [self.filter_results[key] for key in cutflow_labels]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_values)
            ax.bar_label(p, labels=[f"{v}" for v in cutflow_values], padding=3, fontsize=4)
            ax.set_ylabel('Number of Events')
            ax.set_title('Event Cutflow')
            ax.set_yscale('log')
            # rotate x, fontsize to small
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + f"/cutflow_{self.name}.pdf")

            cutflow_normalized = [v / self.filter_results['initial_total_num_events'] for v in cutflow_values]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_normalized)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_normalized], padding=3, fontsize=4)
            ax.set_ylabel('Efficiency')
            ax.set_title('Event Cutflow Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + f"/cutflow_efficiency_{self.name}.pdf")

            # cutflow_relative = [cutflow_values[i] / cutflow_values[i-1] if i > 0 else 1.0 for i in range(len(cutflow_values))]
            cutflow_relative = [1.0]
            tmp_cutflow_label = ['initial_totoal_num_events']
            for i in range(1, len(cutflow_values)):
                rel = cutflow_values[i] / cutflow_values[i-1] if cutflow_values[i-1] > 0 else 0
                label = cutflow_labels[i]
                if rel>1:
                    # if eff>1 then calculate ratio relative to initial num
                    rel = cutflow_values[i] / cutflow_values[0]
                    label = f"{label}/initialNoE"
                cutflow_relative.append(rel)
                tmp_cutflow_label.append(label)

            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(tmp_cutflow_label, cutflow_relative)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_relative], padding=3, fontsize=4)
            ax.set_ylabel('Relative Efficiency')
            ax.set_title('Event Cutflow Relative Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + f"/cutflow_relative_efficiency_{self.name}.pdf")

        return self.data

    
    def save_data(self):
        output_file_prefix = self.output_dir + "/filtered"
        for key, evt in self.data.items():
            output_file = output_file_prefix + f"___{key}.parquet"
            log.info(f"Saving data for region {key} to {output_file}.")
            ak.to_parquet(evt, output_file, compression='snappy')

            log.info(f"Data saved to {output_file}.")


    def postprocess(self):
        # define weight for each event
        weight = 1 if self.is_data else self.norm_factor / self.initial_total_num_events * self.luminosity
        for ch, ch_events in self.data.items():
            ch_events['weight'] = weight * ak.ones_like(ch_events['evtNumber'], dtype=np.float32)

        # rebuild branches that end in p4 
        for ch, ch_events in self.data.items():
            for br in ch_events.fields:
                if br.endswith('p4'):
                    ch_events[br] = rebuild_p4(ch_events[br])

            # define missing p4
            masking_part = (ch_events['Part_charge'] != 0) | (ch_events['Part_pdgId'] == 21) # only consider charged particles and photons for visible p4, the rest is considered as missing
            ch_events['missing_p4'] = vector.zip({
                'px': -ak.sum(ch_events['Part_fourMomentum_fCoordinates_fX'] * masking_part, axis=1),
                'py': -ak.sum(ch_events['Part_fourMomentum_fCoordinates_fY'] * masking_part, axis=1),
                'pz': -ak.sum(ch_events['Part_fourMomentum_fCoordinates_fZ'] * masking_part, axis=1),
                'E': cme - ak.sum(ch_events['Part_fourMomentum_fCoordinates_fT'] * masking_part, axis=1),
                }
            )
            

            for hemisphere, hemisphere_id in [(1, 'a'), (-1, 'b')]:
                # only consider photons near the leading particles
                ch_events[f'lead_{hemisphere_id}_visible_p4'] = ch_events[f'lead_{hemisphere_id}_p4']
                photon_mask = ch_events[f'is_photon_near_lead_{hemisphere_id}'] == 1
                sum_photon_p4 = get_sum_p4_from_ak_events(ch_events, photon_mask)
                ch_events[f'lead_{hemisphere_id}_visible_p4'] = ch_events[f'lead_{hemisphere_id}_visible_p4'] + sum_photon_p4

                # the whole hemisphere
                photon_mask = (ch_events[f'Part_hemisphere'] == hemisphere) & (ch_events[f'Part_pdgId'] == 21)
                sum_photon_p4 = get_sum_p4_from_ak_events(ch_events, photon_mask)
                ch_events[f'hemisphere_{hemisphere_id}_visible_p4'] = sum_photon_p4 + ch_events[f'lead_{hemisphere_id}_p4']

            # build truth QI observables for Ztautau sample
            if self.is_Ztautau:
                pdgId = ch_events['GenPart_pdgId']
                mis_pdg_ID = np.array([-12, -14, 16])

                tau_a_p4 = get_p4_from_ak_events(ch_events, (pdgId == -15), prefix='GenPart_vector')
                mis_a_flag = np.zeros_like(pdgId, dtype=bool)
                for pdg_id in mis_pdg_ID:
                    mis_a_flag = mis_a_flag | (pdgId == -pdg_id)
                mis_a_flag = mis_a_flag & (events['GenPart_status'] == 1)
                mis_a_p4 = get_sum_p4_from_ak_events(ch_events, mis_a_flag, prefix='GenPart_vector')
                vis_a_p4 = tau_a_p4 - mis_a_p4

                tau_b_p4 = get_p4_from_ak_events(ch_events, (pdgId == 15), prefix='GenPart_vector')
                mis_b_flag = np.zeros_like(pdgId, dtype=bool)
                for pdg_id in mis_pdg_ID:
                    mis_b_flag = mis_b_flag | (pdgId == pdg_id)
                mis_b_flag = mis_b_flag & (events['GenPart_status'] == 1)
                mis_b_p4 = get_sum_p4_from_ak_events(ch_events, mis_b_flag, prefix='GenPart_vector')
                vis_b_p4 = tau_b_p4 - mis_b_p4
                truth_observables = build_observables(tau_a_p4=tau_a_p4, tau_b_p4=tau_b_p4, vis_a_p4=vis_a_p4, vis_b_p4=vis_b_p4)
                for obs_name, obs_values in truth_observables.items():
                    ch_events[f'truth_{obs_name}'] = obs_values


                # compute analyzing power for each event
                event_category = ch_events['event_category']
                # non-tau, pion, rho, ele, mu, other 
                analyzing_power = np.array([0, 1, 0.41, -0.33, -0.34, 0])
                pos_power = analyzing_power[event_category // 10]
                neg_power = analyzing_power[event_category % 10]
                ch_events['analyzing_power'] = pos_power * neg_power

                # reweight events by (1 - scale * analyzing_power * cos_AB)/(1 - analyzing_power * cos_AB) 
                scale = self.config.get("scale_correlation", 0.5)
                ch_events['weight'] = ch_events['weight'] * (1 - scale * 0.351 * ch_events['analyzing_power'] * ch_events['truth_cos_AB']) / (1 - 0.351 * ch_events['analyzing_power'] * ch_events['truth_cos_AB'])


                # plot cos_AB distribution for truth-level taus
                outdir = f"{self.output_dir}/QI_observables_scaleCorr_{scale}/"
                os.makedirs(outdir, exist_ok=True)
                fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
                ary = ak.to_numpy(truth_observables['cos_AB'])
                mean, err_of_mean = get_mean_and_err_of_mean(ary, weights=ak.to_numpy(ch_events['weight']))
                ax.hist(truth_observables['cos_AB'], bins=50, range=(-1,1), weights=ch_events['weight'], histtype='step', label=f'Truth (μ={mean:.3f}±{err_of_mean:.3f})', linewidth=2)
                ax.set_xlabel('cos(AB)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Cosine of Angle between Truth Visible Particles for {self.name}')
                ax.legend()
                fig.tight_layout()
                fig.savefig(outdir + f"/{ch}_truth_cos_AB_{self.name}.pdf")
                plt.close(fig)



    def finalize(self):
        log.info("DataLoader finalization complete.")


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, format = ">>> [%(levelname)s]: %(message)s")
    config = {
        "tree_name": "t",
        "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251029_Ztautau_singlePionDecay/simana_job_17827112_*_ttree.root",
        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/simana_job_*_ttree.root",
        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/simana_job_17841601_*_ttree.root",

        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/merged/simana_ttree.root",
    }

    loader = DataLoader(config)

    # import numpy as np
    # import pandas as pd
    # # test reading output file
    # df = pd.read_hdf("filtered_data.h5")
    # print(df)
    # ary = np.load("filtered_data_structured.npy", allow_pickle=True).item()
    # print(ary)
