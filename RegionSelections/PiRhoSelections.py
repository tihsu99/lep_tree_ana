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

def get_cut_name(is_pion_positive):
    return 'pirho_cut' if is_pion_positive else 'rhopi_cut'

def get_dict_of_pirho_selection_names(is_pion_positive):
    prefix = get_cut_name(is_pion_positive)
    region = 'pi-rho' if is_pion_positive else 'rho-pi'
    return {
        f'{prefix}_0': f'start of {region} selection',
        f'{prefix}_1': 'E/p for both leading particles < 0.6',
        f'{prefix}_2': 'photon and mass selections',
        f'{prefix}': f'end of {region} selection',
    }

def get_flag_passes_pirho_region(events: ak.Array, is_pion_positive: bool):
    cut_name_prefix = get_cut_name(is_pion_positive)
    dict_passing_cuts = {
        f'{cut_name_prefix}_0': ak.ones_like(events['evtNumber'], dtype=bool),
    }
    pass_filter = dict_passing_cuts[f'{cut_name_prefix}_0']

    # E/p for both leading particles < 0.6
    for key in ['a', 'b']:
        lead_mask = events[f'is_lead_{key}'] == 1
        lead_E = events['Part_hpcTotalShowerEnergy'][lead_mask]
        lead_p = events['Part_p4'][lead_mask].p
        pass_filter = pass_filter & ak.firsts(lead_E / lead_p < 0.6)
    pass_filter = ak.fill_none(pass_filter, False)
    dict_passing_cuts[f'{cut_name_prefix}_1'] = pass_filter

    # rho candidate selection
    # rho_id = 'b' if is_pion_positive else 'a'
    if is_pion_positive:
        pion_id = 'a'
        rho_id = 'b'
    else:
        pion_id = 'b'
        rho_id = 'a'

    rho_num_photons = events[f'num_photon_near_lead_{rho_id}']
    pion_num_photons = events[f'num_photon_near_lead_{pion_id}']
    rho_mass = events[f'lead_{rho_id}_visible_p4'].mass
    pass_filter = pass_filter & (pion_num_photons==0) & (rho_num_photons >= 1) & (rho_num_photons <= 2) & (rho_mass > 0.5) & (rho_mass < 1.04)
    pass_filter = ak.fill_none(pass_filter, False)
    dict_passing_cuts[f'{cut_name_prefix}_2'] = pass_filter

    dict_passing_cuts[f'{cut_name_prefix}'] = pass_filter
    return dict_passing_cuts
