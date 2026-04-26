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

def get_cut_name():
    return 'pipi_cut'

def get_dict_of_hadhad_selection_names():
    prefix = get_cut_name()
    return {
        f'{prefix}_0': 'start of pi-pi selection',
        f'{prefix}_1': 'number of photons near leading pion == 0 in each hemisphere',
        f'{prefix}_2': 'E/p for both leading particles < 0.75',
        f'{prefix}': 'end of pi-pi selection',
    }

def get_flag_passes_hadhad_region(events: ak.Array):
    cut_name_prefix = get_cut_name()
    dict_passing_cuts = {
        f'{cut_name_prefix}_0': ak.ones_like(events['evtNumber'], dtype=bool),
    }
    pass_filter = dict_passing_cuts[f'{cut_name_prefix}_0']

    # number of photons near leading pion == 0 in each hemisphere
    pass_filter = pass_filter & (events['num_photon_near_lead_a'] == 0) & (events['num_photon_near_lead_b'] == 0)
    pass_filter = ak.fill_none(pass_filter, False)
    dict_passing_cuts[f'{cut_name_prefix}_1'] = pass_filter

    # E/p for both leading particles < 0.75
    for key in ['a', 'b']:
        lead_mask = events[f'is_lead_{key}'] == 1
        lead_E = events['Part_hpcTotalShowerEnergy'][lead_mask]
        lead_p = events['Part_p4'][lead_mask].p
        pass_filter = pass_filter & ak.firsts(lead_E / lead_p < 0.75)
    pass_filter = ak.fill_none(pass_filter, False)
    dict_passing_cuts[f'{cut_name_prefix}_2'] = pass_filter
    dict_passing_cuts[f'{cut_name_prefix}'] = pass_filter

    return dict_passing_cuts
