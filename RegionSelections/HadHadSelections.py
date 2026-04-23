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
    return 'hadhad_cut'

def get_dict_of_hadhad_selection_names():
    prefix = get_cut_name()
    return {
        f'{prefix}_0': 'start of had-had selection',
        f'{prefix}_1': 'lg(1 - thrust) within (2.5, 4.5)',
        f'{prefix}_2': 'nprong=2',
        f'{prefix}_3': 'leading particles are opposite-charge pions',
        f'{prefix}_4': 'number of photons near leading pion <= 3 in each hemisphere',
        f'{prefix}': 'end of had-had selection',
    }


def get_flag_passes_hadhad_region(events: ak.Array):
    cut_name_prefix = get_cut_name()
    dict_passing_cuts = {
        f'{cut_name_prefix}_0': ak.ones_like(events['evtNumber'], dtype=bool),
    }
    pass_filter = dict_passing_cuts[f'{cut_name_prefix}_0']

    # thrust within range
    thrust_magnitude = events['thrust_Mag']
    neglog1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
    pass_filter = (neglog1mthrust > 2.5) & (neglog1mthrust < 4.5) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_1'] = pass_filter

    # nprong = 2
    pass_filter = (events['nprong'] == 2) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_2'] = pass_filter

    # leading particles are opposite-charge pions
    pass_filter = pass_filter & (events['is_leading_OS']==True) & (events['lead_a_is_pion']==True) & (events['lead_b_is_pion']==True)
    dict_passing_cuts[f'{cut_name_prefix}_3'] = pass_filter

    # number of photons near leading pion <= 3 in each hemisphere
    pass_filter = pass_filter & (events['num_photon_near_lead_a'] <= 3) & (events['num_photon_near_lead_b'] <= 3)
    dict_passing_cuts[f'{cut_name_prefix}_4'] = pass_filter
    dict_passing_cuts[f'{cut_name_prefix}'] = pass_filter

    return dict_passing_cuts
