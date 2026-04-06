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
    return 'baseline_cut'

def get_dict_of_baseline_selection_names():
    prefix = get_cut_name()
    return {
        f'{prefix}_0': 'start of baseline selection',
        f'{prefix}_1': 'charge multiplicity in [2, 6]',
        f'{prefix}_2': 'charged leading particle in each hemisphere',
        f'{prefix}_3': 'angular and vertex cuts on leading charged particles',
        f'{prefix}_4': 'charged energy > 0.0875*Ecm',
        f'{prefix}_5': 'missing pt > 0.4 and accollinearity > 0.5 degree for 2-prong events',
        f'{prefix}_6': 'isolation angle > 160 degree',
        f'{prefix}_7': 'E_rad < 0.8 and P_rad < 1',
        f'{prefix}': 'end of baseline selection',
    }

def get_flag_passes_baseline(events: ak.Array):
    cut_name_prefix = get_cut_name()
    dict_passing_cuts = {
        f'{cut_name_prefix}_0': ak.ones_like(events['evtNumber'], dtype=bool),
    }
    pass_filter = dict_passing_cuts[f'{cut_name_prefix}_0']


    # require at least 2 and at most 6 charged particles in the event
    pass_filter = ak.fill_none((events['nprong'] >= 2) & (events['nprong'] <= 6), False) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_1'] = pass_filter


    # require both leading particles in each hemisphere to be charged
    pass_filter = ak.fill_none((events['valid_leadings'] == 1), False) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_2'] = pass_filter


    # angular and vertex cuts on leading charged particles
    cut_lead_a = (np.abs(events['lead_a_p4'].costheta) > 0.035) & (np.abs(events['lead_a_p4'].costheta) < 0.731)
    cut_lead_b = (np.abs(events['lead_b_p4'].costheta) > 0.035) & (np.abs(events['lead_b_p4'].costheta) < 0.731)
    pass_filter = (cut_lead_a | cut_lead_b) & pass_filter

    pass_filter = (np.abs(events['lead_a_z0']) < 4.5) & (np.abs(events['lead_b_z0']) < 4.5) & pass_filter
    pass_filter = ((np.abs(events['lead_a_d0']) < 0.3) | (np.abs(events['lead_b_d0']) < 0.3)) & pass_filter
    pass_filter = ak.fill_none(pass_filter, False)

    dict_passing_cuts[f'{cut_name_prefix}_3'] = pass_filter

    # sum charged E
    charged_E = events['charged_E']
    pass_filter = (charged_E > 0.0875 * cme) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_4'] = pass_filter

    # cut on missing pt and accollinearity for 2-prong events
    missing_pt = events['missing_pt']
    pass_filter = ( ((events['nprong']==2) & (missing_pt > 0.4)) | (events['nprong']!=2) ) & pass_filter

    isolation_angle = events['isolation_angle']
    pass_filter = ( ((events['nprong']==2) & (isolation_angle < 179.5) & (isolation_angle>=0)) | (events['nprong']!=2) ) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_5'] = ak.fill_none(pass_filter, False)

    # isolation angle > 160 degree for all events
    pass_filter = (events['isolation_angle'] > 160) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_6'] = pass_filter

    # Erad Prad
    pass_filter = (events['E_rad'] < 0.8) & (events['P_rad'] < 1) & pass_filter
    dict_passing_cuts[f'{cut_name_prefix}_7'] = pass_filter

    dict_passing_cuts[f'{cut_name_prefix}'] = pass_filter

    return dict_passing_cuts
