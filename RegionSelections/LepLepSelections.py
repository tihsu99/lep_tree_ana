import awkward as ak

def get_dict_of_leplep_selection_names():
    return {
        'leplep_cut_0': 'LepLep: exactly 1 charged track per hemisphere (1 vs 1)',
        'ee_cut': 'ee: PID requirements (e e)',
        'mumu_cut': 'mumu: PID requirements (mu mu)',
        'emu_cut': 'emu: PID requirements (e mu / mu e)',
    }

def get_flag_passes_leplep_region(events: ak.Array):
    """
    Applies leptonic cuts ON TOP of the baseline selection.
    Evaluates ee, mumu, and emu simultaneously.
    """
    dict_passing_cuts = {}
    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)

    # Cut 0: 1 vs 1 topology (1 track per hemisphere)
    pass_filter = events['is_leading_OS'] & events['lead_a_is_lepton'] & events['lead_b_is_lepton']
    dict_passing_cuts['leplep_cut_0'] = pass_filter

    # Cut 1: PID matching for all channels
    is_e_a = events['lead_a_is_electron']
    is_e_b = events['lead_b_is_electron']
    is_mu_a = events['lead_a_is_muon']
    is_mu_b = events['lead_b_is_muon']

    dict_passing_cuts['ee_cut'] = pass_filter & is_e_a & is_e_b
    dict_passing_cuts['mumu_cut'] = pass_filter & is_mu_a & is_mu_b
    dict_passing_cuts['emu_cut'] = pass_filter & ((is_e_a & is_mu_b) | (is_mu_a & is_e_b))

    return dict_passing_cuts