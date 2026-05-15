import awkward as ak

from Preprocess.SelectionBase import RegionSelection


class ZllSelection(RegionSelection):
    """Inclusive Z->ll baseline shared by Zee and Zmumu sub-regions.

    Selection mirrors LepLepSelection's intent but with the kinematic cuts
    that isolate the on-shell Z peak rather than a tau-pair topology.
    """

    selection_name = 'zll'
    cut_descriptions = (
        'zll: nprong = 2 (two charged tracks)',
        'zll: leading particles are opposite-charge (1-vs-1 OS)',
        'zll: mass of leading pair > 70 GeV',
        'zll: total charged energy > 60 GeV',
        'zll: P_rad > 1.0',
        'zll: 40 < lead-particle momentum < 50 GeV in both hemispheres',
    )

    def get_cuts(self, events: ak.Array):
        p_a = events['lead_a_p4'].p
        p_b = events['lead_b_p4'].p
        lead_pair_mass = (events['lead_a_p4'] + events['lead_b_p4']).mass
        return (
            events['nprong'] == 2,
            events['is_leading_OS'],
            lead_pair_mass > 70.0,
            events['charged_E'] > 60.0,
            events['P_rad'] > 1.0,
            (p_a > 40.0) & (p_a < 50.0) & (p_b > 40.0) & (p_b < 50.0),
        )


class ZeeSelection(RegionSelection):
    """Z->ee final-state selection on top of ZllSelection. Uses moderate
    electron-ID cuts (anti-muon + E/p ~ 1) rather than the strict
    `lead_a_is_electron` flag, which is too aggressive on real LEP data.
    """

    selection_name = 'zee'
    cut_descriptions = (
        'zee: leading particles are PDG-id 2 (electrons)',
        'zee: both leading particles pass anti-muon and E/p > 0.7',
    )

    def get_cuts(self, events: ak.Array):
        is_ee_pdg = (abs(events['lead_a_pdgId']) == 2) & (abs(events['lead_b_pdgId']) == 2)

        def _moderate_e(hemi):
            ep = events[f'lead_{hemi}_hpc_E'] / (events[f'lead_{hemi}_p4'].p + 1e-10)
            return (
                (events[f'lead_{hemi}_raw_muon_tag'] == 0)  # anti-muon
                & (ep > 0.7)                                # E/p ~ 1 (vetoes hadrons too)
            )

        return (
            is_ee_pdg,
            _moderate_e('a') & _moderate_e('b'),
        )


class ZmumuSelection(RegionSelection):
    """Z->mumu final-state selection on top of ZllSelection."""

    selection_name = 'zmumu'
    cut_descriptions = (
        'zmumu: leading particles are PDG-id 6 (muons)',
        'zmumu: both leading particles have muon_tag >= 2',
    )

    def get_cuts(self, events: ak.Array):
        is_mumu_pdg = (abs(events['lead_a_pdgId']) == 6) & (abs(events['lead_b_pdgId']) == 6)
        return (
            is_mumu_pdg,
            (events['lead_a_raw_muon_tag'] >= 2) & (events['lead_b_raw_muon_tag'] >= 2),
        )
