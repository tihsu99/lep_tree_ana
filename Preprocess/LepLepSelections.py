import awkward as ak

from Preprocess.SelectionBase import RegionSelection


class LepLepSelection(RegionSelection):
    selection_name = 'lep-lep'
    cut_descriptions = (
        'leading particles are opposite-charge leptons',
    )

    def get_cuts(self, events: ak.Array):
        return (
            events['is_leading_OS'] & events['lead_a_is_lepton'] & events['lead_b_is_lepton'],
        )


class EESelection(RegionSelection):
    selection_name = 'ee'
    cut_descriptions = (
        'leading particles are opposite-charge electrons',
    )

    def get_cuts(self, events: ak.Array):
        return (
            events['lead_a_is_electron'] & events['lead_b_is_electron'],
        )


class MuMuSelection(RegionSelection):
    selection_name = 'mumu'
    cut_descriptions = (
        'leading particles are opposite-charge muons',
    )

    def get_cuts(self, events: ak.Array):
        return (
            events['lead_a_is_muon'] & events['lead_b_is_muon'],
        )


class MuESelection(RegionSelection):
    selection_name = 'emu'
    cut_descriptions = (
        'leading particles are opposite-charge electron-muon pair',
    )

    def get_cuts(self, events: ak.Array):
        return (
            (events['lead_a_is_electron'] & events['lead_b_is_muon'])
            | (events['lead_a_is_muon'] & events['lead_b_is_electron']),
        )


MuMuSelction = MuMuSelection
