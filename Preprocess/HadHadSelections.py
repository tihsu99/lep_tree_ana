import awkward as ak
import numpy as np

from Preprocess.SelectionBase import RegionSelection


class HadHadSelection(RegionSelection):
    selection_name = 'hadhad'
    cut_descriptions = (
        'lg(1 - thrust) within (2.5, 4.5)',
        'nprong=2',
        'leading particles are opposite-charge',
    )

    def get_cuts(self, events: ak.Array):
        neglog1mthrust = -np.log10(1 - events['thrust_Mag'] + 1e-10)
        return (
            (neglog1mthrust > 2.5) & (neglog1mthrust < 4.5),
            events['nprong'] == 2,
            events['is_leading_OS'],
        )


class PiPiSelection(RegionSelection):
    selection_name = 'pipi'
    cut_descriptions = (
        'pipi: both leading particles are pions',
    )

    def get_cuts(self, events: ak.Array):
        return (
            (events['lead_a_is_pion'] & events['lead_b_is_pion']),
        )

class RhoRhoSelection(RegionSelection):
    selection_name = 'rhorho'
    cut_descriptions = (
        'rhorho: both leading particles are rhos',
    )

    def get_cuts(self, events: ak.Array):
        return (
            (events['lead_a_is_rho'] & events['lead_b_is_rho']),
        )


class PiRhoSelection(RegionSelection):
    def __init__(self, is_pion_positive: bool):
        self.is_pion_positive = is_pion_positive
        self.selection_name = 'pirho' if is_pion_positive else 'rhopi'
        self.cut_descriptions = (
            f'{self.selection_name}: leading particles are {self.selection_name}',
        )

    def get_cuts(self, events: ak.Array):
        pion_id, rho_id = ('a', 'b') if self.is_pion_positive else ('b', 'a')

        return (
            (events[f'lead_{pion_id}_is_pion'] & events[f'lead_{rho_id}_is_rho']),
        )
