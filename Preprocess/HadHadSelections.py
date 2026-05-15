import awkward as ak
import numpy as np

from Preprocess.SelectionBase import RegionSelection


class HadHadSelection(RegionSelection):
    selection_name = 'hadhad'
    cut_descriptions = (
        'lg(1 - thrust) within (2.5, 4.5)',
        'nprong=2',
        'leading particles are opposite-charge pions',
        'number of photons near leading pion <= 3 in each hemisphere',
    )

    def get_cuts(self, events: ak.Array):
        neglog1mthrust = -np.log10(1 - events['thrust_Mag'] + 1e-10)
        return (
            (neglog1mthrust > 2.5) & (neglog1mthrust < 4.5),
            events['nprong'] == 2,
            events['is_leading_OS'] & events['lead_a_is_pion'] & events['lead_b_is_pion'],
            (events['num_photon_near_lead_a'] <= 3) & (events['num_photon_near_lead_b'] <= 3),
        )


class PiPiSelection(RegionSelection):
    selection_name = 'pipi'
    cut_descriptions = (
        'pi-pi: E/p for both leading particles < 0.6',
        'pi-pi: number of photons near leading pion == 0 in each hemisphere',
    )

    def get_cuts(self, events: ak.Array):
        return (
            (events['lead_a_E_over_p'] < 0.6) & (events['lead_b_E_over_p'] < 0.6),
            (events['num_photon_near_lead_a'] == 0) & (events['num_photon_near_lead_b'] == 0),
        )


class PiRhoSelection(RegionSelection):
    def __init__(self, is_pion_positive: bool):
        self.is_pion_positive = is_pion_positive
        self.selection_name = 'pirho' if is_pion_positive else 'rhopi'
        self.cut_descriptions = (
            f'{self.selection_name}: E/p for both leading particles < 0.6',
            f'{self.selection_name}: photon and mass selections',
        )

    def get_cuts(self, events: ak.Array):
        pion_id, rho_id = ('a', 'b') if self.is_pion_positive else ('b', 'a')
        rho_num_photons = events[f'num_photon_near_lead_{rho_id}']
        pion_num_photons = events[f'num_photon_near_lead_{pion_id}']
        rho_mass = events[f'lead_{rho_id}_visible_p4'].mass

        return (
            (events[f'lead_{pion_id}_E_over_p'] < 0.6) & (events[f'lead_{rho_id}_E_over_p'] < 0.6),
            (pion_num_photons == 0)
            & (rho_num_photons >= 1)
            & (rho_num_photons <= 2)
            & (rho_mass > 0.5)
            & (rho_mass < 1.04),
        )
