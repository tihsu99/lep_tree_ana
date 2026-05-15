import awkward as ak

from Preprocess.SelectionBase import RegionSelection


# Lephad selections: one charged lepton (e or mu) in one hemisphere and one
# charged hadron (pi or rho) in the other. Modeled on PiRhoSelection: an
# `is_lepton_positive` flag picks which hemisphere (a vs b) holds the lepton,
# `lepton_type` is 'el' or 'mu', and `hadron_type` is 'pi' or 'rho'. Eight
# ordered sub-regions in total: pi_el, el_pi, pi_mu, mu_pi, rho_el, el_rho,
# rho_mu, mu_rho.
LEPHAD_SUBREGIONS = [
    ('pi_el',  False, 'el', 'pi'),   # hadron in a, lepton in b
    ('pi_mu',  False, 'mu', 'pi'),
    ('rho_el', False, 'el', 'rho'),
    ('rho_mu', False, 'mu', 'rho'),
    ('el_pi',  True,  'el', 'pi'),   # lepton in a, hadron in b
    ('mu_pi',  True,  'mu', 'pi'),
    ('el_rho', True,  'el', 'rho'),
    ('mu_rho', True,  'mu', 'rho'),
]

LEPTON_PID_FIELD = {'el': 'is_electron', 'mu': 'is_muon'}


class LepHadSelection(RegionSelection):
    """One lephad sub-region. Cumulative cuts are evaluated relative to the
    baseline selection (and conceptually parallel to PiPi/PiRho on the hadronic
    side).
    """

    def __init__(self, is_lepton_positive: bool, lepton_type: str, hadron_type: str):
        self.is_lepton_positive = is_lepton_positive
        self.lepton_type = lepton_type
        self.hadron_type = hadron_type
        if is_lepton_positive:
            self.selection_name = f'{lepton_type}{hadron_type}'
        else:
            self.selection_name = f'{hadron_type}{lepton_type}'
        self.cut_descriptions = (
            f'{self.selection_name}: 1-vs-1 topology with strict {lepton_type} ID on the leptonic side',
            f'{self.selection_name}: E/p < 0.6 on the hadronic side',
            f'{self.selection_name}: photon and visible-mass cuts for the hadronic side',
        )

    def get_cuts(self, events: ak.Array):
        lep_id = 'a' if self.is_lepton_positive else 'b'
        had_id = 'b' if self.is_lepton_positive else 'a'
        lepton_flag = events[f'lead_{lep_id}_{LEPTON_PID_FIELD[self.lepton_type]}']

        # Topology + lepton PID
        cut_topology = (events['nprong'] == 2) & events['is_leading_OS'] & lepton_flag

        # E/p < 0.6 on the hadronic side, mirroring PiPi
        cut_e_over_p = events[f'lead_{had_id}_E_over_p'] < 0.6

        # Photon count and rho mass on the hadronic side
        n_phot = events[f'num_photon_near_lead_{had_id}']
        if self.hadron_type == 'pi':
            cut_hadron_shape = n_phot == 0
        else:  # rho
            rho_mass = events[f'lead_{had_id}_visible_p4'].mass
            cut_hadron_shape = (
                (n_phot >= 1) & (n_phot <= 2)
                & (rho_mass > 0.5) & (rho_mass < 1.04)
            )

        return (cut_topology, cut_e_over_p, cut_hadron_shape)
