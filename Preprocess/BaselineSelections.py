import awkward as ak
import numpy as np

from Preprocess.SelectionBase import RegionSelection
from utils.common_functions import cme


class BaselineSelection(RegionSelection):
    selection_name = 'baseline'
    cut_descriptions = (
        'charge multiplicity in [2, 6]',
        'charged leading particle in each hemisphere',
        'angular and vertex cuts on leading charged particles',
        'charged energy > 0.0875*Ecm',
        'missing pt > 0.4 and accollinearity > 0.5 degree for 2-prong events',
        'isolation angle > 160 degree',
        'E_rad < 0.8 and P_rad < 1',
    )

    def get_cuts(self, events: ak.Array):
        cut_lead_a = (np.abs(events['lead_a_p4'].costheta) > 0.035) & (np.abs(events['lead_a_p4'].costheta) < 0.731)
        cut_lead_b = (np.abs(events['lead_b_p4'].costheta) > 0.035) & (np.abs(events['lead_b_p4'].costheta) < 0.731)
        leading_particle_quality = (
            (cut_lead_a | cut_lead_b)
            & (np.abs(events['lead_a_z0']) < 4.5)
            & (np.abs(events['lead_b_z0']) < 4.5)
            & ((np.abs(events['lead_a_d0']) < 0.3) | (np.abs(events['lead_b_d0']) < 0.3))
        )

        two_prong_topology = (
            (((events['nprong'] == 2) & (events['missing_pt'] > 0.4)) | (events['nprong'] != 2))
            & (
                ((events['nprong'] == 2) & (events['isolation_angle'] < 179.5) & (events['isolation_angle'] >= 0))
                | (events['nprong'] != 2)
            )
        )

        return (
            (events['nprong'] >= 2) & (events['nprong'] <= 6),
            events['valid_leadings'] == 1,
            leading_particle_quality,
            events['charged_E'] > 0.0875 * cme,
            two_prong_topology,
            events['isolation_angle'] > 160,
            (events['E_rad'] < 0.8) & (events['P_rad'] < 1),
        )
