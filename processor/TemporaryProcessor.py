import numpy as np
from BaseProcessor import BaseProcessor
import utils.plotter as plotter
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak

cme = 91.25 # GeV
def get_color_iterator(n):
    return iter(plt.cm.tab10.colors * (n // 10 + 1))

def get_p4(events, flag, prefix='Part_fourMomentum'):
    px = ak.firsts(events[f'{prefix}_fCoordinates_fX'][flag][...,::-1]).to_numpy()
    py = ak.firsts(events[f'{prefix}_fCoordinates_fY'][flag][...,::-1]).to_numpy()
    pz = ak.firsts(events[f'{prefix}_fCoordinates_fZ'][flag][...,::-1]).to_numpy()
    E =  ak.firsts(events[f'{prefix}_fCoordinates_fT'][flag][...,::-1]).to_numpy()
    p4 = vector.zip({
        "px": px,
        "py": py,
        "pz": pz,
        "E": E,
    })
    return p4

class TemporaryProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        super().__init__(config)
        self.config = config
        self.output_dir = output_dir + "/plots/"
        os.makedirs(self.output_dir, exist_ok=True)


    def run(self, dl: DataLoader.DataLoader):
        #######################################################
        # Pair reco pions to truth neutrinos
        #######################################################
        events = dl.data.get(dl.region_of_interest)

        reco_pdgId = events['Part_pdgId']
        reco_abs_pdgId = np.abs(reco_pdgId)
        reco_charge = events['Part_charge']
        gen_pdgId = events['GenPart_pdgId']

        flag_pi_plus = (reco_charge == 1) & (reco_abs_pdgId == 41)
        flag_pi_minus = (reco_charge == -1) & (reco_abs_pdgId == 41)
        flag_nu = (gen_pdgId == 16)
        flag_anti_nu = (gen_pdgId == -16)

        p4_pi_plus = get_p4(events, flag_pi_plus)
        p4_pi_minus = get_p4(events, flag_pi_minus)
        p4_nu = get_p4(events, flag_nu, prefix='GenPart_vector')
        p4_anti_nu = get_p4(events, flag_anti_nu, prefix='GenPart_vector')

        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel(r'Invariant Mass [GeV]')
        ax.set_ylabel('Events')
        ax.set_title('Reconstructed Tau from Reconstructed Pion and Generator Neutrino')
        color_iterator = get_color_iterator(3)
        bins = np.linspace(0, 3.5, 51)

        #######################################################
        # Pair by charge: pi- with nu, pi+ with anti-nu
        ########################################################
        inv_mass_tau = (p4_pi_minus + p4_nu).mass
        inv_mass_anti_tau = (p4_pi_plus + p4_anti_nu).mass
        color = next(color_iterator)
        ax.hist(inv_mass_tau, bins=bins, histtype='step', color=color, label=r'$\tau^-$: matched by charge')
        ax.hist(inv_mass_anti_tau, bins=bins, histtype='step', color=color, linestyle='dashed', label=r'$\tau^+$: matched by charge')
        
        
        ########################################################
        # Pair by min deltaR
        ########################################################
        deltaR_piplus_nu = p4_pi_plus.deltaR(p4_nu)
        deltaR_piplus_antinu = p4_pi_plus.deltaR(p4_anti_nu)

        flag_piplus_match_nu = deltaR_piplus_nu < deltaR_piplus_antinu
        flag_piplus_match_antinu = ~flag_piplus_match_nu

        inv_mass_tau_dr = ak.concatenate([
            (p4_pi_minus[flag_piplus_match_nu] + p4_anti_nu[flag_piplus_match_nu]).mass,
            (p4_pi_minus[flag_piplus_match_antinu] + p4_nu[flag_piplus_match_antinu]).mass
        ])
        inv_mass_antitau_dr = ak.concatenate([
            (p4_pi_plus[flag_piplus_match_nu] + p4_nu[flag_piplus_match_nu]).mass,
            (p4_pi_plus[flag_piplus_match_antinu] + p4_anti_nu[flag_piplus_match_antinu]).mass
        ])
        
        color = next(color_iterator)
        ax.hist(inv_mass_tau_dr, bins=bins, histtype='step', color=color, label=r'$\tau^-$: matched by $\Delta R$')
        ax.hist(inv_mass_antitau_dr, bins=bins, histtype='step', color=color, linestyle='dashed', label=r'$\tau^+$: matched by $\Delta R$')

        # vertical line at tau mass
        ax.axvline(1.776, color='k', linestyle='dotted', label='Tau Mass')

        ax.legend()
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(self.output_dir + '/match_recopion_to_truthnu.png')
        plt.close(fig)


    def finalize(self):
        pass