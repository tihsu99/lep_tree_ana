import numpy as np
from BaseProcessor import BaseProcessor
import utils.plotter as plotter
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator

class TemporaryProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        super().__init__(config)
        self.config = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dl_dict):
        for dl_name, dl in dl_dict.items():
            self.process_dataloader(dl, dl_name=dl_name)

    def process_dataloader(self, dl: DataLoader.DataLoader, dl_name: str = ""):
        #######################################################
        # Pair reco pions to truth neutrinos
        #######################################################
        events = dl.data.get(dl.region_of_interest)
        cur_output_dir = f"{self.output_dir}/{dl_name}/plots/"
        os.makedirs(cur_output_dir, exist_ok=True)
        if not dl.is_data:
            reco_pdgId = events['Part_pdgId']
            reco_abs_pdgId = np.abs(reco_pdgId)
            reco_charge = events['Part_charge']
            gen_pdgId = events['GenPart_pdgId']

            flag_pi_plus = (reco_charge == 1) & (reco_abs_pdgId == 41)
            flag_pi_minus = (reco_charge == -1) & (reco_abs_pdgId == 41)
            flag_nu = (gen_pdgId == 16)
            flag_anti_nu = (gen_pdgId == -16)

            p4_pi_plus = get_p4_from_ak_events(events, flag_pi_plus)
            p4_pi_minus = get_p4_from_ak_events(events, flag_pi_minus)
            p4_nu = get_p4_from_ak_events(events, flag_nu, prefix='GenPart_vector')
            p4_anti_nu = get_p4_from_ak_events(events, flag_anti_nu, prefix='GenPart_vector')

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
            fig.savefig(cur_output_dir + '/match_recopion_to_truthnu.png')
            plt.close(fig)


    def finalize(self):
        pass