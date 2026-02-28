import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import copy
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme
from utils import plotter


class TruthMatchProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        """
        Processor to make control plots for data/MC comparison.
        """
        super().__init__(config)
        self.config = config
        if config and 'output_dir_name' in config:
            output_dir = f"{output_dir}/{config['output_dir_name']}"
        else:
            # output_dir = f"{output_dir}/TruthMatchStudy/"
            output_dir = f"{output_dir}/TruthMatchStudy/"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dl_dict):
        # TARGET_RECON_PDGID = 41  # study on pion
        TARGET_RECON_NAME = 'pion'
        dR_max = 0.05

        def flag_target_recon_part(events):
            # use Haide_pionTag as an example ID
            # return (events.Haid_pionRich >= 1)
            # return (events.Elid_tag<2) & (events.Muid_tag<1)
            return abs(events.Part_pdgId)==41
            # return abs(events.Part_pdgId)==2
            # return abs(events.Part_pdgId)==6


        # only load Ztautau samples for now
        dl_to_load = [key for key, dl in dl_dict.items() if 'Ztautau' in key]
        for key in dl_to_load:
            dl = dl_dict[key]
            events = dl.data.get(dl.region_of_interest)

            final_state_truth_particles_flag = (events['GenPart_status']==1)
            final_state_truth_particles_p4 = get_all_p4_from_ak_events(events, final_state_truth_particles_flag, prefix='GenPart_vector')
            final_state_truth_particles_pdgId = events['GenPart_pdgId'][final_state_truth_particles_flag]

            # recon_particles_flag = ak.ones_like(events['Part_fourMomentum_fCoordinates_fX'], dtype=bool)
            # recon_particles_flag = (abs(events['Part_pdgId'])==TARGET_RECON_PDGID)
            recon_particles_flag = flag_target_recon_part(events)
            recon_particles_p4 = get_all_p4_from_ak_events(events, recon_particles_flag, prefix='Part_fourMomentum')

            pairs = ak.cartesian(
                {"reco": recon_particles_p4, "truth": final_state_truth_particles_p4},
                axis=1,  # cartesian over the particle axis within each event
                nested=True,
            )

            dR = pairs['reco'].deltaR(pairs['truth'])
            best_truth_idx = ak.argmin(dR, axis=-1)
            best_dR = ak.min(dR, axis=-1)

            is_matched = best_dR < dR_max

            matched_truth_pdgId = ak.where(
                is_matched,
                final_state_truth_particles_pdgId[best_truth_idx],
                0
            )

            # plot dR distribution
            plt.figure()
            hist_data = ak.flatten(best_dR).to_numpy()
            plt.hist(hist_data, bins=100, range=(0,1), histtype='step', color='blue')
            plt.xlabel('Delta R between reco particle and best matched truth particle')
            plt.ylabel('Counts')
            plt.yscale('log')
            plt.title(f'Delta R Distribution between Reco {TARGET_RECON_NAME}s and Truth Particles')
            plt.grid()
            plt.savefig(f"{self.output_dir}/Recon_{TARGET_RECON_NAME}_DeltaR_{key}.png")
            plt.close()

            # plot matched pdgId distribution
            unique, counts = np.unique(ak.flatten(matched_truth_pdgId).to_numpy(), return_counts=True)
            plt.figure(dpi=200)
            dict_to_plot = {str(u): c for u, c in zip(unique, counts)}
            plt.bar(dict_to_plot.keys(), dict_to_plot.values(), fill=False, linewidth=1.5, width=0.6)
            plt.xlabel(f'Mached Truth Particle PDG ID for Reco {TARGET_RECON_NAME}s')
            plt.ylabel('Counts')
            plt.yscale('log')
            plt.title(f'Matched Truth Particle PDG ID Distribution for {key}')
            plt.grid()
            plt.savefig(f"{self.output_dir}/Recon_{TARGET_RECON_NAME}_MatchedPDGID_{key}.png")
            plt.close()


            #################################
            # focus on events with truth pions
            ################################

            def truth_match_recon_efficiency(events, truth_abs_pdgId, dR_max=0.1):
                flag_truth_particle = (events['GenPart_status']==1) & (abs(events['GenPart_pdgId'])==truth_abs_pdgId)
                truth_part_p4 = get_all_p4_from_ak_events(events, flag_truth_particle, prefix='GenPart_vector')
                pairs = ak.cartesian(
                    {"truth": truth_part_p4, "reco": recon_particles_p4},
                    axis=1,  # cartesian over the particle axis within each event
                    nested=True,
                )
                dR = pairs['truth'].deltaR(pairs['reco'])
                best_reco_idx = ak.argmin(dR, axis=-1)
                best_dR = ak.min(dR, axis=-1)
                # replace None dR with large value
                best_dR = ak.where(ak.is_none(best_dR, axis=-1), 9999.0, best_dR)
                is_matched = best_dR < dR_max
                return is_matched, best_dR, truth_part_p4

            # pion efficiency
            is_matched_pion, best_dR_pion, truth_part_p4_pion = truth_match_recon_efficiency(events, truth_abs_pdgId=211, dR_max=dR_max)
            print(f"Identification Rate of truth pions as {TARGET_RECON_NAME}: ", ak.sum(is_matched_pion)/len(ak.flatten(best_dR_pion)))
            # plot efficiency vs pion p
            pion_p = ak.flatten(truth_part_p4_pion.mag)
            identified = ak.flatten(is_matched_pion)
            fig, ax = plotter.plot_y_vs_x(
                x=pion_p,
                y=identified,
                bins=51,
                band_method='stderr',
                label='Pion ID Rate',
            )

            fig_pt, ax_pt = plotter.plot_y_vs_x(
                x=ak.flatten(truth_part_p4_pion.pt),
                y=identified,
                bins=51,
                band_method='stderr',
                label='Pion ID Rate',
            )


            fig_costh, ax_costh = plotter.plot_y_vs_x(
                x=ak.flatten(truth_part_p4_pion.costheta),
                y=identified,
                bins=51,
                band_method='stderr',
                label='Pion ID Rate',
            )

            # electron ID rate
            is_matched_electron, best_dR_electron, truth_part_p4_electron = truth_match_recon_efficiency(events, truth_abs_pdgId=11, dR_max=dR_max)
            print(f"Identification Rate of electrons as {TARGET_RECON_NAME}: ", ak.sum(is_matched_electron)/len(ak.flatten(best_dR_electron)))
            # plot ID rate vs electron p
            electron_p = ak.flatten(truth_part_p4_electron.mag)
            identified = ak.flatten(is_matched_electron)
            fig, ax = plotter.plot_y_vs_x(
                x=electron_p,
                y=identified,
                bins=51,
                band_method='stderr',
                label='Electron ID Rate',
                fig=fig,
                ax=ax
            )

            fig_pt, ax_pt = plotter.plot_y_vs_x(
                x=ak.flatten(truth_part_p4_electron.pt),
                y=identified,
                bins=51,
                band_method='stderr',
                label='Electron ID Rate',
                fig=fig_pt,
                ax=ax_pt
            )

            fig_costh, ax_costh = plotter.plot_y_vs_x(
                x=ak.flatten(truth_part_p4_electron.costheta),
                y=identified,
                bins=51,
                band_method='stderr',
                label='Electron ID Rate',
                fig=fig_costh,
                ax=ax_costh
            )

            # muon ID rate
            is_matched_muon, best_dR_muon, truth_part_p4_muon = truth_match_recon_efficiency(events, truth_abs_pdgId=13, dR_max=dR_max)
            print(f"Identification Rate of muons as {TARGET_RECON_NAME}: ", ak.sum(is_matched_muon)/len(ak.flatten(best_dR_muon)))
            # plot ID rate vs muon p
            muon_p = ak.flatten(truth_part_p4_muon.mag)
            identified = ak.flatten(is_matched_muon)
            fig, ax = plotter.plot_y_vs_x(
                x=muon_p,
                y=identified,
                bins=51,
                band_method='stderr',
                label='Muon ID Rate',
                fig=fig,
                ax=ax
            )

            ax.set_ylim(0,1.1)
            ax.set_xlabel('Truth Particle Momentum [GeV]')
            ax.set_ylabel(f'Rate to be identified as Reco {TARGET_RECON_NAME}')
            ax.set_title(f"Rate of truth particles being identified as Reco {TARGET_RECON_NAME} vs Momentum for {key}", fontsize='medium')
            ax.legend(fontsize='small')
            ax.grid()
            fig.savefig(f"{self.output_dir}/Recon_{TARGET_RECON_NAME}_ID_Rate_vs_P_{key}.png")
            fig.clf()


            fig_pt, ax_pt = plotter.plot_y_vs_x(
                x=ak.flatten(truth_part_p4_muon.pt),
                y=identified,
                bins=51,
                band_method='stderr',
                label='Muon ID Rate',
                fig=fig_pt,
                ax=ax_pt
            )
            ax_pt.set_ylim(0,1.1)
            ax_pt.set_xlabel('Truth Particle Transverse Momentum [GeV]')
            ax_pt.set_ylabel(f'Rate to be identified as Reco {TARGET_RECON_NAME}')
            ax_pt.set_title(f"Rate of truth particles being identified as Reco {TARGET_RECON_NAME} vs Transverse Momentum for {key}", fontsize='medium')
            ax_pt.legend(fontsize='small')
            ax_pt.grid()
            fig_pt.savefig(f"{self.output_dir}/Recon_{TARGET_RECON_NAME}_ID_Rate_vs_PT_{key}.png")
            fig_pt.clf()
            # plt.grid()
            # plt.legend(fontsize='small')
            # plt.savefig(f"{self.output_dir}/Recon_{TARGET_RECON_NAME}_ID_Rate_vs_PT_{key}.png")
            # plt.close()

            fig_costh, ax_costh = plotter.plot_y_vs_x(
                x=ak.flatten(truth_part_p4_muon.costheta),
                y=identified,
                bins=51,
                band_method='stderr',
                label='Muon ID Rate',
                fig=fig_costh,
                ax=ax_costh
            )
            ax_costh.set_ylim(0,1.1)
            ax_costh.set_xlabel('Truth Particle Cos(Theta)')
            ax_costh.set_ylabel(f'Rate to be identified as Reco {TARGET_RECON_NAME}')
            ax_costh.set_title(f"Rate of truth particles being identified as Reco {TARGET_RECON_NAME} vs Cos(Theta) for {key}", fontsize='medium')
            ax_costh.legend(fontsize='small')
            ax_costh.grid()
            fig_costh.savefig(f"{self.output_dir}/Recon_{TARGET_RECON_NAME}_ID_Rate_vs_CosTheta_{key}.png")
            fig_costh.clf()
            # plt.grid()
            # plt.legend(fontsize='small')
            # plt.savefig(f"{self.output_dir}/Recon_{TARGET_RECON_NAME}_ID_Rate_vs_CosTheta_{key}.png")
            # plt.close()


            # has_truth_pion = ak.any((final_state_truth_particles_pdgId == 211) | (final_state_truth_particles_pdgId == -211), axis=-1)
            # events_wtpion = events[has_truth_pion]
            # truth_pion_p4 = get_all_p4_from_ak_events(events_wtpion, (events_wtpion['GenPart_status']==1) & ((abs(events_wtpion['GenPart_pdgId'])==211)), prefix='GenPart_vector')
            # recon_particles_p4_wtpion = get_all_p4_from_ak_events(events_wtpion, ak.ones_like(events_wtpion['Part_fourMomentum_fCoordinates_fX'], dtype=bool), prefix='Part_fourMomentum')
            # pairs_wtpion = ak.cartesian(
            #     {"truth_pion": truth_pion_p4, "reco": recon_particles_p4_wtpion},
            #     axis=1,  # cartesian over the particle axis within each event
            #     nested=True,
            # )
            # dR_wtpion = pairs_wtpion['truth_pion'].deltaR(pairs_wtpion['reco'])
            # best_reco_idx = ak.argmin(dR_wtpion, axis=-1)
            # best_dR_wtpion = ak.min(dR_wtpion, axis=-1)
            # dR_max = 0.1
            # is_matched_wtpion = best_dR_wtpion < dR_max
            # print("Efficiency of identifying truth pions: ", ak.sum(is_matched_wtpion)/len(ak.flatten(best_dR_wtpion)))
            # # plot efficiency vs pion p
            # pion_p = ak.flatten(truth_pion_p4.mag)
            # identified = ak.flatten(is_matched_wtpion)
            # fig, ax = plotter.plot_y_vs_x(
            #     x=pion_p,
            #     y=identified,
            #     bins=51,
            #     band_method='stderr',
            #     # band=("68")
            # )
            # ax.set_ylim(0,1.1)
            # ax.set_xlabel('Truth Pion Momentum [GeV]')
            # ax.set_ylabel('Identification Efficiency')
            # ax.set_title(f'Truth Pion Identification Efficiency vs Momentum for {key}')
            # plt.grid()
            # plt.savefig(f"{self.output_dir}/TruthPionID_Efficiency_vs_P_{key}.png")
            # plt.close()












    def finalize(self):
        pass
