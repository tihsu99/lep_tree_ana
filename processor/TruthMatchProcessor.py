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
        self.regions = config.get('regions', ['pion'])
        self.dl_to_load = config.get('dl_to_load', ['Ztautau'])
        self.dR_threshold = config.get('dR_threshold', 0.05)

        self.target_recon_particle = config.get('target_recon_particle', 'pion')
        os.makedirs(self.output_dir, exist_ok=True)

        if self.target_recon_particle == 'pion':
            self.id_algorithms = [
                "baseline", 
                # "Haid_pionRich", "Haidn_pionTag", "Haidr_pionTag", "Haide_pionTag", "Haidc_pionTag"
            ]

    
    def particle_pass_WP(self, events, id_algo):
        if id_algo == "baseline":
            return abs(events.Part_pdgId)==41
        else:
            return (events[id_algo] >= 1)


    def run(self, dl_dict):
        for id_algo in self.id_algorithms:
            for region in self.regions:
                for dl_name in self.dl_to_load:
                    if dl_name not in dl_dict:
                        print(f"DataLoader {dl_name} not found in dl_dict. Skipping.")
                        continue
                    dl = dl_dict[dl_name]
                    if region not in dl.data:
                        print(f"Region {region} not found in DataLoader {dl_name}. Skipping.")
                        continue
                    events = dl.data.get(region)

                    output_dir = f"{self.output_dir}/{region}_{dl_name}_{id_algo}/"
                    os.makedirs(output_dir, exist_ok=True)

                    final_state_truth_particles_flag = (events['GenPart_status']==1)
                    final_state_truth_particles_p4 = get_all_p4_from_ak_events(events, final_state_truth_particles_flag, prefix='GenPart_vector')
                    final_state_truth_particles_pdgId = events['GenPart_pdgId'][final_state_truth_particles_flag]

                    # recon_particles_flag = ak.ones_like(events['Part_fourMomentum_fCoordinates_fX'], dtype=bool)
                    # recon_particles_flag = (abs(events['Part_pdgId'])==TARGET_RECON_PDGID)
                    recon_particles_flag = self.particle_pass_WP(events, id_algo)
                    recon_particles_p4 = get_all_p4_from_ak_events(events, recon_particles_flag, prefix='Part_fourMomentum')

                    pairs = ak.cartesian(
                        {"reco": recon_particles_p4, "truth": final_state_truth_particles_p4},
                        axis=1,  # cartesian over the particle axis within each event
                        nested=True,
                    )

                    dR = pairs['reco'].deltaR(pairs['truth'])
                    best_truth_idx = ak.argmin(dR, axis=-1)
                    best_dR = ak.min(dR, axis=-1)

                    is_matched = best_dR < self.dR_threshold

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
                    plt.title(f'Delta R Distribution between Reco {self.target_recon_particle}s and Truth Particles')
                    plt.grid()
                    plt.savefig(f"{output_dir}/Recon_{self.target_recon_particle}_DeltaR.png")
                    plt.close()

                    # plot matched pdgId distribution
                    unique, counts = np.unique(ak.flatten(matched_truth_pdgId).to_numpy(), return_counts=True)
                    plt.figure(dpi=200)
                    dict_to_plot = {str(u): c for u, c in zip(unique, counts)}
                    plt.bar(dict_to_plot.keys(), dict_to_plot.values(), fill=False, linewidth=1.5, width=0.6)
                    plt.xlabel(f'Mached Truth Particle PDG ID for Reco {self.target_recon_particle}s')
                    plt.ylabel('Counts')
                    plt.yscale('log')
                    plt.title(f'Matched Truth Particle PDG ID Distribution for {self.target_recon_particle}s with {id_algo} ID Algo')
                    plt.grid()
                    plt.savefig(f"{output_dir}/Recon_{self.target_recon_particle}_MatchedPDGID.png")
                    plt.close()


                    #################################
                    # focus on events with truth pions
                    ################################

                    def truth_match_recon_efficiency(events, truth_abs_pdgId, dR_threshold=0.1):
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
                        is_matched = best_dR < dR_threshold
                        return is_matched, best_dR, truth_part_p4

                    # pion efficiency
                    is_matched_pion, best_dR_pion, truth_part_p4_pion = truth_match_recon_efficiency(events, truth_abs_pdgId=211, dR_threshold=self.dR_threshold)
                    print(f"Identification Rate of truth pions as {self.target_recon_particle}: ", ak.sum(is_matched_pion)/len(ak.flatten(best_dR_pion)))
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
                    is_matched_electron, best_dR_electron, truth_part_p4_electron = truth_match_recon_efficiency(events, truth_abs_pdgId=11, dR_threshold=self.dR_threshold)
                    # plot ID rate vs electron p
                    electron_p = ak.flatten(truth_part_p4_electron.mag)
                    identified = ak.flatten(is_matched_electron)
                    if len(electron_p) > 0:
                        print(f"Identification Rate of electrons as {self.target_recon_particle}: ", ak.sum(is_matched_electron)/len(ak.flatten(best_dR_electron)))
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
                    is_matched_muon, best_dR_muon, truth_part_p4_muon = truth_match_recon_efficiency(events, truth_abs_pdgId=13, dR_threshold=self.dR_threshold)
                    # plot ID rate vs muon p
                    muon_p = ak.flatten(truth_part_p4_muon.mag)
                    identified = ak.flatten(is_matched_muon)
                    if len(muon_p) > 0:
                        print(f"Identification Rate of muons as {self.target_recon_particle}: ", ak.sum(is_matched_muon)/len(ak.flatten(best_dR_muon)))
                        fig, ax = plotter.plot_y_vs_x(
                            x=muon_p,
                            y=identified,
                            bins=51,
                            band_method='stderr',
                            label='Muon ID Rate',
                            fig=fig,
                            ax=ax
                        )


                        fig_pt, ax_pt = plotter.plot_y_vs_x(
                            x=ak.flatten(truth_part_p4_muon.pt),
                            y=identified,
                            bins=51,
                            band_method='stderr',
                            label='Muon ID Rate',
                            fig=fig_pt,
                            ax=ax_pt
                        )

                        fig_costh, ax_costh = plotter.plot_y_vs_x(
                            x=ak.flatten(truth_part_p4_muon.costheta),
                            y=identified,
                            bins=51,
                            band_method='stderr',
                            label='Muon ID Rate',
                            fig=fig_costh,
                            ax=ax_costh
                        )

                    ax.set_ylim(0,1.1)
                    ax.set_xlabel('Truth Particle Momentum [GeV]')
                    ax.set_ylabel(f'Rate to be identified as Reco {self.target_recon_particle}')
                    ax.set_title(f"Rate of truth particles being identified as Reco {self.target_recon_particle} vs Momentum for {self.target_recon_particle}s", fontsize='medium')
                    ax.legend(fontsize='small')
                    ax.grid()
                    fig.savefig(f"{output_dir}/Recon_{self.target_recon_particle}_ID_Rate_vs_P.png")
                    fig.clf()

                    ax_pt.set_ylim(0,1.1)
                    ax_pt.set_xlabel('Truth Particle Transverse Momentum [GeV]')
                    ax_pt.set_ylabel(f'Rate to be identified as Reco {self.target_recon_particle}')
                    ax_pt.set_title(f"Rate of truth particles being identified as Reco {self.target_recon_particle} vs Transverse Momentum for {self.target_recon_particle}s", fontsize='medium')
                    ax_pt.legend(fontsize='small')
                    ax_pt.grid()
                    fig_pt.savefig(f"{output_dir}/Recon_{self.target_recon_particle}_ID_Rate_vs_PT.png")
                    fig_pt.clf()

                    ax_costh.set_ylim(0,1.1)
                    ax_costh.set_xlabel('Truth Particle Cos(Theta)')
                    ax_costh.set_ylabel(f'Rate to be identified as Reco {self.target_recon_particle}')
                    ax_costh.set_title(f"Rate of truth particles being identified as Reco {self.target_recon_particle} vs Cos(Theta) for {self.target_recon_particle}s", fontsize='medium')
                    ax_costh.legend(fontsize='small')
                    ax_costh.grid()
                    fig_costh.savefig(f"{output_dir}/Recon_{self.target_recon_particle}_ID_Rate_vs_CosTheta.png")
                    fig_costh.clf()











    def finalize(self):
        pass
