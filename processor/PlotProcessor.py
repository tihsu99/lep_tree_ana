import numpy as np
from BaseProcessor import BaseProcessor
import utils.plotter as plotter
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator

class PlotProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        super().__init__(config)
        self.config = config
        self.output_dir = output_dir
        self.region_of_interest = config.get("region_of_interest", "baseline")
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dl_dict):
        for dl_name, dl in dl_dict.items():
            self.process_dataloader(dl, dl_name=dl_name)

    def process_dataloader(self, dl: DataLoader.DataLoader, dl_name: str = ""):
        events_dict = dl.data
        RegionNameOfInterest = self.region_of_interest
        events_sr = events_dict.get(RegionNameOfInterest)
        cur_output_dir = f"{self.output_dir}/{dl_name}/plots/"
        os.makedirs(cur_output_dir, exist_ok=True)
        

        #######################################################
        # plot number of reco pions in each event
        #######################################################
        fig, ax = plt.subplots(dpi=300)
        bins = np.arange(0, 18, 1)
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            flag_reco_pi = (abs(events['Part_pdgId']) == 41)
            num_reco_pions = ak.sum(flag_reco_pi, axis=1)
            ax.hist(ak.to_numpy(num_reco_pions), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        ax.set_xlabel('Number of Reconstructed Pions')
        ax.set_ylabel('Entries')
        ax.set_title('Number of Reconstructed Pions per Event')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'num_reco_pions.png'))


        #######################################################
        # plot pion p, costh, energy
        #######################################################
        fig, ax = plt.subplots(3, 1, dpi=300, figsize=(6, 12))
        bins_p = np.linspace(0, 50, 51)
        bins_costh = np.linspace(-1, 1, 51)
        bins_E = np.linspace(0, 100, 51)
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            flag_reco_pi = (abs(events['Part_pdgId']) == 41)

            p_reco_pions = (events['Part_fourMomentum_fCoordinates_fX'][flag_reco_pi]**2 + events['Part_fourMomentum_fCoordinates_fY'][flag_reco_pi]**2 + events['Part_fourMomentum_fCoordinates_fZ'][flag_reco_pi]**2)**0.5
            E_reco_pions = events['Part_fourMomentum_fCoordinates_fT'][flag_reco_pi]
            costh_reco_pions = events['Part_fourMomentum_fCoordinates_fZ'][flag_reco_pi] / p_reco_pions

            ax[0].hist(ak.to_numpy(ak.flatten(p_reco_pions)), bins=bins_p, histtype='step', density=False, label=label, color=color, linewidth=1.5)
            ax[1].hist(ak.to_numpy(ak.flatten(costh_reco_pions)), bins=bins_costh, histtype='step', density=False, label=label, color=color, linewidth=1.5)
            ax[2].hist(ak.to_numpy(ak.flatten(E_reco_pions)), bins=bins_E, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        ax[0].set_xlabel('Reconstructed Pion $p$ [GeV]')
        ax[0].set_ylabel('Entries')
        ax[0].set_title('Reconstructed Pion $p$ Distribution')
        ax[0].legend()
        ax[1].set_xlabel('Reconstructed Pion $cos\theta$')
        ax[1].set_ylabel('Entries')
        ax[1].set_title('Reconstructed Pion $cos\theta$ Distribution')
        ax[1].legend()
        ax[2].set_xlabel('Reconstructed Pion Energy [GeV]')
        ax[2].set_ylabel('Entries')
        ax[2].set_title('Reconstructed Pion Energy Distribution')
        ax[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'reco_pion_properties.png'))



        #######################################################
        # plot p of reco and truth pions
        #######################################################
        if not dl.is_data:
            fig, (ax, ax_ratio) = plt.subplots(2, 1, dpi=300, figsize=(8, 8), gridspec_kw={'height_ratios': [4, 1]})
            bins = np.linspace(0, 50, 51)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            color_iter = get_color_iterator(len(events_dict))
            for label, events in events_dict.items():
                color = next(color_iter)
                flag_reco_pi = (abs(events['Part_pdgId']) == 41)

                p_reco_pions = (events['Part_fourMomentum_fCoordinates_fX'][flag_reco_pi]**2 + events['Part_fourMomentum_fCoordinates_fY'][flag_reco_pi]**2 + events['Part_fourMomentum_fCoordinates_fZ'][flag_reco_pi]**2)**0.5

                flag_gen_pi = (abs(events['GenPart_pdgId']) == 211)
                p_gen_pions = (events['GenPart_vector_fCoordinates_fX'][flag_gen_pi]**2 + events['GenPart_vector_fCoordinates_fY'][flag_gen_pi]**2 + events['GenPart_vector_fCoordinates_fZ'][flag_gen_pi]**2)**0.5

                # ratio plot
                hist_reco, bin_edges = np.histogram(ak.to_numpy(ak.flatten(p_reco_pions)), bins=bins)
                hist_gen, _ = np.histogram(ak.to_numpy(ak.flatten(p_gen_pions)), bins=bins)
                plotter.do_ratio_plot(
                    bin_centers,
                    hist_reco,
                    hist_gen,
                    ax=ax,
                    ax_ratio=ax_ratio,
                    color1=color,
                    color2=color,
                    linestyle1='solid',
                    linestyle2='dashed',
                    label1=f'Reconstructed Pions - {label}',
                    label2=f'Generated Pions - {label}',
                    xlabel='Pion $p$ [GeV]',
                    ylabel='Entries',
                    title='Pion $p$ Distribution',
                    ratio_color=color,
                    ratio_ylabel='Reco / Gen',
                )

            ax.set_yscale('log')
            fig.tight_layout()
            fig.savefig(os.path.join(cur_output_dir, 'p_distribution.png'))

        # bar plot pdgId of reco particles
        pdgid_parser = {
            -41:r'$\pi^{-}$',
            41:r'$\pi^{+}$',
            -2: r'$e^{+}$',
            2: r'$e^{-}$',
            6: r'$\mu^{-}$',
            21: r'$\gamma$',
            42: r'$K^{+}$',
            47: r'$\pi_{0}$',
            61: r'$K_{S}^{0}$',
            62: r'$K_{L}^{0}$',
            65: r'p',
            66: r'n',
            81: r'$\Lambda$',
        }
        fig, ax = plt.subplots(dpi=300)
        # unique_pdgIds = np.unique(ak.flatten(events_dict['raw']['Part_pdgId']))
        unique_pdgIds = pdgid_parser.keys()
        x_axis = [pdgid_parser.get(pdgId, str(pdgId)) for pdgId in unique_pdgIds]
        # p distribution for each pdgId
        fig_p, ax_p = plt.subplots(dpi=300)
        bins_p = np.linspace(0, 100, 51)

        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            pdgId_coutnts = {}
            for pdgId in unique_pdgIds:
                flag = (events['Part_pdgId'] == pdgId)
                count = ak.sum(ak.sum(flag, axis=1))
                pdgId_coutnts[pdgid_parser.get(pdgId, str(pdgId))] = count
                if label==RegionNameOfInterest:
                    # p distribution for each pdgId
                    p_values = (events['Part_fourMomentum_fCoordinates_fX'][flag]**2 + events['Part_fourMomentum_fCoordinates_fY'][flag]**2 + events['Part_fourMomentum_fCoordinates_fZ'][flag]**2)**0.5
                    ax_p.hist(ak.to_numpy(ak.flatten(p_values)), bins=bins_p, histtype='step', density=False, label=f'{pdgid_parser.get(pdgId, str(pdgId))}', linewidth=1.5)

            ax.bar(x_axis, [pdgId_coutnts[x] for x in x_axis], label=label, alpha=0.7,  linewidth=1.5, fill=False, edgecolor=next(color_iter))

        ax.set_xlabel('Particle Type (PDG ID)')
        ax.set_ylabel('Entries')
        ax.set_yscale('log')
        ax.set_title('Reconstructed Particle PDG ID Distribution')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'pdgId_distribution.png'))

        ax_p.set_xlabel('Reconstructed Particle $p$ [GeV]')
        ax_p.set_ylabel('Entries')
        ax_p.set_yscale('log')
        ax_p.set_title('Reconstructed Particle $p$ Distribution in SR')
        ax_p.legend()
        fig_p.tight_layout()
        fig_p.savefig(os.path.join(cur_output_dir, 'pdgId_p_distribution_sr.png'))

        #######################################################
        # plot sum pT
        #######################################################
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(0, 100, 51)
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            sum_px = ak.sum(events['Part_fourMomentum_fCoordinates_fX'], axis=1)
            sum_py = ak.sum(events['Part_fourMomentum_fCoordinates_fY'], axis=1)
            sum_pT = (sum_px**2 + sum_py**2)**0.5
            ax.hist(ak.to_numpy(sum_pT), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        ax.set_xlabel('Sum Reconstructed $p_{T}$ [GeV]')
        ax.set_ylabel('Entries')
        ax.set_title('Sum Reconstructed Transverse Momentum per Event')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'sum_reco_pt.png'))

        #######################################################
        # plot total E
        #######################################################
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(0, 200, 51)
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            total_reco_E = ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=1)
            ax.hist(ak.to_numpy(total_reco_E), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        ax.set_xlabel('Total Reconstructed Particle Energy [GeV]')
        ax.set_ylabel('Entries')
        ax.set_title('Total Reconstructed Particle Energy per Event')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'total_reco_energy.png'))
        # fig, ax = plt.subplots(2, 2, dpi=300, figsize=(12, 10))
        # bins = np.linspace(0, 200, 51)
        # color_iter = get_color_iterator(len(events_dict))
        # for label, events in events_dict.items():
        #     color = next(color_iter)
        #     total_reco_E = ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=1)
        #     ax[0,0].hist(ak.to_numpy(total_reco_E), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        #     event_totalChargedEnergy = events['Event_totalChargedEnergy']
        #     ax[0,1].hist(ak.to_numpy(event_totalChargedEnergy), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        #     event_totalEMEnergy = events['Event_totalEMEnergy']
        #     ax[1,0].hist(ak.to_numpy(event_totalEMEnergy), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        #     event_totalHadronicEnergy = events['Event_totalHadronicEnergy']
        #     ax[1,1].hist(ak.to_numpy(event_totalHadronicEnergy), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)

        # ax[0,0].set_xlabel('Total Reconstructed Particle Energy [GeV]')
        # ax[0,0].set_ylabel('Entries')
        # ax[0,0].set_title('Total Reconstructed Particle Energy per Event')
        # ax[0,0].legend()
        # ax[1,0].set_xlabel('Total EM Energy [GeV]')
        # ax[1,0].set_ylabel('Entries')
        # ax[1,0].set_title('Total EM Energy per Event')
        # ax[1,0].legend()
        # ax[0,1].set_xlabel('Total Charged Energy [GeV]')
        # ax[0,1].set_ylabel('Entries')
        # ax[0,1].set_title('Total Charged Energy per Event')
        # ax[0,1].legend()
        # ax[1,1].set_xlabel('Total Hadronic Energy [GeV]')
        # ax[1,1].set_ylabel('Entries')
        # ax[1,1].set_title('Total Hadronic Energy per Event')
        # ax[1,1].legend()
        # fig.tight_layout()
        # fig.savefig(os.path.join(cur_output_dir, 'total_energy_components.png'))

        #######################################################
        # plot missing pT, pz and energy
        #######################################################
        fig, ax = plt.subplots(3, 1, dpi=300, figsize=(6, 12))
        bins = np.linspace(0, 100, 51)
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            sum_px = ak.sum(events['Part_fourMomentum_fCoordinates_fX'], axis=1)
            sum_py = ak.sum(events['Part_fourMomentum_fCoordinates_fY'], axis=1)
            sum_pz = ak.sum(events['Part_fourMomentum_fCoordinates_fZ'], axis=1)
            sum_E = ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=1)

            missing_pT = (sum_px**2 + sum_py**2)**0.5
            missing_pz = abs(sum_pz)
            # missing_E = np.maximum(cme - sum_E, 0)
            # missing_E = cme - sum_E
            missing_E = (missing_pT**2 + missing_pz**2)**0.5


            ax[0].hist(ak.to_numpy(missing_pT), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
            ax[1].hist(ak.to_numpy(missing_pz), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
            ax[2].hist(ak.to_numpy(missing_E), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)

            if not dl.is_data:
                # truth neutrino p4
                flag_gen_nu = (abs(events['GenPart_pdgId'])) == 16
                truth_nu_px = ak.sum(events['GenPart_vector_fCoordinates_fX'][flag_gen_nu], axis=1)
                truth_nu_py = ak.sum(events['GenPart_vector_fCoordinates_fY'][flag_gen_nu], axis=1)
                truth_nu_pz = ak.sum(events['GenPart_vector_fCoordinates_fZ'][flag_gen_nu], axis=1)
                truth_nu_E = ak.sum(events['GenPart_vector_fCoordinates_fT'][flag_gen_nu], axis=1)
                truth_nu_pT = (truth_nu_px**2 + truth_nu_py**2)**0.5
                truth_nu_pz = abs(truth_nu_pz)
                ax[0].hist(ak.to_numpy(truth_nu_pT), bins=bins, histtype='step', density=False, label=f'{label} - truth neutrinos', color=color, linewidth=1.5, linestyle='dashed', alpha=0.7)
                ax[1].hist(ak.to_numpy(truth_nu_pz), bins=bins, histtype='step', density=False, label=f'{label} - truth neutrinos', color=color, linewidth=1.5, linestyle='dashed', alpha=0.7)
                ax[2].hist(ak.to_numpy(truth_nu_E), bins=bins, histtype='step', density=False, label=f'{label} - truth neutrinos', color=color, linewidth=1.5, linestyle='dashed', alpha=0.7)

        ax[0].set_xlabel('Missing Transverse Momentum [GeV]')
        ax[0].set_ylabel('Entries')
        ax[0].legend()
        ax[1].set_xlabel('Missing Longitudinal Momentum [GeV]')
        ax[1].set_ylabel('Entries')
        ax[1].legend()
        ax[2].set_xlabel('Missing Energy [GeV]')
        ax[2].set_ylabel('Entries')
        ax[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'missing_momentum_energy.png'))

        #######################################################
        # plot m_{pi+ pi-}
        #######################################################
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(0, 100, 51)
        color_iter = get_color_iterator(len(events_dict))
        color = next(color_iter)
        label = RegionNameOfInterest
        events = events_dict.get(RegionNameOfInterest)
        # pion p4
        flag_pi_plus = (events['Part_charge'] == 1) & (abs(events['Part_pdgId']) == 41)
        flag_pi_minus = (events['Part_charge'] == -1) & (abs(events['Part_pdgId']) == 41)
        reco_pi_plus_p4 = get_p4_from_ak_events(events, flag_pi_plus, prefix='Part_fourMomentum')
        reco_pi_minus_p4 = get_p4_from_ak_events(events, flag_pi_minus, prefix='Part_fourMomentum')
        di_pion_p4 = reco_pi_plus_p4 + reco_pi_minus_p4
        ax.hist(ak.to_numpy(di_pion_p4.mass), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
        ax.set_xlabel('Invariant Mass [GeV]')
        ax.set_ylabel('Entries')
        ax.set_title('Di-Pion Invariant Mass Distribution')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'di_pion_invariant_mass_distribution.png'))
        


        # #######################################################
        # # plot m_{pi+ pi- + missing}
        # #######################################################
        # fig, ax = plt.subplots(dpi=300)
        # bins = np.linspace(0, 100, 51)
        # events = events_dict.get(RegionNameOfInterest)
        # label = RegionNameOfInterest
        # sum_px = ak.sum(events['Part_fourMomentum_fCoordinates_fX'], axis=1)
        # sum_py = ak.sum(events['Part_fourMomentum_fCoordinates_fY'], axis=1)
        # sum_pz = ak.sum(events['Part_fourMomentum_fCoordinates_fZ'], axis=1)
        # # sum_E = ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=1)

        # missing_px = -sum_px
        # missing_py = -sum_py
        # missing_pz = -sum_pz
        # missing_E = (missing_px**2 + missing_py**2 + missing_pz**2)**0.5
        # # missing_E = cme - sum_E

        # missing_p4 = vector.zip({
        #     "px": missing_px,
        #     "py": missing_py,
        #     "pz": missing_pz,
        #     "E": missing_E,
        # })

        # # pion p4
        # flag_pi_plus = (events['Part_charge'] == 1) & (abs(events['Part_pdgId']) == 41)
        # flag_pi_minus = (events['Part_charge'] == -1) & (abs(events['Part_pdgId']) == 41)
        # flag_radiation = (events['Part_pdgId'] == 21)
        # reco_pi_plus_p4 = get_p4_from_ak_events(events, flag_pi_plus, prefix='Part_fourMomentum')
        # reco_pi_minus_p4 = get_p4_from_ak_events(events, flag_pi_minus, prefix='Part_fourMomentum')
        # reco_radiation_p4 = vector.zip({
        #     "px": ak.sum(events['Part_fourMomentum_fCoordinates_fX'][flag_radiation], axis=1),
        #     "py": ak.sum(events['Part_fourMomentum_fCoordinates_fY'][flag_radiation], axis=1),
        #     "pz": ak.sum(events['Part_fourMomentum_fCoordinates_fZ'][flag_radiation], axis=1),
        #     "E": ak.sum(events['Part_fourMomentum_fCoordinates_fT'][flag_radiation], axis=1),
        # })
        # di_pion_p4 = reco_pi_plus_p4 + reco_pi_minus_p4
        # di_pion_plus_radiation_p4 = di_pion_p4 + reco_radiation_p4
        # total_p4 = di_pion_plus_radiation_p4 + missing_p4
        # total_mass = total_p4.mass

        # # truth neutrino p4
        # flag_gen_nu = (abs(events['GenPart_pdgId'])) == 16
        # truth_nu_p4 = vector.zip({
        #     "px": ak.sum(events['GenPart_vector_fCoordinates_fX'][flag_gen_nu], axis=1),
        #     "py": ak.sum(events['GenPart_vector_fCoordinates_fY'][flag_gen_nu], axis=1),
        #     "pz": ak.sum(events['GenPart_vector_fCoordinates_fZ'][flag_gen_nu], axis=1),
        #     "E": ak.sum(events['GenPart_vector_fCoordinates_fT'][flag_gen_nu], axis=1),
        # })
        # di_pion_plus_truth_nu_p4 = di_pion_p4 + truth_nu_p4
        # di_pion_plus_radiation_plus_truth_nu_p4 = di_pion_plus_radiation_p4 + truth_nu_p4

        # color_iter = get_color_iterator(5)
        # ax.hist(ak.to_numpy(total_mass), bins=bins, histtype='step', density=False, label=f"{label} - pi+ pi- + radiation + missing", color=next(color_iter), linewidth=1.5, alpha=0.7)
        # ax.hist(ak.to_numpy(di_pion_plus_radiation_p4.mass), bins=bins, histtype='step', density=False, label=f'{label} - di-pion + radiation', color=next(color_iter), linewidth=1.5, alpha=0.7)
        # ax.hist(ak.to_numpy(di_pion_p4.mass), bins=bins, histtype='step', density=False, label=f'{label} - di-pion', color=next(color_iter), linewidth=1.5, alpha=0.7)
        # ax.hist(ak.to_numpy(di_pion_plus_truth_nu_p4.mass), bins=bins, histtype='step', density=False, label=f'{label} - di-pion + truth neutrinos', color=next(color_iter), linewidth=1.5, alpha=0.7)
        # ax.hist(ak.to_numpy(di_pion_plus_radiation_plus_truth_nu_p4.mass), bins=bins, histtype='step', density=False, label=f'{label} - di-pion + radiation + truth neutrinos', color=next(color_iter), linewidth=1.5, alpha=0.7)
        # ax.set_xlabel('Invariant Mass [GeV]')
        # ax.set_ylabel('Entries')
        # # ax.set_yscale('log')
        # ax.set_title('Invariant Mass Distribution')
        # ax.legend(loc='upper left', fontsize='small')
        # fig.tight_layout()
        # fig.savefig(os.path.join(cur_output_dir, 'invariant_mass_distribution.png'))

        #######################################################
        # plot number of particles
        #######################################################
        fig, ax = plt.subplots(dpi=300)
        bins = np.arange(0, 11, 1)
        color_iter = get_color_iterator(len(events_dict))
        num_particles = ak.num(events_sr['Part_pdgId'], axis=1)
        ax.hist(ak.to_numpy(num_particles), bins=bins, histtype='step', density=False, label=RegionNameOfInterest, color=next(color_iter), linewidth=1.5)
        ax.set_xlabel('Number of Reconstructed Particles')
        ax.set_ylabel('Entries')
        ax.set_title('Number of Reconstructed Particles per Event in SR')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'num_reco_particles_sr.png'))
        
            


        #######################################################
        # plot thrust magnitude and theta
        #######################################################
        fig, ax = plt.subplots(2, 2, dpi=300, figsize=(12, 10))
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            thrust_magnitude = events['thrust_Mag'] # avoid division by zero
            thrust_x = events['thrust_x']
            thrust_y = events['thrust_y']
            thrust_z = events['thrust_z']
            thrust_cosine = (thrust_z / thrust_magnitude)
            thrust_theta = np.arccos(thrust_cosine)  # in radians
            # plot thrust magnitude
            ax[0,0].hist(ak.to_numpy(thrust_magnitude), bins=np.linspace(0, 1, 51), histtype='step', density=False, label=label, color=color, linewidth=1.5)
            # log(1-Thrust)
            ax[0,1].hist(ak.to_numpy(-np.log10(1 - thrust_magnitude + 1e-12)), bins=np.linspace(0, 12, 51), histtype='step', density=False, label=label, color=color, linewidth=1.5)
            # plot thrust theta
            ax[1,0].hist(ak.to_numpy(thrust_theta * 180/np.pi), bins=np.linspace(0, 180, 51), histtype='step', density=False, label=label, color=color, linewidth=1.5)
            # plot thrust cosine
            ax[1,1].hist(ak.to_numpy(thrust_cosine), bins=np.linspace(-1, 1, 51), histtype='step', density=False, label=label, color=color, linewidth=1.5)

        ax[0,0].set_xlabel('Thrust Magnitude')
        ax[0,0].set_ylabel('Entries')
        ax[0,0].set_title('Thrust Magnitude Distribution')
        ax[0,0].legend()
        ax[0,1].set_xlabel('-log10(1 - Thrust)')
        ax[0,1].set_ylabel('Entries')
        ax[0,1].set_title('-log10(1 - Thrust) Distribution')
        ax[0,1].legend()
        ax[1,0].set_xlabel('Thrust Angle [deg]')
        ax[1,0].set_ylabel('Entries')
        ax[1,0].set_title('Thrust Angle Distribution')
        ax[1,0].legend()
        ax[1,1].set_xlabel('Thrust Cosine')
        ax[1,1].set_ylabel('Entries')
        ax[1,1].set_title('Thrust Cosine Distribution')
        ax[1,1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'thrust_properties.png'))



        ###################################
        # study pion p4. Both reco and gen
        ###################################
        events_of_interest = events_dict.get(RegionNameOfInterest)
        # store pion p4
        flag_pi_plus = (events_of_interest['Part_charge'] == 1) & (abs(events_of_interest['Part_pdgId']) == 41)
        flag_pi_minus = (events_of_interest['Part_charge'] == -1) & (abs(events_of_interest['Part_pdgId']) == 41)
        reco_pi_plus_p4 = get_p4_from_ak_events(events_of_interest, flag_pi_plus, prefix='Part_fourMomentum')
        reco_pi_minus_p4 = get_p4_from_ak_events(events_of_interest, flag_pi_minus, prefix='Part_fourMomentum')
        if not dl.is_data:
            flag_truth_pi_plus = (events_of_interest['GenPart_pdgId'] == 211)
            flag_truth_pi_minus = (events_of_interest['GenPart_pdgId'] == -211)
            truth_pi_plus_p4 = get_p4_from_ak_events(events_of_interest, flag_truth_pi_plus, prefix='GenPart_vector')
            truth_pi_minus_p4 = get_p4_from_ak_events(events_of_interest, flag_truth_pi_minus, prefix='GenPart_vector')

        # angle between pion pairs
        reco_angle = reco_pi_plus_p4.deltaangle(reco_pi_minus_p4)
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(2.64, np.pi, 51)
        ax.hist(reco_angle, bins=bins, histtype='step', density=False, label='Reconstructed Pions', linestyle='solid', color='orange', linewidth=1.5)
        if not dl.is_data:
            truth_angle = truth_pi_plus_p4.deltaangle(truth_pi_minus_p4)
            ax.hist(truth_angle, bins=bins, histtype='step', density=False, label='Generated Pions', linestyle='dashed', color='blue', linewidth=1.5)
        ax.set_xlabel('Angle between $\pi^{+}$ and $\pi^{-}$ [rad]')
        ax.set_ylabel('Entries')
        ax.set_title('Angle between Pion Pairs')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cur_output_dir, 'pion_pair_angle.png'))
        if not dl.is_data:


            # angle between truth and reco pions
            angle_pi_plus_deg = truth_pi_plus_p4.deltaangle(reco_pi_plus_p4) * 180/np.pi
            angle_pi_minus_deg = truth_pi_minus_p4.deltaangle(reco_pi_minus_p4) * 180/np.pi
            fig, ax = plt.subplots(dpi=300)
            bins = np.linspace(0, 1, 51)
            ax.hist(angle_pi_plus_deg, bins=bins, histtype='step', density=False, label=r'$\pi^{+}$', color='blue', linewidth=1.5)
            ax.hist(angle_pi_minus_deg, bins=bins, histtype='step', density=False, label=r'$\pi^{-}$', color='orange', linewidth=1.5)
            ax.set_xlabel('Angle between Generated and Reconstructed Pions [deg]')
            ax.set_ylabel('Entries')
            ax.set_title('Angle between Generated and Reconstructed Pions')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(cur_output_dir, 'pion_reco_vs_truth_angle.png'))

            # energy difference between truth and reco pions
            energy_diff_pi_plus = reco_pi_plus_p4.E - truth_pi_plus_p4.E
            energy_diff_pi_minus = reco_pi_minus_p4.E - truth_pi_minus_p4.E
            fig, ax = plt.subplots(dpi=300)
            bins = np.linspace(-5, 5, 51)
            ax.hist(energy_diff_pi_plus, bins=bins, histtype='step', density=False, label=r'$\pi^{+}$', color='blue', linewidth=1.5)
            ax.hist(energy_diff_pi_minus, bins=bins, histtype='step', density=False, label=r'$\pi^{-}$', color='orange', linewidth=1.5)
            ax.set_xlabel('Reconstructed Energy - Generated Energy [GeV]')
            ax.set_ylabel('Entries')
            ax.set_title('Energy Difference between Generated and Reconstructed Pions')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(cur_output_dir, 'pion_reco_vs_truth_energy_diff.png'))

        # ###############################################
        # # study truth composition 
        # ################################################
        # if not dl.is_data:
        #     final_state_part = events_of_interest['GenPart_status'] == 1
        #     pdgIds = events_of_interest['GenPart_pdgId']
            
        #     num_pi_plus = ak.sum(pdgIds[final_state_part] == 211, axis=1)
        #     num_pi_minus = ak.sum(pdgIds[final_state_part] == -211, axis=1)
        #     num_pi0 = ak.sum(pdgIds == 111, axis=1)
        #     num_muons = ak.sum(abs(pdgIds[final_state_part]) == 13, axis=1)
        #     num_electrons = ak.sum(abs(pdgIds[final_state_part]) == 11, axis=1)
        #     num_kaons = ak.sum( (abs(pdgIds[final_state_part]) == 321), axis=1)

        #     categories = {
        #         'singlePi_singlePi': ak.sum((num_pi_plus == 1) & (num_pi_minus == 1) & (num_pi0 == 0) & (num_muons == 0) & (num_electrons == 0)),
        #         'singlePi_Pi1Pi0': ak.sum(( (num_pi_plus == 1) & (num_pi_minus == 1) ) & (num_pi0 == 1) & (num_muons == 0) & (num_electrons == 0)),
        #         'singlePi_PiXPi0': ak.sum(( (num_pi_plus == 1) & (num_pi_minus == 1) ) & (num_pi0 >= 2) & (num_muons == 0) & (num_electrons == 0)),
        #         'singlePi_leptonic': ak.sum(( (num_pi_plus + num_pi_minus) == 1 ) & ( (num_muons + num_electrons) == 1 ) & (num_pi0 == 0)),
        #         'Pi1Pi0_Pi1Pi0': ak.sum(( (num_pi_plus == 1) & (num_pi_minus == 1) ) & (num_pi0 == 2) & (num_muons == 0) & (num_electrons == 0)),
        #         'Pi1Pi0_leptonic': ak.sum(( (num_pi_plus + num_pi_minus) == 1 ) & ( (num_muons + num_electrons) == 1 ) & (num_pi0 == 1)),
        #         'leptonic_leptonic': ak.sum(( (num_muons + num_electrons) >= 2 )),
        #     }
        #     categories['others'] = len(events_of_interest) - sum(categories.values())
        #     print(categories['others'])

        #     fig, ax = plt.subplots(dpi=300)
        #     x_axis = list(categories.keys())
        #     ax.bar(x_axis, categories.values(), alpha=0.7,  linewidth=1, fill=False, edgecolor='blue')
        #     ax.set_xticklabels(x_axis, rotation=45, ha='right', fontsize='small')
        #     ax.set_xlabel('Truth Final State Categories')
        #     ax.set_ylabel('Counts')
        #     # ax.set_yscale('log')
        #     ax.set_title('Generated Final State Particle Categories in SR')
        #     fig.tight_layout()
        #     fig.savefig(os.path.join(cur_output_dir, 'gen_final_state_categories_sr.png'))
        #     # unique, counts = np.unique(ak.to_numpy(ak.flatten(pdgIds)), return_counts=True)
        #     # composition = dict(zip(unique, counts))

        #     # fig, ax = plt.subplots(dpi=300)
        #     # x_axis = [pdgid_parser.get(pdgId, str(pdgId)) for
        #     #             pdgId in composition.keys()]
        #     # ax.bar(x_axis, composition.values(), alpha=0.7,  linewidth=1.5, fill=False, edgecolor='blue')
        #     # ax.set_xlabel('Particle Type (PDG ID)')
        #     # ax.set_ylabel('Counts')
        #     # ax.set_yscale('log')
        #     # ax.set_title('Generated Final State Particle Composition in SR')
        #     # fig.tight_layout()
        #     # fig.savefig(os.path.join(cur_output_dir, 'gen_final_state_composition_sr.png'))

        #     # # number of truth particles
        #     # pdgIds = events_of_interest['GenPart_pdgId']
        #     # final_state_pdgIds = events_of_interest['GenPart_pdgId'][final_state_part]
        #     # num_pi_plus = ak.sum(final_state_pdgIds == 211, axis=1)
        #     # num_pi_minus = ak.sum(final_state_pdgIds == -211, axis=1)
        #     # num_pi_pm = ak.sum(abs(final_state_pdgIds) == 211, axis=1)
        #     # num_el = ak.sum(abs(final_state_pdgIds) == 11, axis=1)
        #     # num_mu = ak.sum(abs(final_state_pdgIds) == 13, axis=1)
        #     # num_pi0 = ak.sum(events_of_interest['GenPart_pdgId'] == 111, axis=1)
        #     # num_other_hadrons = ak.sum(
        #     #     (abs(pdgIds) == 221) | (abs(pdgIds) == 321) | (abs(pdgIds) == 223) | (abs(pdgIds) == 2112) | (abs(pdgIds) == 2212), axis=1)


        #     # fig, ax = plt.subplots(dpi=300)
        #     # bins = np.arange(0, 11, 1)
        #     # # ax.hist(ak.to_numpy(num_pi_plus), bins=bins, histtype='step', density=False, label=r'$\pi^{+}$', color='blue', linewidth=1.5)
        #     # # ax.hist(ak.to_numpy(num_pi_minus), bins=bins, histtype='step', density=False, label=r'$\pi^{-}$', color='orange', linewidth=1.5)
        #     # ax.hist(ak.to_numpy(num_pi_pm), bins=bins, histtype='step', density=False, label=r'$\pi^{\pm}$', color='blue', linewidth=1.5)
        #     # ax.hist(ak.to_numpy(num_el), bins=bins, histtype='step', density=False, label='Electrons', color='green', linewidth=1.5)
        #     # ax.hist(ak.to_numpy(num_mu), bins=bins, histtype='step', density=False, label='Muons', color='red', linewidth=1.5)
        #     # ax.hist(ak.to_numpy(num_pi0), bins=bins, histtype='step', density=False, label=r'$\pi^{0}$', color='purple', linewidth=1.5)
        #     # ax.hist(ak.to_numpy(num_other_hadrons), bins=bins, histtype='step', density=False, label='Other Hadrons', color='brown', linewidth=1.5)
        #     # ax.set_xlabel('Number of Generated Particles')
        #     # ax.set_ylabel('Entries')
        #     # ax.set_title('Number of Generated Particles per Event in SR')
        #     # ax.legend()
        #     # fig.tight_layout()
        #     # fig.savefig(os.path.join(cur_output_dir, 'num_gen_particles_sr.png'))

        

    def finalize(self):
        pass