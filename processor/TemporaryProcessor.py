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
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_dir = self.config.get("output_dir", "./") + "/plots/"
        os.makedirs(self.output_dir, exist_ok=True)


    def run(self, dl: DataLoader.DataLoader):
        events_dict = dl.data
        RegionNameOfInterest = dl.region_of_interest
        

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
        fig.savefig(os.path.join(self.output_dir, 'num_reco_pions.png'))


        #######################################################
        # plot pt of reco and truth pions
        #######################################################
        # fig, ax = plt.subplots(dpi=300)
        fig, (ax, ax_ratio) = plt.subplots(2, 1, dpi=300, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
        bins = np.linspace(0, 50, 51)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            flag_reco_pi = (abs(events['Part_pdgId']) == 41)
            flag_gen_pi = (abs(events['GenPart_pdgId']) == 211)

            pt_reco_pions = (events['Part_fourMomentum_fCoordinates_fX'][flag_reco_pi]**2 + events['Part_fourMomentum_fCoordinates_fY'][flag_reco_pi]**2)**0.5
            pt_gen_pions = (events['GenPart_vector_fCoordinates_fX'][flag_gen_pi]**2 + events['GenPart_vector_fCoordinates_fY'][flag_gen_pi]**2)**0.5

            # ratio plot
            hist_reco, bin_edges = np.histogram(ak.to_numpy(ak.flatten(pt_reco_pions)), bins=bins)
            hist_gen, _ = np.histogram(ak.to_numpy(ak.flatten(pt_gen_pions)), bins=bins)
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
                xlabel='Pion $p_{T}$ [GeV]',
                ylabel='Entries',
                title='Pion $p_{T}$ Distribution',
                ratio_color=color,
                ratio_ylabel='Reco / Gen',
            )

        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'pt_distribution.png'))

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
        unique_pdgIds = np.unique(ak.flatten(events_dict['raw']['Part_pdgId']))
        x_axis = [pdgid_parser.get(pdgId, str(pdgId)) for pdgId in unique_pdgIds]
        # pt distribution for each pdgId
        fig_pt, ax_pt = plt.subplots(dpi=300)
        bins_pt = np.linspace(0, 100, 51)

        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            pdgId_coutnts = {}
            for pdgId in unique_pdgIds:
                flag = (events['Part_pdgId'] == pdgId)
                count = ak.sum(ak.sum(flag, axis=1))
                pdgId_coutnts[pdgid_parser.get(pdgId, str(pdgId))] = count
                if label==RegionNameOfInterest:
                    # pt distribution for each pdgId
                    pt_values = (events['Part_fourMomentum_fCoordinates_fX'][flag]**2 + events['Part_fourMomentum_fCoordinates_fY'][flag]**2)**0.5
                    ax_pt.hist(ak.to_numpy(ak.flatten(pt_values)), bins=bins_pt, histtype='step', density=False, label=f'{pdgid_parser.get(pdgId, str(pdgId))}', linewidth=1.5)

            ax.bar(x_axis, [pdgId_coutnts[x] for x in x_axis], label=label, alpha=0.7,  linewidth=1.5, fill=False, edgecolor=next(color_iter))

        ax.set_xlabel('Particle Type (PDG ID)')
        ax.set_ylabel('Entries')
        ax.set_yscale('log')
        ax.set_title('Reconstructed Particle PDG ID Distribution')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'pdgId_distribution.png'))

        ax_pt.set_xlabel('Reconstructed Particle $p_{T}$ [GeV]')
        ax_pt.set_ylabel('Entries')
        ax_pt.set_yscale('log')
        ax_pt.set_title('Reconstructed Particle $p_{T}$ Distribution in SR')
        ax_pt.legend()
        fig_pt.tight_layout()
        fig_pt.savefig(os.path.join(self.output_dir, 'pdgId_pt_distribution_sr.png'))

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
        fig.savefig(os.path.join(self.output_dir, 'sum_reco_pt.png'))

        # #######################################################
        # # plot total E
        # #######################################################
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
        # fig.savefig(os.path.join(self.output_dir, 'total_energy_components.png'))

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
            missing_E = cme - sum_E

            ax[0].hist(ak.to_numpy(missing_pT), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
            ax[1].hist(ak.to_numpy(missing_pz), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)
            ax[2].hist(ak.to_numpy(missing_E), bins=bins, histtype='step', density=False, label=label, color=color, linewidth=1.5)

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
        fig.savefig(os.path.join(self.output_dir, 'missing_momentum_energy.png'))


        #######################################################
        # plot m_{pi+ pi- + missing}
        #######################################################
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(0, 100, 51)
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            sum_px = ak.sum(events['Part_fourMomentum_fCoordinates_fX'], axis=1)
            sum_py = ak.sum(events['Part_fourMomentum_fCoordinates_fY'], axis=1)
            sum_pz = ak.sum(events['Part_fourMomentum_fCoordinates_fZ'], axis=1)
            sum_E = ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=1)

            missing_px = -sum_px
            missing_py = -sum_py
            missing_pz = -sum_pz
            missing_E = cme - sum_E

            missing_p4 = vector.zip({
                "px": missing_px,
                "py": missing_py,
                "pz": missing_pz,
                "E": missing_E,
            })

            # pion p4
            flag_pi_plus = (events['Part_charge'] == 1) & (abs(events['Part_pdgId']) == 41)
            flag_pi_minus = (events['Part_charge'] == -1) & (abs(events['Part_pdgId']) == 41)
            reco_pi_plus_p4 = get_p4(events, flag_pi_plus, prefix='Part_fourMomentum')
            reco_pi_minus_p4 = get_p4(events, flag_pi_minus, prefix='Part_fourMomentum')

            di_pion_p4 = reco_pi_plus_p4 + reco_pi_minus_p4
            total_p4 = di_pion_p4 + missing_p4
            total_mass = total_p4.mass

            ax.hist(ak.to_numpy(total_mass), bins=bins, histtype='step', density=False, label=f"{label} - pi+ pi- + missing", linestyle='solid', color=color, linewidth=1.5)
            ax.hist(ak.to_numpy(di_pion_p4.mass), bins=bins, histtype='step', density=False, label=f'{label} - di-pion', linestyle='dashed', color=color, linewidth=1.5)
        ax.set_xlabel('Invariant Mass [GeV]')
        ax.set_ylabel('Entries')
        ax.set_title('Invariant Mass Distribution')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'invariant_mass_distribution.png'))


        #######################################################
        # plot thrust magnitude and theta
        #######################################################
        fig, ax = plt.subplots(2, 2, dpi=300, figsize=(12, 10))
        color_iter = get_color_iterator(len(events_dict))
        for label, events in events_dict.items():
            color = next(color_iter)
            thrust_magnitude = events['thrust_Mag'] + 1e-12 # avoid division by zero
            thrust_x = events['thrust_x']
            thrust_y = events['thrust_y']
            thrust_z = events['thrust_z']
            thrust_cosine = (thrust_z / thrust_magnitude)
            thrust_theta = np.arccos(thrust_cosine)  # in radians
            # plot thrust magnitude
            ax[0,0].hist(ak.to_numpy(thrust_magnitude), bins=np.linspace(0, 1, 51), histtype='step', density=False, label=label, color=color, linewidth=1.5)
            # log(1-Thrust)
            ax[0,1].hist(ak.to_numpy(-np.log10(1 - thrust_magnitude + 1e-6)), bins=np.linspace(0, 6, 51), histtype='step', density=False, label=label, color=color, linewidth=1.5)
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
        fig.savefig(os.path.join(self.output_dir, 'thrust_properties.png'))



        ###################################
        # study pion p4. Both reco and gen
        ###################################
        events_of_interest = events_dict.get(RegionNameOfInterest)
        # store pion p4
        flag_pi_plus = (events_of_interest['Part_charge'] == 1) & (abs(events_of_interest['Part_pdgId']) == 41)
        flag_pi_minus = (events_of_interest['Part_charge'] == -1) & (abs(events_of_interest['Part_pdgId']) == 41)
        flag_truth_pi_plus = (events_of_interest['GenPart_pdgId'] == 211)
        flag_truth_pi_minus = (events_of_interest['GenPart_pdgId'] == -211)
        reco_pi_plus_p4 = get_p4(events_of_interest, flag_pi_plus, prefix='Part_fourMomentum')
        reco_pi_minus_p4 = get_p4(events_of_interest, flag_pi_minus, prefix='Part_fourMomentum')
        truth_pi_plus_p4 = get_p4(events_of_interest, flag_truth_pi_plus, prefix='GenPart_vector')
        truth_pi_minus_p4 = get_p4(events_of_interest, flag_truth_pi_minus, prefix='GenPart_vector')

        # angle between pion pairs
        reco_angle = reco_pi_plus_p4.deltaangle(reco_pi_minus_p4)
        truth_angle = truth_pi_plus_p4.deltaangle(truth_pi_minus_p4)
        fig, ax = plt.subplots(dpi=300)
        bins = np.linspace(4./5*np.pi, np.pi, 51)
        ax.hist(truth_angle, bins=bins, histtype='step', density=False, label='Generated Pions', linestyle='dashed', color='blue', linewidth=1.5)
        ax.hist(reco_angle, bins=bins, histtype='step', density=False, label='Reconstructed Pions', linestyle='solid', color='orange', linewidth=1.5)
        ax.set_xlabel('Angle between $\pi^{+}$ and $\pi^{-}$ [rad]')
        ax.set_ylabel('Entries')
        ax.set_title('Angle between Pion Pairs')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'pion_pair_angle.png'))

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
        fig.savefig(os.path.join(self.output_dir, 'pion_reco_vs_truth_angle.png'))

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
        fig.savefig(os.path.join(self.output_dir, 'pion_reco_vs_truth_energy_diff.png'))

        



    def finalize(self):
        pass