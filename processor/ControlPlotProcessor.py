import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme
from utils.plotter import do_control_plot


class ControlPlotProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        """
        Processor to make control plots for data/MC comparison.
        """
        super().__init__(config)
        self.config = config
        if 'output_dir_name' in config:
            output_dir = f"{output_dir}/{config['output_dir_name']}"
        else:
            output_dir = f"{output_dir}/control_plots/"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.luminosity = config.get('luminosity', None) 
        self.normalize = (self.luminosity is None)
        # self.normalize = True

    def run(self, dl_dict):
        # temporary cut
        for dl_name, dl in dl_dict.items():
            events = dl.data.get(dl.region_of_interest)
            reco_abs_pdgId = np.abs(events['Part_pdgId'])
            reco_charge = events['Part_charge']
            flag_pion = (reco_abs_pdgId == 41)
            flag_piplus = flag_pion & (reco_charge == 1)
            flag_piminus = flag_pion & (reco_charge == -1)
            p4_piplus = get_p4_from_ak_events(events, flag_piplus)
            p4_piminus = get_p4_from_ak_events(events, flag_piminus)
            p4_dipion = p4_piplus + p4_piminus

            # # P_rad = (p4_piplus.p**2 + p4_piminus.p**2)**0.5
            # P_rad = ((p4_piplus.px**2 + p4_piplus.py**2 + p4_piplus.pz**2) + (p4_piminus.px**2 + p4_piminus.py**2 + p4_piminus.pz**2))**0.5
            # events['P_rad'] = P_rad
            # E_rad = (p4_piplus.E**2 + p4_piminus.E**2)**0.5
            # events['E_rad'] = E_rad

            # dl.region_of_interest = 'test_selection'
            # selection = ak.ones_like(events['evtNumber'], dtype=bool)
            # # cut on pi pair angle
            # angle_between_pions = p4_piplus.deltaangle(p4_piminus)
            # selection = (angle_between_pions > 2.99) & (angle_between_pions < 3.1) & selection
            # # selection = (angle_between_pions > 3.1) & selection
            # # cut on di-pion mass
            # selection = (p4_dipion.mass < 85) & (p4_dipion.mass > 10)  & selection
            # # cut on total energy
            # total_energy = ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=-1)
            # selection = (total_energy < 80) & (total_energy > 20) & selection

            # # cut on number of reco particles
            # num_reco_particles = ak.num(events['Part_pdgId'])
            # selection = (num_reco_particles < 3) & selection

            # # cut on missing momentum
            # missing_px = -ak.sum(events['Part_fourMomentum_fCoordinates_fX'], axis=-1)
            # missing_py = -ak.sum(events['Part_fourMomentum_fCoordinates_fY'], axis=-1)
            # missing_pz = -ak.sum(events['Part_fourMomentum_fCoordinates_fZ'], axis=-1)
            # missing_p = np.sqrt(missing_px**2 + missing_py**2 + missing_pz**2)
            # selection = (missing_p < 40) & selection

            # # cut on P_rad
            # selection = (P_rad < cme/2) & selection

            # # cut on log10(1 - thrust)
            # thrust_magnitude = events['thrust_Mag']
            # log10_1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
            # selection = (log10_1mthrust > 2.5) & selection



            # # cut on E_rad
            # selection = (E_rad < cme/2 * 0.7) & selection

            # # cut on cos(thrust)
            # thrust_z = events['thrust_z']
            # thrust_cosine = (thrust_z / thrust_magnitude)
            # selection = (np.abs(thrust_cosine) < 0.7) & selection

            # # cut on sum pion theta
            # p4_pion_sum = p4_piplus + p4_piminus
            # pion_sum_theta = abs(p4_pion_sum.theta * 180 / np.pi - 90)  # convert to degrees
            # selection = ((pion_sum_theta < 30) | (pion_sum_theta > 50)) & selection

            # piplus_theta = abs(p4_piplus.theta * 180 / np.pi - 90)  # convert to degrees
            # piminus_theta = abs(p4_piminus.theta * 180 / np.pi - 90)  # convert to degrees
            # selection = ((piplus_theta < 30) | (piplus_theta > 50)) & ((piminus_theta < 30) | (piminus_theta > 50)) & selection
            # pi_mean_theta = 0.5 * (piplus_theta + piminus_theta)
            # selection = ((pi_mean_theta < 30) | (pi_mean_theta > 50)) & selection


            # dl.data[dl.region_of_interest] = events[selection]
            # if dl.is_data:
            #     print(f"Remaining data events after selection: {len(dl.data.get(dl.region_of_interest))}")

        ########################################################
        # p_rad and E_rad
        ########################################################
        def get_p_rad(dl):
            events = dl.data.get(dl.region_of_interest)
            return events['P_rad']
        bin_edges = np.linspace(0, 40, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_p_rad,
            bin_edges=bin_edges,
            x_label='P_rad [GeV]',
            title='Control Plot: P_rad',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        # ax.vlines(x=cme/2, ymin=0, ymax=ax.get_ylim()[1], colors='b', linestyles='dashed', label='cme/2')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_p_rad.png")
        # def get_e_rad(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     return events['E_rad']
        # bin_edges = np.linspace(0, 100, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_e_rad,
        #     bin_edges=bin_edges,
        #     x_label='E_rad [GeV]',
        #     title='Control Plot: E_rad',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # ax.vlines(x=cme/2, ymin=0, ymax=ax.get_ylim()[1], colors='b', linestyles='dashed', label='cme/2')
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_e_rad.png")

        ########################################################
        # total energy
        ########################################################
        def get_total_energy(dl):
            events = dl.data.get(dl.region_of_interest)
            total_energy = ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=-1)
            return total_energy
        bin_edges = np.linspace(20, 80, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_total_energy,
            bin_edges=bin_edges,
            x_label='Total Energy [GeV]',
            title='Control Plot: Total Energy',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_total_energy.png")

        ########################################################
        # theta of pions
        ########################################################
        def get_pion_theta(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_pdgId = events['Part_pdgId']
            reco_abs_pdgId = np.abs(reco_pdgId)
            flag_pion = (reco_abs_pdgId == 41)
            p4_pion = get_sum_p4_from_ak_events(events, flag_pion)
            theta = p4_pion.theta * 180 / np.pi  # convert to degrees
            return theta

        bin_edges = np.linspace(0, 180, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_pion_theta,
            bin_edges=bin_edges,
            x_label='sum Pion Theta [degrees]',
            title='Control Plot: Pion Theta',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_sum_pion_theta.png")

        def get_piplus_theta(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_pdgId = events['Part_pdgId']
            reco_charge = events['Part_charge']
            reco_abs_pdgId = np.abs(reco_pdgId)
            flag_pion = (reco_abs_pdgId == 41)
            flag_piplus = flag_pion & (reco_charge == 1)
            p4_piplus = get_p4_from_ak_events(events, flag_piplus)
            theta = p4_piplus.theta * 180 / np.pi  # convert to degrees
            return theta
        bin_edges = np.linspace(0, 180, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_piplus_theta,
            bin_edges=bin_edges,
            x_label='Pion+ Theta [degrees]',
            title='Control Plot: Pion+ Theta',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_pion_plus_theta.png")
        def get_piminus_theta(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_pdgId = events['Part_pdgId']
            reco_charge = events['Part_charge']
            reco_abs_pdgId = np.abs(reco_pdgId)
            flag_pion = (reco_abs_pdgId == 41)
            flag_piminus = flag_pion & (reco_charge == -1)
            p4_piminus = get_p4_from_ak_events(events, flag_piminus)
            theta = p4_piminus.theta * 180 / np.pi  # convert to degrees
            return theta
        bin_edges = np.linspace(0, 180, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_piminus_theta,
            bin_edges=bin_edges,
            x_label='Pion- Theta [degrees]',
            title='Control Plot: Pion- Theta',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_pion_minus_theta.png")

        def get_piplus_phi(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_pdgId = events['Part_pdgId']
            reco_charge = events['Part_charge']
            reco_abs_pdgId = np.abs(reco_pdgId)
            flag_pion = (reco_abs_pdgId == 41)
            flag_piplus = flag_pion & (reco_charge == 1)
            p4_piplus = get_p4_from_ak_events(events, flag_piplus)
            phi = p4_piplus.phi * 180 / np.pi  # convert to degrees
            return phi
        bin_edges = np.linspace(-180, 180, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_piplus_phi,
            bin_edges=bin_edges,
            x_label='Pion+ Phi [degrees]',
            title='Control Plot: Pion+ Phi',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_pion_plus_phi.png")
        def get_piminus_phi(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_pdgId = events['Part_pdgId']
            reco_charge = events['Part_charge']
            reco_abs_pdgId = np.abs(reco_pdgId)
            flag_pion = (reco_abs_pdgId == 41)
            flag_piminus = flag_pion & (reco_charge == -1)
            p4_piminus = get_p4_from_ak_events(events, flag_piminus)
            phi = p4_piminus.phi * 180 / np.pi  # convert to degrees
            return phi
        bin_edges = np.linspace(-180, 180, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_piminus_phi,
            bin_edges=bin_edges,
            x_label='Pion- Phi [degrees]',
            title='Control Plot: Pion- Phi',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_pion_minus_phi.png")


        ########################################################
        # pion pm pt
        #########################################################
        # def get_pion_plus_pt(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     reco_pdgId = events['Part_pdgId']
        #     reco_charge = events['Part_charge']
        #     reco_abs_pdgId = np.abs(reco_pdgId)
        #     flag_pion = (reco_abs_pdgId == 41)
        #     flag_piplus = flag_pion & (reco_charge == 1)
        #     p4_piplus = get_p4_from_ak_events(events, flag_piplus)
        #     return p4_piplus.pt
        # bin_edges = np.linspace(0, 70, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_pion_plus_pt,
        #     bin_edges=bin_edges,
        #     x_label='Transverse Momentum of Pion+ [GeV]',
        #     title='Control Plot: Pion+ Transverse Momentum',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_pion_plus_pt.png")
        # def get_pion_minus_pt(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     reco_pdgId = events['Part_pdgId']
        #     reco_charge = events['Part_charge']
        #     reco_abs_pdgId = np.abs(reco_pdgId)
        #     flag_pion = (reco_abs_pdgId == 41)
        #     flag_piminus = flag_pion & (reco_charge == -1)
        #     p4_piminus = get_p4_from_ak_events(events, flag_piminus)
        #     return p4_piminus.pt
        # bin_edges = np.linspace(0, 70, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_pion_minus_pt,
        #     bin_edges=bin_edges,
        #     x_label='Transverse Momentum of Pion- [GeV]',
        #     title='Control Plot: Pion- Transverse Momentum',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_pion_minus_pt.png")
        for charge in ['plus', 'minus']:
            def get_pion_pt(dl, charge=charge):
                events = dl.data.get(dl.region_of_interest)
                reco_pdgId = events['Part_pdgId']
                reco_charge = events['Part_charge']
                reco_abs_pdgId = np.abs(reco_pdgId)
                flag_pion = (reco_abs_pdgId == 41)
                if charge == 'plus':
                    flag = flag_pion & (reco_charge == 1)
                else:
                    flag = flag_pion & (reco_charge == -1)
                p4_pion = get_p4_from_ak_events(events, flag)
                return p4_pion.pt
            bin_edges = np.linspace(0, 70, 21)
            fig, ax, ax_ratio = do_control_plot(
                dl_dict,
                func_get_variable=lambda dl, charge=charge: get_pion_pt(dl, charge),
                bin_edges=bin_edges,
                x_label=f'Transverse Momentum of Pion{charge} [GeV]',
                title=f'Control Plot: Pion{charge} Transverse Momentum',
                luminosity=self.luminosity, normalize=self.normalize,
            )
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/control_plot_pion_{charge}_pt.png") 

            def get_pion_px(dl, charge=charge):
                events = dl.data.get(dl.region_of_interest)
                reco_pdgId = events['Part_pdgId']
                reco_charge = events['Part_charge']
                reco_abs_pdgId = np.abs(reco_pdgId)
                flag_pion = (reco_abs_pdgId == 41)
                if charge == 'plus':
                    flag = flag_pion & (reco_charge == 1)
                else:
                    flag = flag_pion & (reco_charge == -1)
                p4_pion = get_p4_from_ak_events(events, flag)
                return p4_pion.px
            bin_edges = np.linspace(-70, 70, 21)
            fig, ax, ax_ratio = do_control_plot(
                dl_dict,
                func_get_variable=lambda dl, charge=charge: get_pion_px(dl, charge),
                bin_edges=bin_edges,
                x_label=f'Px of Pion{charge} [GeV]',
                title=f'Control Plot: Pion{charge} Px',
                luminosity=self.luminosity, normalize=self.normalize,
            )
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/control_plot_pion_{charge}_px.png")
            def get_pion_py(dl, charge=charge):
                events = dl.data.get(dl.region_of_interest)
                reco_pdgId = events['Part_pdgId']
                reco_charge = events['Part_charge']
                reco_abs_pdgId = np.abs(reco_pdgId)
                flag_pion = (reco_abs_pdgId == 41)
                if charge == 'plus':
                    flag = flag_pion & (reco_charge == 1)
                else:
                    flag = flag_pion & (reco_charge == -1)
                p4_pion = get_p4_from_ak_events(events, flag)
                return p4_pion.py
            bin_edges = np.linspace(-70, 70, 21)
            fig, ax, ax_ratio = do_control_plot(
                dl_dict,
                func_get_variable=lambda dl, charge=charge: get_pion_py(dl, charge),
                bin_edges=bin_edges,
                x_label=f'Py of Pion{charge} [GeV]',
                title=f'Control Plot: Pion{charge} Py',
                luminosity=self.luminosity, normalize=self.normalize,
            )
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/control_plot_pion_{charge}_py.png")
            def get_pion_pz(dl, charge=charge):
                events = dl.data.get(dl.region_of_interest)
                reco_pdgId = events['Part_pdgId']
                reco_charge = events['Part_charge']
                reco_abs_pdgId = np.abs(reco_pdgId)
                flag_pion = (reco_abs_pdgId == 41)
                if charge == 'plus':
                    flag = flag_pion & (reco_charge == 1)
                else:
                    flag = flag_pion & (reco_charge == -1)
                p4_pion = get_p4_from_ak_events(events, flag)
                return p4_pion.pz
            bin_edges = np.linspace(-100, 100, 21)
            fig, ax, ax_ratio = do_control_plot(
                dl_dict,
                func_get_variable=lambda dl, charge=charge: get_pion_pz(dl, charge),
                bin_edges=bin_edges,
                x_label=f'Pz of Pion{charge} [GeV]',
                title=f'Control Plot: Pion{charge} Pz',
                luminosity=self.luminosity, normalize=self.normalize,
            )
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/control_plot_pion_{charge}_pz.png")

            

        ########################################################
        # num recon particles
        #########################################################
        def get_num_reco_particles(dl):
            events = dl.data.get(dl.region_of_interest)
            num_particles = ak.num(events['Part_pdgId'])
            return num_particles
        bin_edges = np.array([2, 3])
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_num_reco_particles,
            bin_edges=bin_edges,
            x_label='Number of Reconstructed Particles',
            title='Control Plot: Number of Reconstructed Particles',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_num_reco_particles.png")

        ########################################################
        #  Invariant mass of di-pion system
        ########################################################
        def get_dipion_mass(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_pdgId = events['Part_pdgId']
            reco_abs_pdgId = np.abs(reco_pdgId)
            flag_pion = (reco_abs_pdgId == 41)
            p4_pipi = get_sum_p4_from_ak_events(events, flag_pion)
            return p4_pipi.mass

        bin_edges = np.linspace(10, 85, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_dipion_mass,
            bin_edges=bin_edges,
            x_label='Invariant Mass [GeV]',
            title='Control Plot: Invariant Mass of Di-Pion',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        # ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_dipion_mass.png")

        ################################################
        # Angle between two pions
        ################################################
        def get_angle_between_pions(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_pdgId = events['Part_pdgId']
            reco_charge = events['Part_charge']
            reco_abs_pdgId = np.abs(reco_pdgId)
            flag_pion = (reco_abs_pdgId == 41)
            flag_piplus = flag_pion & (reco_charge == 1)
            flag_piminus = flag_pion & (reco_charge == -1)
            p4_piplus = get_p4_from_ak_events(events, flag_piplus)
            p4_piminus = get_p4_from_ak_events(events, flag_piminus)
            angles = p4_piplus.deltaangle(p4_piminus)
            return angles
        bin_edges = np.linspace(2.99, 3.1, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_angle_between_pions,
            bin_edges=bin_edges,
            x_label='Angle between two pions [rad]',
            title='Control Plot: Angle between Two Pions',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_angle_between_pions.png")

        # ################################################
        # # Thrust costh
        # ################################################
        # def get_thrust_costh(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     thrust_magnitude = events['thrust_Mag'] # avoid division by zero
        #     thrust_z = events['thrust_z']
        #     thrust_cosine = (thrust_z / thrust_magnitude)
        #     return thrust_cosine
        # bin_edges = np.linspace(-1, 1, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_thrust_costh,
        #     bin_edges=bin_edges,
        #     x_label=r'cos(\theta_{thrust})',
        #     title=r'Control Plot: Thrust cos(\theta)',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_thrust_costh.png")

        # def get_thrust_phi(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     thrust_x = events['thrust_x']
        #     thrust_y = events['thrust_y']
        #     thrust_phi = np.arctan2(thrust_y, thrust_x)
        #     return thrust_phi
        # bin_edges = np.linspace(-np.pi, np.pi, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_thrust_phi,
        #     bin_edges=bin_edges,
        #     x_label=r'$\phi_{thrust}$ [rad]',
        #     title=r'Control Plot: Thrust $\phi$',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_thrust_phi.png")

        # def get_thrust_x(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     thrust_x = events['thrust_x']
        #     return thrust_x
        # bin_edges = np.linspace(-1, 1, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_thrust_x,
        #     bin_edges=bin_edges,
        #     x_label=r'Thrust X',
        #     title=r'Control Plot: Thrust X',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_thrust_x.png")
        # def get_thrust_y(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     thrust_y = events['thrust_y']
        #     return thrust_y
        # bin_edges = np.linspace(-1, 1, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_thrust_y,
        #     bin_edges=bin_edges,
        #     x_label=r'Thrust Y',
        #     title=r'Control Plot: Thrust Y',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_thrust_y.png")
        # def get_thrust_z(dl):
        #     events = dl.data.get(dl.region_of_interest)
        #     thrust_z = events['thrust_z']
        #     return thrust_z
        # bin_edges = np.linspace(-1, 1, 21)
        # fig, ax, ax_ratio = do_control_plot(
        #     dl_dict,
        #     func_get_variable=get_thrust_z,
        #     bin_edges=bin_edges,
        #     x_label=r'Thrust Z',
        #     title=r'Control Plot: Thrust Z',
        #     luminosity=self.luminosity, normalize=self.normalize,
        # )
        # plt.tight_layout()
        # plt.savefig(f"{self.output_dir}/control_plot_thrust_z.png")


        ################################################
        # -log10(1-thrust)
        ################################################
        def get_neglog1mthrust(dl):
            events = dl.data.get(dl.region_of_interest)
            thrust_magnitude = events['thrust_Mag']
            neglog1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
            return neglog1mthrust
        bin_edges = np.linspace(2.5, 4, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_neglog1mthrust,
            bin_edges=bin_edges,
            x_label=r'-log10(1 - thrust)',
            title=r'Control Plot: -log10(1 - thrust)',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_thrust_neglog1mthrust.png")


        ################################################
        # pt of charged particles
        ################################################
        def get_charged_particle_pt(dl):
            events = dl.data.get(dl.region_of_interest)
            reco_charge = events['Part_charge']
            flag_charged = (reco_charge != 0)
            p4_charged = get_sum_p4_from_ak_events(events, flag_charged)
            return p4_charged.pt
        bin_edges = np.linspace(0, 40, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_charged_particle_pt,
            bin_edges=bin_edges,
            x_label='Transverse Momentum of Charged Particles [GeV]',
            title='Control Plot: pt of Charged Particles',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_charged_particle_pt.png")

        ################################################
        # pt of all particles
        ################################################
        def get_total_particle_pt(dl):
            events = dl.data.get(dl.region_of_interest)
            flag = ak.ones_like(events['Part_charge'], dtype=bool)
            p4 = get_sum_p4_from_ak_events(events, flag)
            return p4.pt
        bin_edges = np.linspace(0, 100, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_total_particle_pt,
            bin_edges=bin_edges,
            x_label='Transverse Momentum of All Particles [GeV]',
            title='Control Plot: pt of All Particles',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_total_particle_pt.png")

        ################################################
        # HT
        ################################################
        def get_ht(dl):
            events = dl.data.get(dl.region_of_interest)
            flag = ak.ones_like(events['Part_charge'], dtype=bool)
            p4_all = get_all_p4_from_ak_events(events, flag)
            ht = ak.sum(p4_all.pt, axis=-1)
            return ht
        bin_edges = np.linspace(10, 65, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_ht,
            bin_edges=bin_edges,
            x_label='Ht [GeV]',
            title='Control Plot: HT',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_ht.png")



        ################################################
        # Missing transverse and longitudinal momentum
        ################################################
        def get_missing_momentum(dl):
            events = dl.data.get(dl.region_of_interest)
            missing_px = -ak.sum(events['Part_fourMomentum_fCoordinates_fX'], axis=-1)
            missing_py = -ak.sum(events['Part_fourMomentum_fCoordinates_fY'], axis=-1)
            missing_pz = -ak.sum(events['Part_fourMomentum_fCoordinates_fZ'], axis=-1)
            missing_p = np.sqrt(missing_px**2 + missing_py**2 + missing_pz**2)
            return missing_p
        bin_edges = np.linspace(0, 40, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_missing_momentum,
            bin_edges=bin_edges,
            x_label='Missing Momentum [GeV]',
            title='Control Plot: Missing Momentum',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        # ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_missing_momentum.png")

        # missing pt
        def get_missing_pt(dl):
            events = dl.data.get(dl.region_of_interest)
            missing_px = -ak.sum(events['Part_fourMomentum_fCoordinates_fX'], axis=-1)
            missing_py = -ak.sum(events['Part_fourMomentum_fCoordinates_fY'], axis=-1)
            missing_pt = np.sqrt(missing_px**2 + missing_py**2)
            return missing_pt
        bin_edges = np.linspace(0, 40, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_missing_pt,
            bin_edges=bin_edges,
            x_label='Missing Pt [GeV]',
            title='Control Plot: Missing Pt',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_missing_pt.png")

        # missing pz
        def get_missing_pz(dl):
            events = dl.data.get(dl.region_of_interest)
            missing_pz = -ak.sum(events['Part_fourMomentum_fCoordinates_fZ'], axis=-1)
            return missing_pz
        bin_edges = np.linspace(-25, 25, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_missing_pz,
            bin_edges=bin_edges,
            x_label='Missing Pz [GeV]',
            title='Control Plot: Missing Pz',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_missing_pz.png")
        # missing E
        def get_missing_energy(dl):
            events = dl.data.get(dl.region_of_interest)
            missing_E = cme - ak.sum(events['Part_fourMomentum_fCoordinates_fT'], axis=-1)
            return missing_E
        bin_edges = np.linspace(20, 90, 21)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_missing_energy,
            bin_edges=bin_edges,
            x_label='Missing Energy [GeV]',
            title='Control Plot: Missing Energy',
            luminosity=self.luminosity, normalize=self.normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_missing_energy.png")




    def finalize(self):
        pass