import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events
from utils.plotter import do_control_plot


class ControlPlotProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        """
        Processor to make control plots for data/MC comparison.
        """
        super().__init__(config)
        self.config = config
        self.output_dir = f"{output_dir}/control_plots/"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dl_dict):
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

        bin_edges = np.linspace(0, 100, 51)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_dipion_mass,
            bin_edges=bin_edges,
            x_label='Invariant Mass [GeV]',
            title='Control Plot: Invariant Mass of Di-Pion'
        )
        ax.set_yscale('log')
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
        bin_edges = np.linspace(0, np.pi, 51)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_angle_between_pions,
            bin_edges=bin_edges,
            x_label='Angle between two pions [rad]',
            title='Control Plot: Angle between Two Pions'
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_angle_between_pions.png")

        ################################################
        # Thrust costh
        ################################################
        def get_thrust_costh(dl):
            events = dl.data.get(dl.region_of_interest)
            thrust_magnitude = events['thrust_Mag'] # avoid division by zero
            thrust_z = events['thrust_z']
            thrust_cosine = (thrust_z / thrust_magnitude)
            return thrust_cosine
        bin_edges = np.linspace(-1, 1, 51)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_thrust_costh,
            bin_edges=bin_edges,
            x_label=r'cos(\theta_{thrust})',
            title=r'Control Plot: Thrust cos(\theta)'
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_thrust_costh.png")

        ################################################
        # -log10(1-thrust)
        ################################################
        def get_neglog1mthrust(dl):
            events = dl.data.get(dl.region_of_interest)
            thrust_magnitude = events['thrust_Mag']
            neglog1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
            return neglog1mthrust
        bin_edges = np.linspace(0, 10, 51)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_neglog1mthrust,
            bin_edges=bin_edges,
            x_label=r'-log10(1 - thrust)',
            title=r'Control Plot: -log10(1 - thrust)'
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_neglog1mthrust.png")




    def finalize(self):
        pass