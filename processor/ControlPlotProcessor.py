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
        # Example control plot: Invariant mass of di-pion system
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
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/control_plot_dipion_mass.png")


    def finalize(self):
        pass