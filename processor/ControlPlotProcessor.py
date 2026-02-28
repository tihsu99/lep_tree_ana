import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme
from utils.plotter import do_control_plot

    

def make_control_plots_tautau(dl_dict, luminosity, normalize, output_dir, region_name="tautau"):
    # isolation angle
    def get_isolation_angle(dl):
        events = dl.data.get(region_name)
        isolation_angle = ak.to_numpy(events['isolation_angle'], allow_missing=False)
        return isolation_angle

    bin_edges = np.linspace(140, 180, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_isolation_angle,
        bin_edges=bin_edges,
        x_label='Isolation Angle [deg]',
        title='Control Plot: Isolation Angle',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_isolation_angle.png")

    # Erad
    def get_erad(dl):
        events = dl.data.get(region_name)
        erad = ak.to_numpy(events['E_rad'], allow_missing=False)
        return erad
    bin_edges = np.linspace(0, 2, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_erad,
        bin_edges=bin_edges,
        x_label='E_rad',
        title='Control Plot: E_rad',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_erad.png")

    # p_rad
    def get_prad(dl):
        events = dl.data.get(region_name)
        prad = ak.to_numpy(events['P_rad'], allow_missing=False)
        return prad
    bin_edges = np.linspace(0, 2, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_prad,
        bin_edges=bin_edges,
        x_label='P_rad',
        title='Control Plot: P_rad',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_prad.png")

    # charged_E
    def get_charged_E(dl):
        events = dl.data.get(region_name)
        charged_E = ak.to_numpy(events['charged_E'], allow_missing=False)
        return charged_E
    bin_edges = np.linspace(0, cme, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_charged_E,
        bin_edges=bin_edges,
        x_label='charged_E',
        title='Control Plot: charged_E',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_chargedE.png")





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
        self.regions = config.get('regions', ['pipi'])
        # self.normalize = True

    def run(self, dl_dict):
        if 'tautau' in self.regions:
            output_dir_tautau = f"{self.output_dir}/tautau/"
            os.makedirs(output_dir_tautau, exist_ok=True)
            make_control_plots_tautau(
                dl_dict,
                luminosity=self.luminosity,
                normalize=self.normalize,
                output_dir=output_dir_tautau,
                region_name="tautau",
            )



    def finalize(self):
        pass