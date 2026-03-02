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

    ###################################################
    # Common event variables
    ###################################################

    # HT
    def get_ht(dl):
        events = dl.data.get(region_name)
        flag = ak.ones_like(events['Part_charge'], dtype=bool)
        p4_all = get_all_p4_from_ak_events(events, flag)
        ht = ak.sum(p4_all.pt, axis=-1)
        return ht
    bin_edges = np.linspace(0, 100, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_ht,
        bin_edges=bin_edges,
        x_label='Ht [GeV]',
        title='Control Plot: HT',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_ht.png")

    # missing pT
    def get_missing_pt(dl):
        events = dl.data.get(region_name)
        missing_pt = ak.to_numpy(events['missing_pt'], allow_missing=False)
        return missing_pt
    bin_edges = np.linspace(0, 100, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_missing_pt,
        bin_edges=bin_edges,
        x_label='Missing pT [GeV]',
        title='Control Plot: Missing pT',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_missing_pt.png")


    # nprong
    def get_nprong(dl):
        events = dl.data.get(region_name)
        nprong = ak.to_numpy(events['nprong'], allow_missing=False)
        return nprong

    bin_edges = np.arange(2, 7, 1)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_nprong,
        bin_edges=bin_edges,
        x_label='nprong',
        title='Control Plot: nprong',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_nprong.png")

    
    # -log10(1-thrust)
    def get_neglog1mthrust(dl):
        events = dl.data.get(region_name)
        thrust_magnitude = events['thrust_Mag']
        neglog1mthrust = -np.log10(1 - thrust_magnitude + 1e-10) # avoid log(0)
        return neglog1mthrust
    bin_edges = np.linspace(0, 10, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_neglog1mthrust,
        bin_edges=bin_edges,
        x_label=r'-log10(1 - thrust)',
        title=r'Control Plot: -log10(1 - thrust)',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_thrust_neglog1mthrust.png")


    ################################################
    # lead part in each hemisphere
    ################################################

    # Four momentum of lead part in each hemisphere
    for var in ['pt', 'theta', 'phi', 'E']:
        def get_lead_part_var(dl, var=var):
            events = dl.data.get(region_name)
            lead_a_var = ak.to_numpy( getattr(events['lead_a_p4'], var), allow_missing=False)
            lead_b_var = ak.to_numpy( getattr(events['lead_b_p4'], var), allow_missing=False)
            lead_parts_var = np.concatenate([lead_a_var, lead_b_var])
            return lead_parts_var
        if var == 'pt':
            x_label = 'Lead Part pT [GeV]'
            title = 'Control Plot: Lead Part pT'
            bin_edges = np.linspace(0, 50, 101)
        elif var == 'theta':
            x_label = 'Lead Part Theta [rad]'
            title = 'Control Plot: Lead Part Theta'
            bin_edges = np.linspace(0, np.pi, 101)
        elif var == 'phi':
            x_label = 'Lead Part Phi [rad]'
            title = 'Control Plot: Lead Part Phi'
            bin_edges = np.linspace(-np.pi, np.pi, 101)
        elif var == 'E':
            x_label = 'Lead Part E [GeV]'
            title = 'Control Plot: Lead Part E'
            bin_edges = np.linspace(0, 50, 101)

        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_lead_part_var,
            bin_edges=bin_edges,
            x_label=x_label,
            title=title,
            luminosity=luminosity, normalize=normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_plot_lead_parts_{var}.png")

    # four momentum of lead part pair
    for var in ['pt', 'theta', 'phi', 'E', 'M']:
        def get_lead_part_pair_var(dl, var=var):
            events = dl.data.get(region_name)
            lead_pair_p4 = events['lead_a_p4'] + events['lead_b_p4']
            lead_pair_var = ak.to_numpy( getattr(lead_pair_p4, var), allow_missing=False)
            return lead_pair_var
        if var == 'pt':
            x_label = 'Lead Part Pair pT [GeV]'
            title = 'Control Plot: Lead Part Pair pT'
            bin_edges = np.linspace(0, 100, 101)
        elif var == 'theta':
            x_label = 'Lead Part Pair Theta [rad]'
            title = 'Control Plot: Lead Part Pair Theta'
            bin_edges = np.linspace(0, np.pi, 101)
        elif var == 'phi':
            x_label = 'Lead Part Pair Phi [rad]'
            title = 'Control Plot: Lead Part Pair Phi'
            bin_edges = np.linspace(-np.pi, np.pi, 101)
        elif var == 'E':
            x_label = 'Lead Part Pair E [GeV]'
            title = 'Control Plot: Lead Part Pair E'
            bin_edges = np.linspace(0, 100, 101)
        elif var == 'M':
            x_label = 'Lead Part Pair Invariant Mass [GeV]'
            title = 'Control Plot: Lead Part Pair Invariant Mass'
            bin_edges = np.linspace(0, 100, 101)

        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            func_get_variable=get_lead_part_pair_var,
            bin_edges=bin_edges,
            x_label=x_label,
            title=title,
            luminosity=luminosity, normalize=normalize,
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_plot_lead_part_pair_{var}.png")


    # dR between lead parts in two hemispheres
    def get_lead_part_pair_dR(dl):
        events = dl.data.get(region_name)
        lead_a_p4 = events['lead_a_p4']
        lead_b_p4 = events['lead_b_p4']
        dR = lead_a_p4.deltaR(lead_b_p4)
        dR = ak.to_numpy(dR, allow_missing=False)
        return dR
    bin_edges = np.linspace(160 / 180 * np.pi, np.pi, 101)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_lead_part_pair_dR,
        bin_edges=bin_edges,
        x_label='dR between Lead Parts',
        title='Control Plot: dR between Lead Parts',
        luminosity=luminosity, normalize=normalize,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_part_pair_dR.png")





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