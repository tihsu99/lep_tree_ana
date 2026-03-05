import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme
from utils.plotter import do_control_plot

    

def make_control_plots_tautau(dl_dict, luminosity, normalize, output_dir, region_name="tautau", log_scale=True):
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
        log_scale=log_scale,
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
        log_scale=log_scale,
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
        log_scale=log_scale,
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
        log_scale=log_scale,
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
        log_scale=log_scale,
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
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_missing_pt.png")


    # nprong
    def get_nprong(dl):
        events = dl.data.get(region_name)
        nprong = ak.to_numpy(events['nprong'], allow_missing=False)
        return nprong

    bin_edges = np.arange(2, 8, 1)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_nprong,
        bin_edges=bin_edges,
        x_label='nprong',
        title='Control Plot: nprong',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_nprong.png")


    # number of neutral particles 
    def get_n_neutral(dl):
        events = dl.data.get(region_name)
        neutral_mask = events['Part_charge'] == 0
        n_neutral = ak.to_numpy(ak.sum(neutral_mask, axis=-1), allow_missing=False)
        return n_neutral
    bin_edges = np.linspace(0, 10, 11)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_n_neutral,
        bin_edges=bin_edges,
        x_label='Number of Neutral Particles',
        title='Control Plot: Number of Neutral Particles',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_n_neutral.png")
    
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
        log_scale=log_scale,
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
            log_scale=log_scale,
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
            log_scale=log_scale,
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
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_lead_part_pair_dR.png")



def make_control_plots_pion(dl_dict, luminosity, normalize, output_dir, region_name="pion", log_scale=True):

    make_control_plots_tautau(dl_dict, luminosity, normalize, output_dir, region_name=region_name, log_scale=log_scale)

    # plot pdg id for charged particles
    # for better visualization, first map others, pi, el, mu to 0, 1, 2, 3, then plot histogram with x-ticks showing the mapping
    map_pdgId = {
        41: 1,  # pi
        2: 2,   # el
        6: 3,   # mu
    }
    def get_charged_pdgId(dl):
        events = dl.data.get(region_name)
        charged_mask = events['Part_charge'] != 0
        charged_pdgId = ak.to_numpy(events['Part_pdgId'][charged_mask], allow_missing=False)
        charged_pdgId = np.abs(charged_pdgId)  
        # map pdgId for better visualization
        mask_others = np.ones_like(charged_pdgId, dtype=bool)
        for pdgId in map_pdgId.keys():
            mask_others = mask_others & (charged_pdgId != pdgId)
            charged_pdgId[charged_pdgId == pdgId] = map_pdgId[pdgId]
        charged_pdgId[mask_others] = 0
        return charged_pdgId

    bin_edges = np.linspace(-0.5, 3.5, 5)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_charged_pdgId,
        bin_edges=bin_edges,
        x_label='Charged Particle PDG ID',
        title='Control Plot: Charged Particle PDG ID',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Other', 'Pi', 'El', 'Mu'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_charged_pdgId.png")

    # number of photons in the event
    def get_n_photons(dl):
        events = dl.data.get(region_name)
        photon_mask = np.abs(events['Part_pdgId']) == 21
        n_photons = ak.to_numpy(ak.sum(photon_mask, axis=-1), allow_missing=False)
        return n_photons
    
    bin_edges = np.linspace(0, 10, 11)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_n_photons,
        bin_edges=bin_edges,
        x_label='Number of Photons',
        title='Control Plot: Number of Photons',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_n_photons.png")


    # number of pions in the event
    def get_n_pions(dl):
        events = dl.data.get(region_name)
        pion_mask = np.abs(events['Part_pdgId']) == 41
        n_pions = ak.to_numpy(ak.sum(pion_mask, axis=-1), allow_missing=False)
        return n_pions
    bin_edges = np.linspace(0, 8, 9)
    fig, ax, ax_ratio = do_control_plot(
        dl_dict,
        func_get_variable=get_n_pions,
        bin_edges=bin_edges,
        x_label='Number of Pions',
        title='Control Plot: Number of Pions',
        luminosity=luminosity, normalize=normalize,
        log_scale=log_scale,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_plot_n_pions.png")



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

        if 'pion' in self.regions:
            output_dir_pion = f"{self.output_dir}/pion/"
            os.makedirs(output_dir_pion, exist_ok=True)
            make_control_plots_pion(
                dl_dict,
                luminosity=self.luminosity,
                normalize=self.normalize,
                output_dir=output_dir_pion,
                region_name="pion",
                log_scale=False,
            )
        
        if 'pipi' in self.regions:
            output_dir_pipi = f"{self.output_dir}/pipi/"
            os.makedirs(output_dir_pipi, exist_ok=True)
            make_control_plots_pion(
                dl_dict,
                luminosity=self.luminosity,
                normalize=self.normalize,
                output_dir=output_dir_pipi,
                region_name="pipi",
                log_scale=True,
            )



    def finalize(self):
        pass