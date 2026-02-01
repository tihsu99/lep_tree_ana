import numpy as np
from BaseProcessor import BaseProcessor
import utils.plotter as plotter
from utils.plotter import do_control_plot_from_hists
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator
import copy
import logging

log = logging.getLogger(__name__)

def parser_muid_wp(ary_muid_wp):
    """Parse the MUID working point array via bitwise operations."""
    parsed_ary = copy.deepcopy(ary_muid_wp)
    for i in range(4):
        mask = ary_muid_wp & (1 << i)
        parsed_ary = ak.where(mask != 0, i + 1, parsed_ary)
    return parsed_ary

class IDAnaProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        super().__init__(config)
        self.config = config
        output_dir_name = config.get('output_dir_name', 'IDAnalysis')
        self.output_dir = f"{output_dir}/{output_dir_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.luminosity = config.get('luminosity', 46.3)  # in pb^-1
        self.targeted_recon_pdgId = config.get('targeted_recon_pdgId', [41])  
        self.regions = config.get('regions', ['raw'])

        self.id_of_interest = [
            'Elid_tag',
            'Muid_tag',
            'Haid_pionRich',
            "Haidn_pionTag", 
            "Haidr_pionTag",
            "Haide_pionTag",
            "Haidc_pionTag"
        ]

    def run(self, dl_dict):
        bin_edges = np.linspace(-1.5, 4.5, 7)
        for region in self.regions:
            output_dir_region = f"{self.output_dir}/{region}"
            os.makedirs(output_dir_region, exist_ok=True)
            for pdgId in self.targeted_recon_pdgId:
                for id_name in self.id_of_interest:
                    log.info(f"Processing ID: {id_name} for reconstructed pdgId: {pdgId} in region: {region}")
                    hist_data = np.zeros(len(bin_edges) - 1)
                    hists_MC = {}
                    hists_MC_err2 = {}

                    for dl_name, dl in dl_dict.items():
                        events = dl.data.get(region)
                        mask_recon_id = np.abs(events['Part_pdgId']) == pdgId
                        id_wp = events[id_name][mask_recon_id]
                        if id_name.startswith('Muid'):
                            id_values = parser_muid_wp(id_wp)
                        else:
                            id_values = id_wp
                        # convert to numpy array
                        id_values = ak.to_numpy(ak.flatten(id_values))
                        hist, _ = np.histogram(id_values, bins=bin_edges)

                        if dl.is_data:
                            hist_data += hist
                        else:
                            scale = dl.norm_factor / dl.initial_total_num_events * self.luminosity
                            if dl_name not in hists_MC:
                                hists_MC[dl_name] = np.zeros(len(bin_edges) - 1)
                                hists_MC_err2[dl_name] = np.zeros(len(bin_edges) - 1)
                            hists_MC[dl_name] += hist * scale
                            hists_MC_err2[dl_name] += hist * scale  # Poisson errors: variance = count

                    fig, ax, ax_ratio = plotter.do_control_plot_from_hists(
                        hist_data=hist_data,
                        hist_MC_dict=hists_MC,
                        hist_MC_err2_dict=hists_MC_err2,
                        bin_edges=bin_edges,
                        x_label=id_name,
                        title=f"Reconstructed PdgId {pdgId} - {id_name} WP Distribution",
                        normalize=False
                    )
                    plt.tight_layout()
                    fig.savefig(f"{output_dir_region}/ReconPdgid_{pdgId}_{id_name}_WP_Distribution.png")





    def finalize(self):
        pass