import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme, load_events_from_parquet, print_and_write_to_opened_file
from utils.plotter import do_control_plot
from quantum.observables_builder import get_observable_names, get_mean_and_err_of_mean, derive_results
import quantum.unfold as unfold
import ROOT


def plot_quantum_observables(dl_dict, output_dir, region_name="hadhad", log_scale=False, blind=True):
    os.makedirs(output_dir, exist_ok=True)
    observables = get_observable_names()
    for obs in observables:
        assert all([obs in dl_dict[dl_name].data[region_name].fields for dl_name in dl_dict.keys()]), f"Observable {obs} not found in all datasets for region {region_name}"
        def get_obs(events):
            obs_values = ak.to_numpy(events[obs], allow_missing=False)
            flag_valid = events['flags_valid'] > 0
            obs_values = obs_values[flag_valid]
            weights = events['weight'][flag_valid]
            return obs_values, weights
        bin_edges = np.linspace(-1, 1, 101)
        fig, ax, ax_ratio = do_control_plot(
            dl_dict,
            region_name=region_name,
            func_get_variable=get_obs,
            bin_edges=bin_edges,
            x_label=f'{obs}',
            title=f'Control Plot: {obs}',
            normalize=False,
            log_scale=log_scale,
            blind=blind,
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/control_plot_{obs}.png")


def binning_variable(var, bin_edges):
    # bin the variable according to the provided bin edges
    var = np.asarray(var)
    binned_var = np.digitize(var, bin_edges) - 1  # digitize returns indices starting from 1
    binned_var[var < bin_edges[0]] = -1  # underflow
    binned_var[var >= bin_edges[-1]] = len(bin_edges) - 1  # overflow
    return binned_var



class QIProcessor(BaseProcessor):
    def __init__(self, config, output_dir):
        """
        Processor to make control plots for data/MC comparison.
        """
        super().__init__(config)
        self.config = config
        if 'output_dir_name' in config:
            output_dir = f"{output_dir}/{config['output_dir_name']}"
        else:
            output_dir = f"{output_dir}/QI_results/"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.regions = config.get('regions', ['hadhad'])
        self.particle_level_only = config.get('particle_level_only', False)
        self.verbosity = config.get('verbosity', 0)
        
        # under development: unfolding results
        self.unfolding_target_sample = 'Ztautau_pipi'
        self.response_matrix = {}
        self.num_bins = 20
        self.bin_edges = np.linspace(-1, 1, self.num_bins + 1)

        self.initialize()

    def initialize(self):
        self.raw_ztautau_events = load_events_from_parquet(f"{self.config.get('default_output_dir')}/Ztautau/filtered___raw.parquet")
        self.events_fiducial_region = self.raw_ztautau_events[(self.raw_ztautau_events['truth_QI_region'] == 1) & (self.raw_ztautau_events['event_category'] == 11)]
        self.load_response_matrix()

    def get_binned_observable(self, var, events):
        var_values = ak.to_numpy(events[var], allow_missing=False)
        binned_var = binning_variable(var_values, self.bin_edges)
        return binned_var

    def build_response_matrix(self):
        # build response matrix using raw Ztautau events
        raw_events = self.raw_ztautau_events

        # mask for fiducial region and analysis region
        mask_fiducial_region = raw_events['truth_QI_region'] == 1
        mask_analysis_region = (raw_events['hadhad_cut'] == 1) & (raw_events['flags_valid'] > 0) & (raw_events['theta_cm']*2/np.pi > 0.6)
        mask_target_channel = raw_events['event_category'] == 11

        # only use events that pass selection
        mask_unfolding_events = mask_target_channel & (mask_fiducial_region | mask_analysis_region)
        events = raw_events[mask_unfolding_events]
        mask_fiducial_region = mask_fiducial_region[mask_unfolding_events]
        mask_analysis_region = mask_analysis_region[mask_unfolding_events]

        for var in get_observable_names():
            print(f"Building response matrix for {var}...")
            binned_var_recon = self.get_binned_observable(var, events).astype(float)
            binned_var_truth = self.get_binned_observable(f'truth_{var}', events).astype(float)
            weight = ak.to_numpy(events['weight'], allow_missing=False)

            # set truth (recon) observable to np.nan if the event is outside of the fiducial (analysis) region
            binned_var_truth[~mask_fiducial_region] = np.nan
            binned_var_recon[~mask_analysis_region] = np.nan

            # build response matrix
            self.response_matrix[var] = unfold.build_response(binned_var_recon, binned_var_truth, num_bins=self.num_bins, weight=weight, name=f"response_{var}")
        
        fout = ROOT.TFile(f"{self.output_dir}/response.root", "RECREATE")
        for var in self.response_matrix.keys():
            self.response_matrix[var].Write()
        fout.Close()


    def load_response_matrix(self):
        loaded = True
        for var in get_observable_names():
            loaded = False
            file_path = f"{self.output_dir}/response.root"
            if not os.path.exists(file_path):
                print(f"Response matrix file for {var} not found at {file_path}. Rebuilding all response matrices...")
                break
            fin = ROOT.TFile(file_path, "READ")
            self.response_matrix[var] = fin.Get(f"response_{var}")
            # self.response_matrix[var] = fin.Get(f"response")
            fin.Close()
            loaded = True
        if not loaded:
            self.build_response_matrix()


    def run(self, dl_dict):
        f_out = open(f"{self.output_dir}/results.txt", 'w')
        for region in self.regions:
            f_out.write(f"Region: {region}\n")
            f_out.write("    Valid Fraction (unweighted):\n")
            for dl_name, dl in dl_dict.items():
                events = dl.data[region]
                valid_fraction = ak.sum(events['flags_valid'] > 0) / len(events)
                f_out.write(f"        {dl_name}: {valid_fraction:.4f}\n")

            output_dir = f"{self.output_dir}/{region}/"
            os.makedirs(output_dir, exist_ok=True)

            # plot quantum observables
            if self.verbosity > 0:
                plot_quantum_observables(
                    dl_dict, 
                    f"{output_dir}/plots/",
                    region_name=region,
                    log_scale=False,
                    blind=True
                )


            # unfold (under development)
            output_dir_unfold = f"{output_dir}/unfolding/"
            os.makedirs(output_dir_unfold, exist_ok=True)
            events_to_unfold = dl_dict[self.unfolding_target_sample].data[region]
            events_to_unfold = events_to_unfold[events_to_unfold['flags_valid'] > 0]  # only unfold valid events
            events_to_unfold = events_to_unfold[events_to_unfold['theta_cm']*2/np.pi > 0.6] 

            # unfold the target variables
            unfold_histograms = {}
            truth_histograms = {}
            for var in get_observable_names():
                print(f"Unfolding {var}...")
                var_recon_binned = self.get_binned_observable(var, events_to_unfold)
                weight = ak.to_numpy(events_to_unfold['weight'], allow_missing=False)

                # unfold the variable
                h_measure = unfold.build_TH1D(f"h_{var}_reco", var_recon_binned, num_bins=self.num_bins, weight=weight)
                unfold_result = ROOT.RooUnfoldSvd(self.response_matrix[var], h_measure, 5).Hunfold(2)

                # build truth distribution using fiducial region events for comparison
                var_truth_binned = self.get_binned_observable(f'truth_{var}', self.events_fiducial_region)
                truth_weight = ak.to_numpy(self.events_fiducial_region['weight'], allow_missing=False)
                h_truth = unfold.build_TH1D(f"h_{var}_truth", var_truth_binned, num_bins=self.num_bins, weight=truth_weight)

                # plot the results
                unfold.plot_unfolded_results(unfold_result, save_path=f"{output_dir_unfold}/{var}_unfold.pdf", h_truth=h_truth, h_reco=h_measure, var_name=var)

                unfold_histograms[var] = unfold.build_Hist_from_TH1D(unfold_result, bin_edges=self.bin_edges)
                truth_histograms[var] = unfold.build_Hist_from_TH1D(h_truth, bin_edges=self.bin_edges)

            # derive quantum results using unfolded histograms
            unfolded_BC_matrices, unfolded_quantum_results = derive_results(unfold_histograms, analyzing_power_a=1, analyzing_power_b=-1)
            truth_BC_matrices, truth_quantum_results = derive_results(truth_histograms, analyzing_power_a=1, analyzing_power_b=-1)
            for res_type, results in zip(['Unfolded', 'Truth'], [unfolded_BC_matrices, truth_BC_matrices]):
                print_and_write_to_opened_file(f"\n    {res_type} B and C matrices:", f_out)
                for key, value in results.items():
                    nominal, err_up, err_down = value.value, value.err_up, value.err_down
                    print_and_write_to_opened_file(f"        {key}: {nominal:.4f} +{err_up:.4f}/-{err_down:.4f}", f_out)

            for res_type, results in zip(['Unfolded', 'Truth'], [unfolded_quantum_results, truth_quantum_results]):
                print_and_write_to_opened_file(f"\n    {res_type} Quantum results:", f_out)
                for key, value in results.items():
                    nominal, err_up, err_down = value.value, value.err_up, value.err_down
                    print_and_write_to_opened_file(f"        {key}: {nominal:.4f} +{err_up:.4f}/-{err_down:.4f}", f_out)


        f_out.close()

    def finalize(self):
        pass