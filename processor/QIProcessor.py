import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import load_events_from_parquet, print_and_write_to_opened_file, get_event_category_from_signal_name
from utils.plotter import do_control_plot
from quantum.observables_builder import get_observable_names, derive_results, shift_SDM_element
import quantum.unfold as unfold
import ROOT


def plot_quantum_observables(dl_dict, output_dir, region_name="hadhad", log_scale=False, blind=True):
    os.makedirs(output_dir, exist_ok=True)
    observables = get_observable_names()
    for obs in observables:
        assert all([obs in dl_dict[dl_name].data[region_name].fields for dl_name in dl_dict.keys()]), f"Observable {obs} not found in all datasets for region {region_name}"
        def get_obs(events):
            if len(events) == 0:
                return np.array([]), np.array([])
            obs_values = ak.to_numpy(events[obs], allow_missing=False)
            flag_valid = events['flags_valid'] > 0
            obs_values = obs_values[flag_valid]
            weights = events['weight'][flag_valid]
            return obs_values, weights
        bin_edges = np.linspace(-1, 1, 11)
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
        self.dict_region_to_signals = config.get('dict_region_to_signals', {})
        self.verbosity = config.get('verbosity', 0)
        self.asimov_data = config.get('asimov_data', True)
        
        # under development: unfolding results
        default_output_dir = config['default_output_dir']
        self.path_response_matrices = f"{default_output_dir}/response_matrices/"
        self.response_matrix = {f"{region}_{signal_name}": {} for region in self.dict_region_to_signals.keys() for signal_name in self.dict_region_to_signals.get(region, [])}
        self.num_bins = 10
        self.bin_edges = np.linspace(-1, 1, self.num_bins + 1)

        self.initialize()

    def initialize(self):
        self.raw_ztautau_events, _ = DataLoader.DataLoader.load_processed_data(self.config['processed_data_dir'], "Ztautau", region_name='raw')
        self.events_truth_region = self.raw_ztautau_events[(self.raw_ztautau_events['truth_QI_region'] == 1)]
        self.load_response_matrix()

    def get_binned_observable(self, var, events):
        var_values = ak.to_numpy(events[var], allow_missing=False)
        binned_var = binning_variable(var_values, self.bin_edges)
        return binned_var.astype(float)

    def build_response_matrix(self, region):
        # build response matrix using raw Ztautau events
        raw_events = self.raw_ztautau_events

        for signal_name in self.dict_region_to_signals.get(region, []):
            # mask for truth region and analysis region
            mask_truth_region = raw_events['truth_QI_region'] == 1
            mask_analysis_region = (raw_events[f'{region}_cut'] == 1) & (raw_events['flags_valid'] > 0) & (raw_events['theta_cm'] > 0.6) & (raw_events['mtautau'] > 80)
            event_category = get_event_category_from_signal_name(signal_name)
            mask_target_signal = raw_events['event_category'] == event_category

            # only use events that pass selection
            mask_unfolding_events = mask_target_signal & (mask_truth_region | mask_analysis_region)
            events = raw_events[mask_unfolding_events]
            mask_truth_region = mask_truth_region[mask_unfolding_events]
            mask_analysis_region = mask_analysis_region[mask_unfolding_events]

            for var in get_observable_names():
                print(f"Building response matrix for {var}...")
                binned_var_recon = self.get_binned_observable(var, events).astype(float)
                binned_var_truth = self.get_binned_observable(f'truth_{var}', events).astype(float)
                weight = ak.to_numpy(events['weight_nominal'], allow_missing=False)

                # set truth (recon) observable to np.nan if the event is outside of the truth (analysis) region
                binned_var_truth[~mask_truth_region] = np.nan
                binned_var_recon[~mask_analysis_region] = np.nan

                # build response matrix
                self.response_matrix[f"{region}_{signal_name}"][var] = unfold.build_response(binned_var_recon, binned_var_truth, num_bins=self.num_bins, weight=weight, name=f"{region}_{signal_name}_{var}")
            
        fout = ROOT.TFile(f"{self.path_response_matrices}/response_{region}.root", "RECREATE")
        for signal_name in self.dict_region_to_signals.get(region, []):
            for var in get_observable_names():
                self.response_matrix[f"{region}_{signal_name}"][var].Write()
        fout.Close()


    def load_response_matrix(self):
        for region in self.dict_region_to_signals.keys():
            signal_names = self.dict_region_to_signals.get(region, [])
            loaded = True
            file_path = f"{self.path_response_matrices}/response_{region}.root"
            if not os.path.exists(file_path):
                os.makedirs(self.path_response_matrices, exist_ok=True)
                print(f"Response matrix file for {region} not found at {file_path}. Rebuilding all response matrices...")
                loaded = False
            else:
                fin = ROOT.TFile(file_path, "READ")
                for signal_name in signal_names:
                    for var in get_observable_names():
                        if not fin.GetListOfKeys().Contains(f"{region}_{signal_name}_{var}"):
                            print(f"Response matrix for {signal_name} and {var} not found in file. Rebuilding all response matrices...")
                            loaded = False
                            break
                        self.response_matrix[f"{region}_{signal_name}"][var] = fin.Get(f"{region}_{signal_name}_{var}")
                    if not loaded: break
                fin.Close()
            if not loaded:
                self.build_response_matrix(region)


    def run(self, dl_dict):
        f_out = open(f"{self.output_dir}/results.txt", 'w')
        # for region in self.regions:
        for region in self.dict_region_to_signals.keys():
            print_and_write_to_opened_file(f"\n\nRegion: {region}", f_out)
            print_and_write_to_opened_file("    Valid Fraction (unweighted):", f_out)
            for dl_name, dl in dl_dict.items():
                events = dl.data[region]
                if len(events) == 0:
                    continue
                valid_fraction = ak.sum(events['flags_valid'] > 0) / len(events)
                print_and_write_to_opened_file(f"        {dl_name}: {valid_fraction:.4f}", f_out)

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
            for signal_name in self.dict_region_to_signals.get(region, []):
                print_and_write_to_opened_file(f"\n\nUnfolding results for signal {signal_name} in region {region}:", f_out)
                output_dir_unfold = f"{output_dir}/unfolding/{signal_name}/"
                os.makedirs(output_dir_unfold, exist_ok=True)

                event_category = get_event_category_from_signal_name(signal_name)

                signal_events, background_events, data_events = [], [], []
                for dl_name, dl in dl_dict.items():
                    events = dl.data[region]
                    events = events[events['flags_valid'] > 0]  # only unfold valid events
                    events = events[events['theta_cm'] > 0.6] 
                    events = events[events['mtautau'] > 80]
                    # events = events[events['event_category'] == event_category] # only unfold events in the target signal category
                    if len(events) == 0:
                        print(f"No valid events to unfold for {dl_name} in region {region}. Skipping...")
                        continue

                    is_signal = dl_name == signal_name
                    is_mc = not dl.is_data
                    # Categorize MC samples into signal or background
                    if is_signal:
                        signal_events.append(events)
                    elif is_mc:
                        background_events.append(events)
                    # Build data container:
                    # - real data if not using Asimov data
                    # - all MC components if using Asimov data
                    if (dl.is_data and not self.asimov_data) or (self.asimov_data and is_mc):
                        data_events.append(events)

                # weight = ak.to_numpy(events_to_unfold['weight'], allow_missing=False)
                weight_data = np.concatenate([ak.to_numpy(events['weight'], allow_missing=False) for events in data_events])
                weight_bkg = np.concatenate([ak.to_numpy(events['weight'], allow_missing=False) for events in background_events])
                weight_signal = np.concatenate([ak.to_numpy(events['weight'], allow_missing=False) for events in signal_events])

                truth_events = self.events_truth_region[self.events_truth_region['event_category'] == event_category] 
                truth_weight = ak.to_numpy(truth_events['weight'], allow_missing=False)
                # shift SDM if the recon events are shifted
                if dl_dict[signal_name].current_variation[0] != 'nominal':
                    element_name, variation = dl_dict[signal_name].current_variation
                    truth_weight = shift_SDM_element(truth_events, element_name=element_name, variation=variation)

                # unfold the target variables
                unfold_histograms = {}
                truth_histograms = {}
                for var in get_observable_names():
                    print(f"Unfolding {var}...")

                    # unfold the variable
                    # get binned_vars and weights for both data and background to be unfolded
                    binned_var_data = np.concatenate([self.get_binned_observable(var, events) for events in data_events])
                    h_measure_data = unfold.build_TH1D(f"h_{var}_data", binned_var_data, num_bins=self.num_bins, weight=weight_data)

                    binned_var_bkg = np.concatenate([self.get_binned_observable(var, events) for events in background_events])
                    h_measure_bkg = unfold.build_TH1D(f"h_{var}_bkg", binned_var_bkg, num_bins=self.num_bins, weight=weight_bkg)

                    # set bin error to sqrt of bin content to mimic Poisson uncertainty for asimov data
                    if self.asimov_data:
                        for i in range(1, h_measure_data.GetNbinsX() + 1):
                            content = h_measure_data.GetBinContent(i)
                            h_measure_data.SetBinContent(i, content)
                            error = np.sqrt(content)
                            h_measure_data.SetBinError(i, error)
                    h_measure = h_measure_data.Clone(f"h_{var}_measure")
                    h_measure.Add(h_measure_bkg, -1.0)

                    # # uncomment the following lines to only unfold the signal component (for testing purpose)
                    # binned_var_signal = np.concatenate([self.get_binned_observable(var, events) for events in signal_events])
                    # h_measure_signal = unfold.build_TH1D(f"h_{var}_signal", binned_var_signal, num_bins=self.num_bins, weight=weight_signal)
                    # h_measure = h_measure_signal

                    unfold_result = ROOT.RooUnfoldSvd(self.response_matrix[f"{region}_{signal_name}"][var], h_measure, 5).Hunfold(2)

                    # build truth distribution using truth region events for comparison
                    var_truth_binned = self.get_binned_observable(f'truth_{var}', truth_events)
                    h_truth = unfold.build_TH1D(f"h_{var}_truth", var_truth_binned, num_bins=self.num_bins, weight=truth_weight)

                    # plot the results
                    unfold.plot_unfolded_results(unfold_result, save_path=f"{output_dir_unfold}/{var}_unfold.pdf", h_truth=h_truth, h_reco=h_measure, var_name=var)

                    unfold_histograms[var] = unfold.build_Hist_from_TH1D(unfold_result, bin_edges=self.bin_edges)
                    truth_histograms[var] = unfold.build_Hist_from_TH1D(h_truth, bin_edges=self.bin_edges)

                # derive quantum results using unfolded histograms
                analyzing_power_a = truth_events['analyzing_power_a'][0]*(-1)
                analyzing_power_b = truth_events['analyzing_power_b'][0]
                unfolded_BC_matrices, unfolded_quantum_results = derive_results(unfold_histograms, analyzing_power_a=analyzing_power_a, analyzing_power_b=analyzing_power_b)
                truth_BC_matrices, truth_quantum_results = derive_results(truth_histograms, analyzing_power_a=analyzing_power_a, analyzing_power_b=analyzing_power_b)
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