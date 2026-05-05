import numpy as np
import DataLoader
import matplotlib.pyplot as plt
import os
import awkward as ak
from utils.common_functions import get_event_category_from_signal_name
from quantum.observables_builder import get_observable_names
import quantum.unfold as unfold
import ROOT


class ResponseMatricesManager:
    def __init__(self, data_dir, output_dir, dict_region_to_signals):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.path_response_matrices = f"{self.output_dir}/response_matrices/"
        self.dict_region_to_signals = dict_region_to_signals

        self.num_bins = unfold.get_num_bins()
        self.bin_edges = unfold.get_bin_edges()
        self.unfold_vars = [obs for obs in get_observable_names() if 'cos' in obs]
        self.response_matrices = {}
        self.initialize()


    def get_response_matrix(self, region, signal_name, var):
        key = f"{region}_{signal_name}"
        if key not in self.response_matrices:
            raise ValueError(f"Response matrix for {key} not found.")
        if var not in self.response_matrices[key]:
            raise ValueError(f"Variable {var} not found in response matrix for {key}.")
        return self.response_matrices[key][var]


    def initialize(self):
        self.raw_ztautau_events = None
        for region in self.dict_region_to_signals.keys():
            for signal_name in self.dict_region_to_signals.get(region, []):
                self.response_matrices[f"{region}_{signal_name}"] = {var: None for var in self.unfold_vars}
        self.load_matrices()

        # release raw Ztautau events from memory after building response matrices
        if self.raw_ztautau_events is not None:
            del self.raw_ztautau_events
            self.raw_ztautau_events = None

    
    def load_matrices(self):
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
                    for var in self.unfold_vars:
                        if not fin.GetListOfKeys().Contains(f"{region}_{signal_name}_{var}"):
                            print(f"Response matrix for {signal_name} and {var} not found in file. Rebuilding all response matrices...")
                            loaded = False
                            break
                        self.response_matrices[f"{region}_{signal_name}"][var] = fin.Get(f"{region}_{signal_name}_{var}")
                    if not loaded: break
                fin.Close()
            if not loaded:
                self.build_response_matrices(region)

    def build_response_matrices(self, region):
        # build response matrix using raw Ztautau events
        if self.raw_ztautau_events is None:
            self.raw_ztautau_events, _ = DataLoader.DataLoader.load_processed_data(self.data_dir, "Ztautau", "raw")
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

            for var in self.unfold_vars:
                print(f"Building response matrix for {var}...")
                binned_var_recon = self.get_binned_observable(var, events)
                binned_var_truth = self.get_binned_observable(f'truth_{var}', events)
                weight = ak.to_numpy(events['weight_nominal'], allow_missing=False)

                # set truth (recon) observable to np.nan if the event is outside of the truth (analysis) region
                binned_var_truth[~mask_truth_region] = np.nan
                binned_var_recon[~mask_analysis_region] = np.nan

                # build response matrix
                self.response_matrices[f"{region}_{signal_name}"][var] = unfold.build_response(binned_var_recon, binned_var_truth, num_bins=self.num_bins, weight=weight, name=f"{region}_{signal_name}_{var}")
            
        fout = ROOT.TFile(f"{self.path_response_matrices}/response_{region}.root", "RECREATE")
        for signal_name in self.dict_region_to_signals.get(region, []):
            for var in self.unfold_vars:
                self.response_matrices[f"{region}_{signal_name}"][var].Write()
        fout.Close()


    def get_binned_observable(self, var, events):
        var_values = ak.to_numpy(events[var], allow_missing=False)
        binned_var = unfold.bin_variable(var_values, self.bin_edges)
        return binned_var.astype(float)

