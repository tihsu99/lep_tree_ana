import numpy as np
import DataLoader
import matplotlib.pyplot as plt
import os
import awkward as ak
from utils.tau_decay import get_event_category_from_signal_name, NOMINAL_BC_VALUES, get_analyzing_powers_from_event_category
from quantum.observables_builder import get_observable_names, get_bc_name_from_variable_name, get_theoretical_distribution
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
        self.raw_ztautau_events = None
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
            self.raw_ztautau_events, _ = DataLoader.DataLoader.load_processed_data(self.data_dir, "Ztautau", "raw", is_trainset=True)
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
                weight_sf = ak.to_numpy(events[f'{var}_reweight_sf'], allow_missing=False)
                weight = weight * weight_sf

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


    def closure_test(self, region, signal_name, var):
        output_dir = f"{self.output_dir}/closure_test/"
        os.makedirs(output_dir, exist_ok=True)
        signal_category = get_event_category_from_signal_name(signal_name)
        response_matrix = self.get_response_matrix(region, signal_name, var)
        if self.raw_ztautau_events is None:
            self.raw_ztautau_events, _ = DataLoader.DataLoader.load_processed_data(self.data_dir, "Ztautau", "raw", is_trainset=False)
        raw_events = self.raw_ztautau_events
        mask_target_signal = raw_events['event_category'] == signal_category
        raw_events = raw_events[mask_target_signal]
        

        mask_truth_region = raw_events['truth_QI_region'] == 1
        mask_analysis_region = (raw_events[f'{region}_cut'] == 1) & (raw_events['flags_valid'] > 0) & (raw_events['theta_cm'] > 0.6) & (raw_events['mtautau'] > 80)
        # mask_analysis_fake_events = mask_analysis_region & (~mask_truth_region)

        # events = raw_events[mask_target_signal & (mask_truth_region | mask_analysis_region)]
        events = raw_events
        weight = ak.to_numpy(events['weight_nominal'], allow_missing=False)
        weight_sf = ak.to_numpy(events[f'{var}_reweight_sf'], allow_missing=False)
        weight = weight * weight_sf


        # build h_truth_theo
        h_theo, _ = get_theoretical_distribution(var, signal_category, norm=ak.sum(weight[mask_truth_region]), bin_edges=self.bin_edges)
        h_theo = unfold.build_TH1D(f"h_theo", np.arange(self.num_bins), num_bins=self.num_bins, weight=h_theo)
        for i in range(h_theo.GetNbinsX()):
            h_theo.SetBinError(i+1, 0)

        # build truth distribution using truth region events
        var_truth_binned = self.get_binned_observable(f'truth_{var}', events[mask_truth_region])
        h_truth = unfold.build_TH1D(f"h_truth", var_truth_binned, num_bins=self.num_bins, weight=weight[mask_truth_region])

        # build reco distribution using analysis region events and response matrix
        var_recon_binned = self.get_binned_observable(var, events[mask_analysis_region])
        h_recon = unfold.build_TH1D(f"h_recon", var_recon_binned, num_bins=self.num_bins, weight=weight[mask_analysis_region])

        # var_recon_fake_binned = self.get_binned_observable(var, events[mask_analysis_fake_events])
        # h_recon_fake = unfold.build_TH1D(f"h_recon_fake", var_recon_fake_binned, num_bins=self.num_bins, weight=weight[mask_analysis_fake_events])

        # unfold reco distribution and compare with truth distribution
        # h_to_unfold = h_recon.Clone("h_to_unfold")
        # h_to_unfold.Add(h_recon_fake, -1)
        # unfold_result = ROOT.RooUnfoldSvd(response_matrix, h_to_unfold, 5).Hunfold(2)
        # unfold_result = ROOT.RooUnfoldInvert(response_matrix, h_to_unfold).Hunfold(2)
        # unfold_result = ROOT.RooUnfoldTUnfold(response_matrix, h_recon).Hunfold(2)
        unfold_result = ROOT.RooUnfoldBayes(response_matrix, h_recon, niter=4, handleFakes=True).Hunfold(2)
        unfold_result.SetTitle("h_unfolded")

        # forward fold the truth distribution and compare with reco distribution
        h_truth_forward_folded = response_matrix.ApplyToTruth(h_truth)
        h_reco_fake = response_matrix.Hfakes()
        h_truth_forward_folded.Add(h_reco_fake)
        h_truth_forward_folded.SetTitle("folded truth")

        # forward fold the theoretical distribution
        h_theo_forward_folded = response_matrix.ApplyToTruth(h_theo)
        h_theo_forward_folded.Add(h_reco_fake)
        h_theo_forward_folded.SetTitle("folded theoretical")

        # plot
        ROOT.gStyle.SetOptStat(0)
        c = ROOT.TCanvas(f"closure_{region}_{signal_name}_{var}", f"closure_{region}_{signal_name}_{var}", 800, 600)
        h_truth.GetYaxis().SetRangeUser(0, max(h_truth.GetMaximum(), h_recon.GetMaximum(), unfold_result.GetMaximum())*1.2)
        h_truth.GetXaxis().SetTitle(var)

        h_truth.SetLineColor(ROOT.kBlack)
        h_truth.Draw("hist E1")
        h_recon.SetLineColor(ROOT.kRed)
        h_recon.Draw("hist same E1")
        h_theo.SetLineColor(ROOT.kMagenta)
        h_theo.Draw("hist same E1")

        unfold_result.SetLineColor(ROOT.kBlue)
        unfold_result.Draw("hist same E1")
        h_truth_forward_folded.SetLineColor(ROOT.kGreen + 3)
        h_truth_forward_folded.Draw("hist same  E1")
        h_theo_forward_folded.SetLineColor(ROOT.kMagenta)
        h_theo_forward_folded.Draw("hist same E1")
        l = None
        if 'times' in var:
            l = c.BuildLegend(0.7, 0.75, 0.9, 0.9)
        else:
            l = c.BuildLegend()
        l.SetTextSize(0.03)
        c.SaveAs(f"{output_dir}/closure_{region}_{signal_name}_{var}.png")
        

if __name__ == "__main__":
    data_dir = "./20260512-dataset/"
    output_dir = "./20260512-closure-test/"
    dict_region_to_signals = {
        "pipi": ["Ztautau_pipi"],
    }
    response_manager = ResponseMatricesManager(data_dir, output_dir, dict_region_to_signals)
    for obs in response_manager.unfold_vars:
    # for obs in ['cos_theta_A_k']:
        response_manager.closure_test("pipi", "Ztautau_pipi", obs)

