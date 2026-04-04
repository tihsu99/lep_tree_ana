import numpy as np
from BaseProcessor import BaseProcessor
import DataLoader
import matplotlib.pyplot as plt
import os
import vector
import awkward as ak
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events, get_all_p4_from_ak_events, cme
from utils.plotter import do_control_plot
from quantum.observables_builder import get_observable_names, get_mean_and_err_of_mean
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
        
        # under development: unfolding results
        self.unfolding_target_sample = 'Ztautau_pipi'
        self.response_matrix = {}
        self.num_bins = 20

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
            if len(self.response_matrix) == 0:
                for var in get_observable_names():
                    print(f"Building response matrix for {var}...")
                    var_recon = ak.to_numpy(events_to_unfold[var], allow_missing=False)
                    var_truth = ak.to_numpy(events_to_unfold[f'truth_{var}'], allow_missing=False)
                    weight = ak.to_numpy(events_to_unfold['weight'], allow_missing=False)
                    binned_var_recon = binning_variable(var_recon, np.linspace(-1, 1, self.num_bins + 1))
                    binned_var_truth = binning_variable(var_truth, np.linspace(-1, 1, self.num_bins + 1))
                    self.response_matrix[var] = unfold.build_response(binned_var_recon, binned_var_truth, num_bins=self.num_bins, weight=weight)
                    events_to_unfold[f'{var}_binned'] = binned_var_recon
                    events_to_unfold[f'truth_{var}_binned'] = binned_var_truth

            # unfold the target variables
            for var in get_observable_names():
                print(f"Unfolding {var}...")
                var_recon_binned = ak.to_numpy(events_to_unfold[f'{var}_binned'], allow_missing=False)
                var_truth_binned = ak.to_numpy(events_to_unfold[f'truth_{var}_binned'], allow_missing=False)
                weight = ak.to_numpy(events_to_unfold['weight'], allow_missing=False)

                # unfold the variable
                h_measure = unfold.build_TH1D(f"h_{var}_reco", var_recon_binned, num_bins=self.num_bins, weight=weight)
                unfold_result = ROOT.RooUnfoldSvd(self.response_matrix[var], h_measure, 5).Hunfold(2)
                h_truth = unfold.build_TH1D(f"h_{var}_truth", var_truth_binned, num_bins=self.num_bins, weight=weight)

                # plot the results
                unfold.plot_unfolded_results(unfold_result, save_path=f"{output_dir_unfold}/{var}_unfold.pdf", h_truth=h_truth, h_reco=h_measure, var_name=var)

        f_out.close()

    def finalize(self):
        pass