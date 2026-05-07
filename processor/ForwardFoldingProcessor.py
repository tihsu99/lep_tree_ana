import os
from collections import OrderedDict

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from BaseProcessor import BaseProcessor
from scipy.optimize import minimize

import DataLoader
import quantum.unfold as unfold
from processor.ResponseMatricesManager import ResponseMatricesManager
import quantum.observables_builder as ob
from utils import common_functions as cf


class ForwardFoldingProcessor(BaseProcessor):
    """
    Fit B and C matrix elements by forward folding truth-level templates into
    signal-region observable distributions.
    """

    def __init__(self, config, output_dir):
        super().__init__(config)
        self.config = config
        if "output_dir_name" in config:
            output_dir = f"{output_dir}/{config['output_dir_name']}"
        else:
            output_dir = f"{output_dir}/forward_folding/"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.dict_region_to_signals = config.get("dict_region_to_signals", {})
        self.verbosity = config.get("verbosity", 0)
        self.asimov_data = config.get("asimov_data", True)
        self.fit_parameter_bounds = config.get("fit_parameter_bounds", (-2.0, 2.0))

        self.num_bins = unfold.get_num_bins()
        self.bin_edges = unfold.get_bin_edges()
        self.bin_centers = np.array(0.5 * (self.bin_edges[:-1] + self.bin_edges[1:]))
        self.response_manager = ResponseMatricesManager(
            self.config["processed_data_dir"],
            self.config["default_output_dir"],
            self.dict_region_to_signals,
        )
        self.unfold_vars = [
            obs for obs in self.response_manager.unfold_vars
            if ob.get_bc_name_from_variable_name(obs) is not None
        ]

        # Total Ztautau signal yield in truth region
        self.expected_yields_truth_region = 32177.19
        
        # branching ratio: nonTau, pi, rho, e, mu, others
        self.branching_ratios = [
            0,
            0.1077,
            0.2537,
            0.1773,
            0.1731,
            0
        ]

    def get_branching_ratio_from_event_category(self, event_category):
        pos_id = event_category // 10
        neg_id = event_category % 10
        assert 0 <= pos_id < len(self.branching_ratios) and 0 <= neg_id < len(self.branching_ratios), f"Invalid event category {event_category}"
        return self.branching_ratios[pos_id] * self.branching_ratios[neg_id]

    def get_binned_observable(self, var, events):
        var_values = ak.to_numpy(events[var], allow_missing=False)
        binned_var = unfold.bin_variable(var_values, self.bin_edges)
        return binned_var.astype(float)


    def build_truth_hist_Bi(self, analyzing_power, parameter_value, total_bin_contents=1.0):
        """Build truth-level Bi template."""
        slope = analyzing_power * parameter_value
        bin_contents = 0.5 * (1 + slope * self.bin_centers) 
        bin_contents = bin_contents * total_bin_contents / np.sum(bin_contents) if np.sum(bin_contents) > 0 else bin_contents
        return unfold.Hist(bin_edges=self.bin_edges, values=bin_contents, errors=np.zeros_like(bin_contents))

    def build_truth_hist_Cij(self, analyzing_power_product, parameter_value, total_bin_contents=1.0):
        """Build truth-level Cij template."""
        slope = analyzing_power_product * parameter_value
        bin_contents = -0.5 * (1 + slope * self.bin_centers) * np.log(np.abs(self.bin_centers))
        bin_contents = bin_contents * total_bin_contents / np.sum(bin_contents) if np.sum(bin_contents) > 0 else bin_contents
        return unfold.Hist(bin_edges=self.bin_edges, values=bin_contents, errors=np.zeros_like(bin_contents))

    def fold_truth_hist(self, response_matrix, truth_hist, name):
        folded = response_matrix.ApplyToTruth(truth_hist, name)
        folded.SetDirectory(0)
        return folded

    def build_expected_signal_hist(self, region, signal_names, branching_ratios, analyzing_powers, var, parameter_value, truth_hist_build_func):
        """Build expected signal histogram by folding truth templates through response matrices.
        
        Each signal gets its own truth template based on its analyzing power and branching ratio.
        
        Args:
            region: Signal region identifier
            signal_names: List of signal component names (e.g., ['Ztautau_pipi', 'Ztautau_rhorho', ...])
            analyzing_powers: List of analyzing powers per signal (per signal event category)
            var: Observable variable name
            parameter_value: Fit parameter value
            truth_hist_build_func: Function to build truth histograms (Bi or Cij)
            
        Returns:
            ROOT.TH1D histogram of expected signal in reco space (sum of all signal contributions)
        """
        expected = None
        
        for idx, signal_name in enumerate(signal_names):
            branching_ratio = branching_ratios[idx] 
            
            # Build truth template specific to this signal's analyzing power and branching ratio
            truth_hist = truth_hist_build_func(analyzing_powers[idx], parameter_value, total_bin_contents=self.expected_yields_truth_region * branching_ratio)
            
            truth_hist = unfold.build_TH1D_from_Hist(
                truth_hist,
                f"h_truth_{region}_{signal_name}_{var}_{parameter_value:.5f}"
            )
            truth_hist.SetDirectory(0)
            
            # Fold through this signal's response matrix
            response_matrix = self.response_manager.get_response_matrix(region, signal_name, var)
            folded = self.fold_truth_hist(
                response_matrix,
                truth_hist,
                f"h_ff_{region}_{signal_name}_{var}_{parameter_value:.5f}",
            )
            if expected is None:
                expected = folded.Clone(f"h_expected_signal_{region}_{var}_{parameter_value:.5f}")
                expected.SetDirectory(0)
            else:
                expected.Add(folded)
        
        # Return empty histogram if no signals (defensive)
        if expected is None:
            expected = unfold.build_TH1D(f"h_expected_signal_empty_{region}_{var}", np.zeros(self.num_bins), self.num_bins)
            expected.SetDirectory(0)
        
        return expected

    def build_reco_hist(self, name, events_list, var):
        if len(events_list) == 0:
            return unfold.build_TH1D(name, [], self.num_bins)
        binned_var = np.concatenate([self.get_binned_observable(var, events) for events in events_list])
        weight = np.concatenate([ak.to_numpy(events["weight"], allow_missing=False) for events in events_list])
        return unfold.build_TH1D(name, binned_var, self.num_bins, weight)

    def th1_to_arrays(self, hist):
        """Extract bin contents and errors from ROOT histogram efficiently."""
        nbins = hist.GetNbinsX()
        values = np.array([hist.GetBinContent(i) for i in range(1, nbins + 1)], dtype=float)
        errors = np.array([hist.GetBinError(i) for i in range(1, nbins + 1)], dtype=float)
        return values, errors

    def get_region_events(self, dl_dict, region, signal_names):
        signal_events = OrderedDict((signal_name, []) for signal_name in signal_names)
        background_events, data_events = [], []

        for dl_name, dl in dl_dict.items():
            events = dl.data[region]
            events = events[events["flags_valid"] > 0]
            events = events[events["theta_cm"] > 0.6]
            events = events[events["mtautau"] > 80]
            if len(events) == 0:
                continue

            is_signal = dl_name in signal_names
            is_mc = not dl.is_data
            if is_signal:
                signal_events[dl_name].append(events)
            elif is_mc:
                background_events.append(events)

            if (dl.is_data and not self.asimov_data) or (self.asimov_data and is_mc):
                data_events.append(events)

        return signal_events, background_events, data_events

    def fit_observable(self, region, signal_names, var, h_data, h_bkg):
        bc_name = ob.get_bc_name_from_variable_name(var)
        nominal_value = ob.NominalBCValues[bc_name]
        data_values, data_errors = self.th1_to_arrays(h_data)
        bkg_values, _ = self.th1_to_arrays(h_bkg)

        analyzing_powers = []
        branching_ratios = []
        for signal_name in signal_names:
            event_category = cf.get_event_category_from_signal_name(signal_name)
            ap_pos, ap_neg = ob.get_analyzing_power_from_event_category(event_category)
            if bc_name.startswith("B_A"):
                analyzing_powers.append(ap_pos*-1)
            elif bc_name.startswith("B_B"):
                analyzing_powers.append(ap_neg)
            elif bc_name.startswith("C_"):
                analyzing_powers.append(-1 * ap_pos * ap_neg)

            branching_ratio = self.get_branching_ratio_from_event_category(event_category)
            branching_ratios.append(branching_ratio)

        truth_hist_build_func = self.build_truth_hist_Bi if bc_name.startswith("B_") else self.build_truth_hist_Cij

        # Prepare data errors: use Poisson if no error info or Asimov is enabled
        if self.asimov_data:
            data_errors = np.sqrt(np.clip(data_values, 1.0, None))
        else:
            # Use provided errors where valid, fall back to Poisson
            data_errors = np.where(data_errors > 0, data_errors, np.sqrt(np.clip(data_values, 1.0, None)))
        
        # Pre-compute inverse errors squared for chi2 computation (avoids recomputing in loop)
        inv_errors_sq = 1.0 / (data_errors ** 2)

        def chi2(x):
            """Chi-squared objective function for minimization."""
            parameter_value = float(x[0])
            h_signal = self.build_expected_signal_hist(region, signal_names, branching_ratios, analyzing_powers, var, parameter_value, truth_hist_build_func)
            signal_values, _ = self.th1_to_arrays(h_signal)
            expected_values = signal_values + bkg_values
            residuals = data_values - expected_values
            return np.sum((residuals ** 2) * inv_errors_sq)

        result = minimize(
            chi2,
            x0=np.array([nominal_value], dtype=float),
            bounds=[tuple(self.fit_parameter_bounds)],
            method="L-BFGS-B",
        )
        best_value = float(result.x[0])
        best_chi2 = float(result.fun)
        err = self.estimate_fit_uncertainty(chi2, best_value)
        return ob.ValueWithUncertainty(best_value, err, err), best_chi2, result

    def estimate_fit_uncertainty(self, chi2, best_value):
        step = 1e-3 * max(1.0, abs(best_value))
        bounds = tuple(self.fit_parameter_bounds)
        x_low = max(bounds[0], best_value - step)
        x_high = min(bounds[1], best_value + step)
        if x_low == best_value or x_high == best_value:
            return 0.0
        f0 = chi2([best_value])
        second_derivative = (chi2([x_high]) - 2.0 * f0 + chi2([x_low])) / (step**2)
        if second_derivative <= 0 or not np.isfinite(second_derivative):
            return 0.0
        return np.sqrt(2.0 / second_derivative)

    def plot_forward_fold_sanity(self, output_dir, region, signal_events, var):
        os.makedirs(output_dir, exist_ok=True)
        nominal_value = ob.NominalBCValues[ob.get_bc_name_from_variable_name(var)]
        bc_name = ob.get_bc_name_from_variable_name(var)
        truth_hist_build_func = self.build_truth_hist_Bi if bc_name.startswith("B_") else self.build_truth_hist_Cij
        for signal_name, events_list in signal_events.items():
            if len(events_list) == 0:
                continue
            h_mc = self.build_reco_hist(f"h_mc_{region}_{signal_name}_{var}", events_list, var)
            signal_category = cf.get_event_category_from_signal_name(signal_name)
            ap_pos, ap_neg = ob.get_analyzing_power_from_event_category(signal_category)
            branching_ratio = self.get_branching_ratio_from_event_category(signal_category)

            analyzing_power = 0.0
            if bc_name.startswith("B_A"):
                analyzing_power = ap_pos * -1
            elif bc_name.startswith("B_B"):
                analyzing_power = ap_neg
            elif bc_name.startswith("C_"):
                analyzing_power = -1 * ap_pos * ap_neg

            response_matrix = self.response_manager.get_response_matrix(region, signal_name, var)

            truth_hist = truth_hist_build_func(analyzing_power, nominal_value, total_bin_contents=self.expected_yields_truth_region * branching_ratio)
            # truth_hist = truth_hist_build_func(analyzing_power, nominal_value, total_bin_contents=374.06)
            truth_hist = unfold.build_TH1D_from_Hist(
                truth_hist,
                f"h_truth_sanity_{region}_{signal_name}_{var}",
            )
            h_folded = self.fold_truth_hist(
                response_matrix,
                truth_hist,
                f"h_folded_sanity_{region}_{signal_name}_{var}",
            )
            # h_folded = truth_hist
            self.plot_hist_comparison(
                h_folded,
                h_mc,
                f"{output_dir}/{signal_name}_{var}_forward_fold_vs_mc.pdf",
                title=f"{region} {signal_name} {var}",
                label_a="Forward folded truth",
                label_b="Reco MC",
            )

    def plot_hist_comparison(self, h_a, h_b, save_path, title, label_a, label_b):
        values_a, errors_a = self.th1_to_arrays(h_a)
        values_b, errors_b = self.th1_to_arrays(h_b)
        x = np.arange(self.num_bins)

        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        ax.errorbar(x, values_a, yerr=errors_a, fmt="o-", label=label_a)
        ax.errorbar(x, values_b, yerr=errors_b, fmt="s-", label=label_b)
        ax.set_ylabel("Events")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)

        ratio = np.divide(values_a, values_b, out=np.zeros_like(values_a), where=values_b != 0)
        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.plot(x, ratio, "o-")
        ax_ratio.set_xlabel("Reco bin")
        ax_ratio.set_ylabel("Fold / MC")
        ax_ratio.set_ylim(0.0, 2.0)
        ax_ratio.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def print_results(self, f_out, label, results):
        cf.print_and_write_to_opened_file(f"\n    {label}:", f_out)
        for key, value in results.items():
            cf.print_and_write_to_opened_file(
                f"        {key}: {value.value:.4f} +{value.err_up:.4f}/-{value.err_down:.4f}", f_out
            )

    def run(self, dl_dict):
        with open(f"{self.output_dir}/results.txt", "w") as f_out:
            for region, signal_names in self.dict_region_to_signals.items():
                cf.print_and_write_to_opened_file(f"\n\nRegion: {region}", f_out)
                region_output_dir = f"{self.output_dir}/{region}"
                os.makedirs(region_output_dir, exist_ok=True)

                signal_events, background_events, data_events = self.get_region_events(dl_dict, region, signal_names)
                if len(data_events) == 0:
                    cf.print_and_write_to_opened_file("    No data events found after selection. Skipping.", f_out)
                    continue

                fitted_bc = {}
                fit_diagnostics = {}
                for var in self.unfold_vars:
                    cf.print_and_write_to_opened_file(f"\n    Fitting {var}", f_out)
                    h_data = self.build_reco_hist(f"h_data_{region}_{var}", data_events, var)
                    h_bkg = self.build_reco_hist(f"h_bkg_{region}_{var}", background_events, var)

                    if self.verbosity >= 1:
                        self.plot_forward_fold_sanity(
                            f"{region_output_dir}/sanity_forward_fold",
                            region,
                            signal_events,
                            var,
                        )

                    fit_value, chi2, fit_result = self.fit_observable(region, signal_names, var, h_data, h_bkg)
                    bc_name = ob.get_bc_name_from_variable_name(var)
                    fitted_bc[bc_name] = fit_value
                    fit_diagnostics[var] = (chi2, fit_result.success, fit_result.message)
                    cf.print_and_write_to_opened_file(
                        f"        {bc_name}: {fit_value.value:.4f} +/- {fit_value.err_up:.4f}, chi2={chi2:.2f}",
                        f_out,
                    )

                self.print_results(f_out, "Fitted B and C matrices", fitted_bc)

                if all(name in fitted_bc for name in ob.NominalBCValues):
                    quantum_results = ob.evaluate_quantum_results_with_uncertainties(fitted_bc)
                    self.print_results(f_out, "Fitted quantum results", quantum_results)

                if self.verbosity >= 0:
                    cf.print_and_write_to_opened_file("\n    Fit diagnostics:", f_out)
                    for var, (chi2, success, message) in fit_diagnostics.items():
                        cf.print_and_write_to_opened_file(
                            f"        {var}: success={success}, chi2={chi2:.2f}, message={message}",
                            f_out,
                        )

    def finalize(self):
        pass
