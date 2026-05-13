import os
import json
from collections import OrderedDict

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from BaseProcessor import BaseProcessor

import quantum.unfold as unfold
from processor.forward_folding_fit import NuisanceParameterSpec, SinglePOIFitter
from processor.ResponseMatricesManager import ResponseMatricesManager
import quantum.observables_builder as ob
import utils.common_functions as cf
from utils.tau_decay import (
    NOMINAL_BC_VALUES,
    get_analyzing_powers_from_event_category,
    get_branching_ratio_from_event_category,
    get_event_category_from_signal_name,
)


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
        self.nuisance_parameter_specs = self.build_nuisance_parameter_specs(
            config.get("nuisance_parameters")
        )
        self.fit_results = OrderedDict()

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
        
    def build_nuisance_parameter_specs(self, nuisance_config):
        if nuisance_config is None:
            nuisance_config = {
                "signal_norm": {"initial_value": 1.0, "bounds": (0.0, 2.0), "fit": False},
                "background_norm": {"initial_value": 1.0, "bounds": (0.0, 2.0), "fit": False},
            }

        specs = []
        for name, cfg in nuisance_config.items():
            if cfg is None:
                cfg = {}
            specs.append(
                NuisanceParameterSpec(
                    name=name,
                    initial_value=float(cfg.get("initial_value", cfg.get("initial", 1.0))),
                    bounds=tuple(cfg.get("bounds", (0.0, 2.0))),
                    fit=bool(cfg.get("fit", False)),
                    constraint_sigma=(
                        None
                        if cfg.get("constraint_sigma", cfg.get("sigma")) is None
                        else float(cfg.get("constraint_sigma", cfg.get("sigma")))
                    ),
                )
            )
        return specs

    def get_binned_observable(self, var, events):
        var_values = ak.to_numpy(events[var], allow_missing=False)
        binned_var = unfold.bin_variable(var_values, self.bin_edges)
        return binned_var.astype(float)


    def fold_truth_hist(self, response_matrix, truth_hist, name):
        folded = response_matrix.ApplyToTruth(truth_hist, name)
        folded.SetDirectory(0)
        return folded

    def build_expected_signal_hist(self, region, signal_names, branching_ratios, event_categories, var, parameter_value):
        expected = None
        
        for idx, signal_name in enumerate(signal_names):
            branching_ratio = branching_ratios[idx] 
            
            # Build truth template specific to this signal's analyzing power and branching ratio
            truth_hist, _ = ob.get_theoretical_distribution(var_name=var, signal_category=event_categories[idx], norm=self.expected_yields_truth_region * branching_ratio, bin_edges=self.bin_edges, bc_value=parameter_value)
            truth_hist = unfold.build_TH1D(f"h_truth_{region}_{signal_name}_{var}_{parameter_value:.5f}", var=np.arange(self.num_bins), num_bins=self.num_bins, weight=truth_hist)
            for i in range(truth_hist.GetNbinsX()):
                truth_hist.SetBinError(i+1, 0)
            truth_hist.SetDirectory(0)
            
            # Fold through this signal's response matrix
            response_matrix = self.response_manager.get_response_matrix(region, signal_name, var)
            folded = self.fold_truth_hist(
                response_matrix,
                truth_hist,
                f"h_ff_{region}_{signal_name}_{var}_{parameter_value:.5f}",
            )
            h_fake = response_matrix.Hfakes()
            folded.Add(h_fake)
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
        weight = weight * np.concatenate([ak.to_numpy(events[f'{var}_reweight_sf'], allow_missing=False) for events in events_list])
        return unfold.build_TH1D(name, binned_var, self.num_bins, weight)

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

    def get_signal_model_inputs(self, signal_names, bc_name):
        # analyzing_powers = []
        branching_ratios = []
        event_categories = []
        for signal_name in signal_names:
            event_category = get_event_category_from_signal_name(signal_name)
            event_categories.append(event_category)

            branching_ratio = get_branching_ratio_from_event_category(event_category)
            branching_ratios.append(branching_ratio)
        # truth_hist_build_func = self.build_truth_hist_Bi if bc_name.startswith("B_") else self.build_truth_hist_Cij
        return branching_ratios, event_categories

    def build_expected_values(self, region, signal_names, var, parameter_value, nuisance_parameters, h_bkg):
        bc_name = ob.get_bc_name_from_variable_name(var)
        # branching_ratios, analyzing_powers, truth_hist_build_func = self.get_signal_model_inputs(signal_names, bc_name)
        branching_ratios, event_categories = self.get_signal_model_inputs(signal_names, bc_name)
        h_signal = self.build_expected_signal_hist(
            region,
            signal_names,
            branching_ratios,
            event_categories,
            var,
            parameter_value,
        )
        signal_values = unfold.build_Hist_from_TH1D(h_signal).values
        bkg_values = unfold.build_Hist_from_TH1D(h_bkg).values
        signal_norm = nuisance_parameters.get("signal_norm", 1.0)
        background_norm = nuisance_parameters.get("background_norm", 1.0)
        return signal_norm * signal_values + background_norm * bkg_values

    def fit_observable(self, region, signal_names, var, h_data, h_bkg):
        bc_name = ob.get_bc_name_from_variable_name(var)
        data_hist = unfold.build_Hist_from_TH1D(h_data)
        data_values, data_errors = data_hist.values, data_hist.errors

        # Prepare data errors: use Poisson if no error info or Asimov is enabled
        if self.asimov_data:
            data_errors = np.sqrt(np.clip(data_values, 1.0, None))
        else:
            # Use provided errors where valid, fall back to Poisson
            data_errors = np.where(data_errors > 0, data_errors, np.sqrt(np.clip(data_values, 1.0, None)))

        fitter = SinglePOIFitter(
            poi_name=bc_name,
            nominal_poi_value=NOMINAL_BC_VALUES[bc_name],
            poi_bounds=tuple(self.fit_parameter_bounds),
            nuisance_parameter_specs=self.nuisance_parameter_specs,
            data_values=data_values,
            data_errors=data_errors,
            build_expected_values=lambda poi_value, nuisance_parameters: self.build_expected_values(
                region, signal_names, var, poi_value, nuisance_parameters, h_bkg
            ),
        )
        fit_result = fitter.fit()
        err = fit_result.poi_uncertainty
        fit_value = ob.ValueWithUncertainty(
            fit_result.postfit.pois[bc_name],
            err,
            err,
        )
        return fit_value, fit_result

    def plot_data_mc_comparison(
        self,
        output_dir,
        region,
        signal_names,
        var,
        h_data,
        h_bkg,
        parameter_value,
        nuisance_parameters,
        label,
    ):
        os.makedirs(output_dir, exist_ok=True)
        bc_name = ob.get_bc_name_from_variable_name(var)
        branching_ratios, event_categories = self.get_signal_model_inputs(signal_names, bc_name)
        h_signal = self.build_expected_signal_hist(
            region,
            signal_names,
            branching_ratios,
            event_categories,
            var,
            parameter_value,
        )

        data_hist = unfold.build_Hist_from_TH1D(h_data)
        data_values, data_errors = data_hist.values, data_hist.errors
        signal_values = unfold.build_Hist_from_TH1D(h_signal).values
        bkg_values = unfold.build_Hist_from_TH1D(h_bkg).values
        signal_norm = nuisance_parameters.get("signal_norm", 1.0)
        background_norm = nuisance_parameters.get("background_norm", 1.0)
        signal_values = signal_norm * signal_values
        bkg_values = background_norm * bkg_values
        mc_values = signal_values + bkg_values
        if self.asimov_data:
            data_errors = np.sqrt(np.clip(data_values, 1.0, None))
        else:
            data_errors = np.where(data_errors > 0, data_errors, np.sqrt(np.clip(data_values, 1.0, None)))

        x = np.arange(self.num_bins)
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        ax.bar(x, bkg_values, label="Background", color="tab:gray", alpha=0.65)
        ax.bar(x, signal_values, bottom=bkg_values, label="Signal", color="tab:blue", alpha=0.65)
        ax.errorbar(x, data_values, yerr=data_errors, fmt="ko", label="Data")
        ax.set_ylabel("Events")
        ax.set_title(f"{region} {var} {label}: {bc_name}={parameter_value:.4f}")
        ax.legend()
        ax.grid(alpha=0.25)

        ratio = np.divide(data_values, mc_values, out=np.zeros_like(data_values), where=mc_values != 0)
        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.errorbar(x, ratio, fmt="ko")
        ax_ratio.set_xlabel("Reco bin")
        ax_ratio.set_ylabel("Data / MC")
        ax_ratio.set_ylim(0.0, 2.0)
        ax_ratio.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/{var}_{label}_data_mc.png")
        plt.close(fig)

    def write_fit_snapshots(self, region, fit_results):
        payload = OrderedDict()
        for bc_name, fit_result in fit_results.items():
            payload[bc_name] = {
                "prefit": fit_result.prefit.to_dict(),
                "postfit": fit_result.postfit.to_dict(),
                "poi_uncertainty": fit_result.poi_uncertainty,
                "neg2_log_likelihood": fit_result.neg2_log_likelihood,
                "success": bool(fit_result.optimizer_result.success),
                "message": str(fit_result.optimizer_result.message),
            }
        with open(f"{self.output_dir}/{region}/fit_parameters.json", "w") as f_json:
            json.dump(payload, f_json, indent=2)

    def run(self, dl_dict):
        f_results = open(f"{self.output_dir}/results.txt", "w")
        f_running_log = open(f"{self.output_dir}/running_log.txt", "w")
        for region, signal_names in self.dict_region_to_signals.items():
            cf.print_and_write_to_opened_file(f"Region: {region}", f_results)
            cf.print_and_write_to_opened_file(f"Region: {region}", f_running_log)
            region_output_dir = f"{self.output_dir}/{region}"
            os.makedirs(region_output_dir, exist_ok=True)

            _, background_events, data_events = self.get_region_events(dl_dict, region, signal_names)
            if len(data_events) == 0:
                cf.print_and_write_to_opened_file("    No data events found after selection. Skipping.", f_running_log)
                continue

            fitted_bc = {}
            observable_fit_results = OrderedDict()
            fit_diagnostics = {}
            for var in self.unfold_vars:
                cf.print_and_write_to_opened_file(f"\n    Fitting {var}", f_running_log)
                h_data = self.build_reco_hist(f"h_data_{region}_{var}", data_events, var)
                h_bkg = self.build_reco_hist(f"h_bkg_{region}_{var}", background_events, var)

                bc_name = ob.get_bc_name_from_variable_name(var)
                if self.verbosity >= 1:
                    prefit_nps = OrderedDict(
                        (spec.name, spec.initial_value)
                        for spec in self.nuisance_parameter_specs
                    )
                    self.plot_data_mc_comparison(
                        f"{region_output_dir}/data_mc_prefit",
                        region,
                        signal_names,
                        var,
                        h_data,
                        h_bkg,
                        NOMINAL_BC_VALUES[bc_name],
                        prefit_nps,
                        "prefit",
                    )

                fit_value, fit_result = self.fit_observable(region, signal_names, var, h_data, h_bkg)
                fitted_bc[bc_name] = fit_value
                observable_fit_results[bc_name] = fit_result
                fit_diagnostics[var] = (
                    fit_result.neg2_log_likelihood,
                    fit_result.optimizer_result.success,
                    fit_result.optimizer_result.message,
                )
                cf.print_and_write_to_opened_file(
                    f"        {bc_name}: {fit_value.value:.4f} +/- {fit_value.err_up:.4f}, -2logL={fit_result.neg2_log_likelihood:.2f}",
                    f_running_log,
                )

                if self.verbosity >= 1:
                    self.plot_data_mc_comparison(
                        f"{region_output_dir}/data_mc_postfit",
                        region,
                        signal_names,
                        var,
                        h_data,
                        h_bkg,
                        fit_result.postfit.pois[bc_name],
                        fit_result.postfit.nuisance_parameters,
                        "postfit",
                    )

            self.fit_results[region] = observable_fit_results
            self.write_fit_snapshots(region, observable_fit_results)
            ob.print_results(f_results, fitted_bc)

            cf.print_and_write_to_opened_file("\n    Post-fit nuisance parameters:", f_running_log)
            for bc_name, fit_result in observable_fit_results.items():
                np_text = ", ".join(
                    f"{name}={value:.4f}"
                    for name, value in fit_result.postfit.nuisance_parameters.items()
                )
                cf.print_and_write_to_opened_file(f"        {bc_name}: {np_text}", f_running_log)

            if all(name in fitted_bc for name in NOMINAL_BC_VALUES):
                quantum_results = ob.evaluate_quantum_results_with_uncertainties(fitted_bc)
                ob.print_results(f_results, quantum_results)

            if self.verbosity >= 0:
                cf.print_and_write_to_opened_file("\n    Fit diagnostics:", f_running_log)
                for var, (neg2_log_likelihood, success, message) in fit_diagnostics.items():
                    cf.print_and_write_to_opened_file(
                        f"        {var}: success={success}, -2logL={neg2_log_likelihood:.2f}, message={message}",
                        f_running_log,
                    )
            cf.print_and_write_to_opened_file("\n\n", f_results)
            cf.print_and_write_to_opened_file("\n\n", f_running_log)

        f_results.close()
        f_running_log.close()

    def finalize(self):
        pass
