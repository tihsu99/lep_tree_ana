import numpy as np
import pandas as pd
import uproot as ur
import matplotlib.pyplot as plt
import logging
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import awkward as ak
import Preprocess.DefineVariables as DefineVariables
import Preprocess.BaselineSelections as BaselineSelections
import Preprocess.HadHadSelections as HadHadSelections
import Preprocess.LepLepSelections as LepLepSelections

from tqdm import tqdm

log = logging.getLogger(__name__)

def apply_selection(raw_events, filter_log_dict, selection, parent_flag=None):
    selection_name = selection.selection_name if selection.selection_name is not None else selection.__class__.__name__
    selection_results = selection.get_flags(raw_events)

    final_flag = None
    for i_cut, description in enumerate(selection.cut_descriptions):
        flag_passes_cut = selection_results[description]
        if parent_flag is not None:
            flag_passes_cut = flag_passes_cut & parent_flag
        filter_log_dict[description] = filter_log_dict.get(description, 0) + ak.sum(flag_passes_cut)
        raw_events[f'{selection_name}_cut_{i_cut}'] = flag_passes_cut
        final_flag = flag_passes_cut
    return final_flag


def filter_event(events: ak.Array, filter_log_dict: dict, is_Ztautau=False):
    raw_events = events

    raw_events = DefineVariables.define_recon_level_variables(raw_events)
    if is_Ztautau:
        raw_events = DefineVariables.define_signal_exclusive_variables(raw_events)
    
    flags = {}
    baseline_selection = BaselineSelections.BaselineSelection()
    flag_passes_baseline = apply_selection(raw_events, filter_log_dict, baseline_selection)
    raw_events['baseline_cut'] = flag_passes_baseline
    flags['baseline'] = flag_passes_baseline

    hadhad_selection = HadHadSelections.HadHadSelection()
    flag_passes_hadhad = apply_selection(raw_events, filter_log_dict, hadhad_selection, flag_passes_baseline)
    raw_events['hadhad_cut'] = flag_passes_hadhad
    flags['hadhad'] = flag_passes_hadhad

    pipi_selection = HadHadSelections.PiPiSelection()
    flag_passes_pipi = apply_selection(raw_events, filter_log_dict, pipi_selection, flag_passes_hadhad)
    raw_events['pipi_cut'] = flag_passes_pipi
    flags['pipi'] = flag_passes_pipi

    for region_name, is_pion_positive in [('pirho', True), ('rhopi', False)]:
        pirho_selection = HadHadSelections.PiRhoSelection(is_pion_positive)
        flag_passes_pirho = apply_selection(raw_events, filter_log_dict, pirho_selection, flag_passes_hadhad)
        raw_events[f'{region_name}_cut'] = flag_passes_pirho
        flags[region_name] = flag_passes_pirho

    leplep_selection = LepLepSelections.LepLepSelection()
    flag_passes_leplep = apply_selection(raw_events, filter_log_dict, leplep_selection, flag_passes_baseline)

    lepton_channel_selections = {
        'mumu': LepLepSelections.MuMuSelection(),
        'ee': LepLepSelections.EESelection(),
        'emu': LepLepSelections.MuESelection(),
    }
    for channel, selection in lepton_channel_selections.items():
        flag_passes = apply_selection(raw_events, filter_log_dict, selection, flag_passes_leplep)
        raw_events[f'{channel}_cut'] = flag_passes
        flags[channel] = flag_passes

    return raw_events, filter_log_dict, flags


def get_tree_num_entries(file, tree_name):
    with ur.open(file) as f:
        return f[tree_name].num_entries


def process_input_file(args):
    file, tree_name, branches_to_load, part_branches, event_offset, is_Ztautau = args
    filter_results = {
        'initial_total_num_events': 0,
    }

    with ur.open(file) as f:
        tree = f[tree_name]
        events = tree.arrays(branches_to_load, library="ak")

    if len(events) == 0:
        return {}, filter_results, 0

    events['evtNumber'] = events['Event_evtNumber'] + event_offset

    part_abscosth = abs(events['Part_fourMomentum_fCoordinates_fZ']) / (
        (
            events['Part_fourMomentum_fCoordinates_fX']**2
            + events['Part_fourMomentum_fCoordinates_fY']**2
            + events['Part_fourMomentum_fCoordinates_fZ']**2
        )**0.5
    )
    events['Part_isGood'] = (
        (events['Part_isGood'] == 1)
        & (part_abscosth < 0.732)
        & (part_abscosth > 0.035)
    )

    for part_branch in part_branches:
        if part_branch != 'Part_isGood':
            events[part_branch] = events[part_branch][events['Part_isGood']]

    filter_results['initial_total_num_events'] += len(events)
    return filter_event(events, filter_results, is_Ztautau=is_Ztautau) + (len(events),)


class Preprocessor:
    def __init__(self, config, output_dir):
        self.config = config
        self.norm_factor = config.get("norm_factor", 1.0)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.name = self.config.get("name", "")
        self.tree_name = self.config.get("tree_name", "t")
        self.input_files = self.config.get("input_files", [])
        self.is_data = self.config.get("is_data", False)
        self.initial_total_num_events = 0
        self.luminosity = self.config.get("luminosity", 0)
        if self.luminosity == 0 and not self.is_data:
            log.warning("Luminosity is set to 0 for MC sample. Please set luminosity in config for proper normalization.")

        self.is_Ztautau = "Ztautau" in self.name

        if not self.input_files:
            raise ValueError("Input files must be specified.")
        elif isinstance(self.input_files, str):
            self.input_files = glob.glob(self.input_files)
        else:
            all_files = []
            for pattern in self.input_files:
                all_files.extend(glob.glob(pattern))
            # sort files for consistency
            all_files = sorted(all_files)
            self.input_files = all_files

        self.filter_results = {
            'initial_total_num_events': 0,
        }

        self.load_data()

    

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.output_dir + f"/filtered___raw.parquet"):
            log.info(f"Loading existing raw data from {self.output_dir}/filtered___raw.parquet")
            self.raw_events = ak.from_parquet(self.output_dir + f"/filtered___raw.parquet")
            self.initial_total_num_events = self.raw_events['initial_total_num_events'][0]
            self.filter_results['initial_total_num_events'] = self.initial_total_num_events
            self.raw_events, self.filter_results, flags = filter_event(self.raw_events, self.filter_results, is_Ztautau=self.is_Ztautau)
            self.regions = list(flags.keys())
        else:
            log.info("Loading data from input files.")
            # Identify branches to load
            with ur.open(self.input_files[0]) as f:
                tree = f[self.tree_name]

            common_evt_branches = ["Event_evtNumber", "Event_totalChargedEnergy", "Event_totalEMEnergy", "Event_totalHadronicEnergy", "thrust_Mag", "thrust_x", "thrust_y", "thrust_z", "nGoodPart", 
                "event_category"
            ]
            gen_part_branches = ["pdgId", "status", "vector_fCoordinates_fX", "vector_fCoordinates_fY", "vector_fCoordinates_fZ", "vector_fCoordinates_fT"]
            gen_part_branches = [f"GenPart_{b}" for b in gen_part_branches]
            
            part_branches = [
                "charge", "pdgId", "fourMomentum_fCoordinates_fX", "fourMomentum_fCoordinates_fY", "fourMomentum_fCoordinates_fZ", "fourMomentum_fCoordinates_fT", "isGood", "vtxIdx", 
                "hpcShowerEnergy", "hpcShowerTheta", "hpcShowerPhi", "hpcParticleCode", "hpcNumLayers", "hpcLayerHitPattern", "hpcNumAssociatedShowers", "hpcTotalShowerEnergy", 
                "hacShowerEnergy", "hacShowerTheta", "hacShowerPhi", "hacParticleCode", "hacNumTowers", "hacTowerHitPattern", "hacNumAssociatedShowers", "hacTotalShowerEnergy", 
                "sticShowerEnergy", "sticShowerTheta", "sticShowerPhi", "sticNumTowers", "sticChargedTag", "sticSiliconVertexPos",  
                "lock",
            ]
            part_branches = [f'Part_{b}' for b in part_branches]
            id_branches = [
                "Elid_partIdx", "Elid_tag", "Elid_gammaConversion",
                "Muid_partIdx", "Muid_tag", "Muid_hitPattern",
                "Haid_pionRich", "Haidn_pionTag", "Haidr_pionTag", "Haide_pionTag", "Haidc_pionTag"
            ]
            track_branches = [ f'Trac_{b}' for b in 
                [
                    "originVtxIdx", "impParToVertexRPhi", "impParToVertexZ", "impParRPhi", "impParZ",
                ]
            ]
            
            # Dedx branches are not Part_ prefixed, they are top-level
            dedx_branches = ["Dedx_value", "Dedx_error", "Dedx_nrWires"]

            part_branches = part_branches + id_branches + track_branches

            vertex_branches = [ f'Vtx_{b}' for b in 
                ["position_fCoordinates_fX", "position_fCoordinates_fY", "position_fCoordinates_fZ",]
            ]

            branches_to_load = common_evt_branches + part_branches + vertex_branches + dedx_branches
            if not self.is_data:
                branches_to_load += gen_part_branches

            files_to_process = []
            event_offsets = []
            initial_total_num_events = 0
            for file in self.input_files:
                try:
                    num_entries = get_tree_num_entries(file, self.tree_name)
                except Exception as e:
                    log.error(f"Error reading file {file} or tree {self.tree_name}: {e}")
                    continue

                event_offsets.append(initial_total_num_events)
                files_to_process.append(file)
                initial_total_num_events += num_entries

            self.initial_total_num_events = initial_total_num_events
            max_workers = self.config.get("num_workers", os.cpu_count() or 1)
            max_workers = min(max_workers, len(files_to_process)) if files_to_process else 1
            log.info(f"Processing {len(files_to_process)} input files with {max_workers} workers.")
            jobs = [
                (file, self.tree_name, branches_to_load, part_branches, event_offset, self.is_Ztautau)
                for file, event_offset in zip(files_to_process, event_offsets)
            ]

            results = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(process_input_file, job): idx
                    for idx, job in enumerate(jobs)
                }
                completed_futures = as_completed(future_to_index)
                completed_futures = tqdm(
                    completed_futures,
                    total=len(future_to_index),
                    desc=f"Preprocessing {self.name}",
                    unit="file",
                )

                for future in completed_futures:
                    idx = future_to_index[future]
                    file = files_to_process[idx]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        log.error(f"Error processing file {file} or tree {self.tree_name}: {e}")
                    log.info(f"Processed {len(results)}/{len(future_to_index)} files for {self.name}.")

            self.raw_events = []
            self.regions = None
            for idx in sorted(results):
                raw_events, file_filter_results, flags, init_num_events = results[idx]

                for key, value in file_filter_results.items():
                    self.filter_results[key] = self.filter_results.get(key, 0) + value

                self.raw_events.append(raw_events)
                if self.regions is None and flags is not None:
                    self.regions = list(flags.keys())

            self.raw_events = ak.concatenate(self.raw_events, axis=0)
        # reconstruct neutrinos of Ztautau raw events for later use in unfolding
        if self.is_Ztautau:
            self.raw_events = DefineVariables.define_region_specific_variables(self.raw_events)

        # Store the raw events with all the defined variables
        self.raw_events['initial_total_num_events'] = self.initial_total_num_events
        output_file_name = self.output_dir + f"/filtered___raw.parquet"
        ak.to_parquet(self.raw_events, output_file_name, compression='snappy')
        log.info(f"Raw data saved to {output_file_name}.")

        # Concatenate data from all files
        for key in self.regions:
            data = self.raw_events[self.raw_events[f'{key}_cut']]
            output_file_name = self.output_dir + f"/filtered___{key}.parquet"
            ak.to_parquet(data, output_file_name, compression='snappy')
            log.info(f"Data for region {key} saved to {output_file_name}.")


        self.weight = 1 if self.is_data else self.norm_factor / self.initial_total_num_events * self.luminosity

        # Log filter results
        if self.filter_results['initial_total_num_events'] > 0:
            initial_count = self.filter_results['initial_total_num_events']
            initial_weighted_count = self.weight * initial_count
            previous_count = initial_count
            previous_weighted_count = initial_weighted_count
            cutflow_records = []

            for step, (key, value) in enumerate(self.filter_results.items()):
                weighted_value = self.weight * value
                efficiency = value / initial_count
                weighted_efficiency = weighted_value / initial_weighted_count if initial_weighted_count > 0 else 0
                relative_efficiency = value / previous_count if previous_count > 0 else 1.0
                weighted_relative_efficiency = weighted_value / previous_weighted_count if previous_weighted_count > 0 else 1.0

                log.info(f"Filter result - {key}: {value}. Filter efficiency: {efficiency:.4f}")
                cutflow_records.append({
                    "step": step,
                    "cut": key,
                    "events": int(value),
                    "weighted_events": float(weighted_value),
                    "efficiency": float(efficiency),
                    "weighted_efficiency": float(weighted_efficiency),
                    "relative_efficiency": float(relative_efficiency),
                    "weighted_relative_efficiency": float(weighted_relative_efficiency),
                })

                previous_count = value
                previous_weighted_count = weighted_value

            cutflow_df = pd.DataFrame(cutflow_records)
            # cutflow_df.to_csv(self.output_dir + f"/cutflow_{self.name}.csv", index=False)
            cutflow_df.to_json(self.output_dir + f"/cutflow_{self.name}.json", orient="records", indent=2)

            # plot filter results
            cutflow_labels = list(self.filter_results.keys())
            cutflow_values = [self.filter_results[key] * self.weight for key in cutflow_labels]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_values)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_values], padding=3, fontsize=4, rotation=90)
            ax.set_ylabel('Number of Events')
            ax.set_title('Event Cutflow')
            ax.set_yscale('log')
            # rotate x, fontsize to small
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + f"/cutflow_{self.name}_weighted.pdf")

            cutflow_normalized = [v / cutflow_values[0] for v in cutflow_values]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_normalized)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_normalized], padding=3, fontsize=4, rotation=90)
            ax.set_ylabel('Efficiency')
            ax.set_title('Event Cutflow Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + f"/cutflow_efficiency_{self.name}.pdf")

            cutflow_relative = [1.0]
            tmp_cutflow_label = ['initial_totoal_num_events']
            for i in range(1, len(cutflow_values)):
                rel = cutflow_values[i] / cutflow_values[i-1] if cutflow_values[i-1] > 0 else 0
                label = cutflow_labels[i]
                if rel>1:
                    # if eff>1 then calculate ratio relative to initial num
                    rel = cutflow_values[i] / cutflow_values[0]
                    label = f"{label}/initialNoE"
                cutflow_relative.append(rel)
                tmp_cutflow_label.append(label)

            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(tmp_cutflow_label, cutflow_relative)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_relative], padding=3, fontsize=4, rotation=90)
            ax.set_ylabel('Relative Efficiency')
            ax.set_title('Event Cutflow Relative Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + f"/cutflow_relative_efficiency_{self.name}.pdf")


