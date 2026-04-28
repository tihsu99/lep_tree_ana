import numpy as np
import pandas as pd
import uproot as ur
import matplotlib.pyplot as plt
import logging
import vector
import glob
import os
import awkward as ak
import copy
from utils.common_functions import get_p4_from_ak_events, get_color_iterator, get_sum_p4_from_ak_events,\
            get_all_p4_from_ak_events, cme, rebuild_p4, load_events_from_parquet
from quantum.observables_builder import build_observables, get_mean_and_err_of_mean, shift_SDM_element
import RegionSelections.DefineVariables as DefineVariables
import RegionSelections.BaselineSelections as BaselineSelections
import RegionSelections.HadHadSelections as HadHadSelections
import RegionSelections.PiPiSelections as PiPiSelections
import RegionSelections.LepLepSelections as LepLepSelections
import RegionSelections.PiRhoSelections as PiRhoSelections

log = logging.getLogger(__name__)

def filter_event(events: ak.Array, filter_log_dict: dict, is_Ztautau=False):
    raw_events = events
    filtered_events_dict = {
        'raw': raw_events,
    }

    raw_events = DefineVariables.define_recon_level_variables(raw_events)
    if is_Ztautau:
        raw_events = DefineVariables.define_signal_exclusive_variables(raw_events)
    
    # baseline selection
    baseline_selection_results = BaselineSelections.get_flag_passes_baseline(raw_events)
    for cut_name, flag_passes_cut in baseline_selection_results.items():
        cut_title = BaselineSelections.get_dict_of_baseline_selection_names()[cut_name]
        filter_log_dict[cut_title] = filter_log_dict.get(cut_title, 0) + ak.sum(flag_passes_cut)
        raw_events[cut_name] = flag_passes_cut
    flag_passes_baseline = baseline_selection_results[BaselineSelections.get_cut_name()]
    filtered_events_dict['baseline'] = raw_events[flag_passes_baseline]

    # had-had selection on top of baseline selection
    hadhad_selection_results = HadHadSelections.get_flag_passes_hadhad_region(raw_events)
    for cut_name, flag_passes_cut in hadhad_selection_results.items():
        cut_title = HadHadSelections.get_dict_of_hadhad_selection_names()[cut_name]
        flag_passes_cut = flag_passes_cut & flag_passes_baseline
        filter_log_dict[cut_title] = filter_log_dict.get(cut_title, 0) + ak.sum(flag_passes_cut)
        raw_events[cut_name] = flag_passes_cut
    flag_passes_hadhad = hadhad_selection_results[HadHadSelections.get_cut_name()] & flag_passes_baseline
    filtered_events_dict['hadhad'] = raw_events[flag_passes_hadhad]

    # pi-pi selection on top of had-had selection
    pipi_selection_results = PiPiSelections.get_flag_passes_pipi_region(raw_events)
    for cut_name, flag_passes_cut in pipi_selection_results.items():
        cut_title = PiPiSelections.get_dict_of_pipi_selection_names()[cut_name]
        flag_passes_cut = flag_passes_cut & flag_passes_hadhad
        filter_log_dict[cut_title] = filter_log_dict.get(cut_title, 0) + ak.sum(flag_passes_cut)
        raw_events[cut_name] = flag_passes_cut
    flag_passes_pipi = pipi_selection_results[PiPiSelections.get_cut_name()] & flag_passes_hadhad
    filtered_events_dict['pipi'] = raw_events[flag_passes_pipi]

    # pi-rho selection on top of had-had selection
    for region_name, is_pion_positive in [('pirho', True), ('rhopi', False)]:
        pirho_selection_results = PiRhoSelections.get_flag_passes_pirho_region(raw_events, is_pion_positive)
        for cut_name, flag_passes_cut in pirho_selection_results.items():
            cut_title = PiRhoSelections.get_dict_of_pirho_selection_names(is_pion_positive)[cut_name]
            flag_passes_cut = flag_passes_cut & flag_passes_hadhad
            filter_log_dict[cut_title] = filter_log_dict.get(cut_title, 0) + ak.sum(flag_passes_cut)
            raw_events[cut_name] = flag_passes_cut
        flag_passes_pirho = pirho_selection_results[PiRhoSelections.get_cut_name(is_pion_positive)] & flag_passes_hadhad
        filtered_events_dict[region_name] = raw_events[flag_passes_pirho]

    # ---------------------------------------------------------
    # Unified Leptonic Selections (ee, mumu, emu)
    # ---------------------------------------------------------
    # Call the selection function OUTSIDE the loop
    selection_results = LepLepSelections.get_flag_passes_leplep_region(raw_events)
    
    # Log the cuts and append the flags to raw_events OUTSIDE the channel split loop
    for cut_name, flag_passes_cut in selection_results.items():
        cut_title = LepLepSelections.get_dict_of_leplep_selection_names()[cut_name]
        flag_passes_cut = flag_passes_cut & flag_passes_baseline
        filter_log_dict[cut_title] = filter_log_dict.get(cut_title, 0) + ak.sum(flag_passes_cut)
        raw_events[cut_name] = flag_passes_cut
            
    # Now simply split the final arrays into their respective dictionaries
    for channel in ['mumu', 'ee', 'emu']:
        cut_name = f"{channel}_cut"
        flag_passes = selection_results[cut_name] & flag_passes_baseline
        filtered_events_dict[channel] = raw_events[flag_passes]

    # Store the raw events with all the defined variables 
    filtered_events_dict['raw'] = raw_events

    return filtered_events_dict, filter_log_dict


class DataLoader:
    def __init__(self, config, output_dir):
        self.config = config
        # load all config into member variables
        for key, value in config.items():
            setattr(self, key, value)
        self.norm_factor = config.get("norm_factor", 1.0)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.name = self.config.get("name", "")
        self.tree_name = self.config.get("tree_name", "t")
        self.input_files = self.config.get("input_files", [])
        self.region_of_interest = self.config.get("region_of_interest", "pipi")
        self.load_regions = self.config.get("load_regions", [self.region_of_interest])
        self.load_regions = list(set(self.load_regions)) # remove duplicates
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

        self.data = {}
        self.filter_results = {
            'initial_total_num_events': 0,
        }

        _data_loaded = False
        if len(glob.glob(self.output_dir + f"/filtered___{self.region_of_interest}.parquet")) > 0:
            log.info(f"Loading existing filtered data from {self.output_dir}")
            for region in self.load_regions:
                file = self.output_dir + "/filtered___" + region + ".parquet"
                if os.path.exists(file):
                    self.data[region] = load_events_from_parquet(file)
                    if len(self.data[region]) == 0:
                        log.warning(f"Filtered data for region {region} is empty. This may be due to previous filtering steps removing all events. Creating empty array for this region.")
                        empty_events = next(iter(self.data.values())) # get the structure of events from any existing region
                        filter_events = ak.zeros_like(empty_events['evtNumber'], dtype=bool)
                        self.data[region] = empty_events[filter_events]
                    if self.initial_total_num_events == 0 and len(self.data[region]) > 0:
                        self.initial_total_num_events = self.data[region]['initial_total_num_events'][0]
                    _data_loaded = True
                else:
                    log.warning(f"Filtered data file for region {region} does not exist. Re-loading and filtering data from input files.")
                    _data_loaded = False
                    break

        if not _data_loaded:
            self.load_data()
            self.save_data()
            keys = list(self.data.keys())
            for key in keys:
                if key not in self.load_regions:
                    del self.data[key]
            _data_loaded = True

        log.info(f"DataLoader initialization complete. Loaded {len(self.input_files)} files.")

    

    def load_data(self) -> pd.DataFrame:
        if os.path.exists(self.output_dir + f"/filtered___raw.parquet"):
            log.info(f"Loading existing raw data from {self.output_dir}/filtered___raw.parquet")
            self.data['raw'] = ak.from_parquet(self.output_dir + f"/filtered___raw.parquet")
            self.initial_total_num_events = self.data['raw']['initial_total_num_events'][0]
            self.filter_results['initial_total_num_events'] = self.initial_total_num_events
            filtered_events, self.filter_results = filter_event(self.data['raw'], self.filter_results, is_Ztautau=self.is_Ztautau)
            for key, evt in filtered_events.items():
                self.data[key] = evt
                self.data[key]['initial_total_num_events'] = self.initial_total_num_events
        else:
            log.info("Loading data from input files.")
            # Identify branches to load
            f = ur.open(self.input_files[0])
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

            # Load data from all files
            initial_total_num_events = 0
            for file in self.input_files:
                log.info(f"Loading data from file: {file}")
                try:
                # if True:
                    f = ur.open(file)
                    tree = f[self.tree_name]
                    # load all events as awkward array 
                    events = tree.arrays(branches_to_load, library="ak")
                except Exception as e:
                    log.error(f"Error reading file {file} or tree {self.tree_name}: {e}")
                    continue

                # adjust event index to be unique across files
                # the original Event_evtNumber starts from 1 for each file
                if len(events) == 0:
                    continue
                events['evtNumber'] = events['Event_evtNumber'] + initial_total_num_events
                # events['initial_total_num_events'] = len(events)
                initial_total_num_events += len(events)

                # select Part_xxx via isGood flag
                part_abscosth = abs(events['Part_fourMomentum_fCoordinates_fZ']) / ((events['Part_fourMomentum_fCoordinates_fX'])**2 + (events['Part_fourMomentum_fCoordinates_fY'])**2 + (events['Part_fourMomentum_fCoordinates_fZ'])**2)**0.5
                flag_not_0pdgid = (events['Part_pdgId'] != 0)
                events['Part_isGood'] = (events['Part_isGood']==1) & (part_abscosth < 0.732) & (part_abscosth > 0.035) # & flag_not_0pdgid
                for part_branch in part_branches:
                    if part_branch != 'Part_isGood':
                        events[part_branch] = events[part_branch][events['Part_isGood']] 

                # filter events
                self.filter_results['initial_total_num_events'] += len(events)
                events_pass_filter, self.filter_results = filter_event(events, self.filter_results, is_Ztautau=self.is_Ztautau)

                # record filtered events into self.data
                for key, evt in events_pass_filter.items():
                    self.data.setdefault(key, []).append(evt)

            # Concatenate data from all files
            for key in self.data:
                self.data[key] = ak.concatenate(self.data[key], axis=0)
                self.initial_total_num_events = initial_total_num_events
                self.data[key]['initial_total_num_events'] = initial_total_num_events
                self.weight = 1 if self.is_data else self.norm_factor / self.initial_total_num_events * self.luminosity
                self.data[key]['weight'] = self.weight * ak.ones_like(self.data[key]['evtNumber'], dtype=np.float32)

        self.weight = 1 if self.is_data else self.norm_factor / self.initial_total_num_events * self.luminosity
        # reconstruct neutrinos of Ztautau raw events for later use in unfolding
        if self.is_Ztautau:
            raw_events = self.data['raw']
            self.data['raw'] = DefineVariables.define_region_specific_variables(raw_events)

        # Log filter results
        if self.filter_results['initial_total_num_events'] > 0:
            with open(self.output_dir + f"/cutflow_{self.name}.txt", "w") as f:
                f.write(f"{'Cut':<40} {'Events':<20} {'Efficiency':<20} {'Relative Efficiency':<20}\n")
                previous_count = self.filter_results['initial_total_num_events']
                for key, value in self.filter_results.items():
                    log.info(f"Filter result - {key}: {value}. Filter efficiency: {value / self.filter_results['initial_total_num_events']:.4f}")
                    efficiency = value / self.filter_results['initial_total_num_events']
                    relative_efficiency = value / previous_count if previous_count > 0 else 1.0
                    f.write(f"{key:<40} {value:<20} {efficiency:<20.4f} {relative_efficiency:<20.4f}\n")
                    previous_count = value

            with open(self.output_dir + f"/cutflow_{self.name}_weighted.txt", "w") as f:
                f.write(f"{'Cut':<40} {'Weighted Events':<20} {'Efficiency':<20} {'Relative Efficiency':<20}\n")
                previous_count = self.weight * self.filter_results['initial_total_num_events']
                for key, value in self.filter_results.items():
                    weighted_value = self.weight * value
                    efficiency = weighted_value / (self.weight * self.filter_results['initial_total_num_events'])
                    relative_efficiency = weighted_value / previous_count if previous_count > 0 else 1.0
                    f.write(f"{key:<40} {weighted_value:<20.4f} {efficiency:<20.4f} {relative_efficiency:<20.4f}\n")
                    previous_count = weighted_value

            # plot filter results
            cutflow_labels = list(self.filter_results.keys())
            cutflow_values = [self.filter_results[key] * self.weight for key in cutflow_labels]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_values)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_values], padding=3, fontsize=4)
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
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_normalized], padding=3, fontsize=4)
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
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_relative], padding=3, fontsize=4)
            ax.set_ylabel('Relative Efficiency')
            ax.set_title('Event Cutflow Relative Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + f"/cutflow_relative_efficiency_{self.name}.pdf")

        return self.data

    
    def save_data(self):
        output_file_prefix = self.output_dir + "/filtered"
        for key, evt in self.data.items():
            output_file = output_file_prefix + f"___{key}.parquet"
            log.info(f"Saving data for region {key} to {output_file}.")
            ak.to_parquet(evt, output_file, compression='snappy')

            log.info(f"Data saved to {output_file}.")


    def postprocess(self):
        # define weight for each event
        if self.initial_total_num_events == 0:
            log.warning("Initial total number of events is 0. This may be due to all events being filtered out or an issue in loading data. Setting weight to 1 for all events to avoid division by zero.")
            weight = 1
        else:
            weight = 1 if self.is_data else self.norm_factor / self.initial_total_num_events * self.luminosity
        for ch, ch_events in self.data.items():
            ch_events['weight_nominal'] = weight * ak.ones_like(ch_events['evtNumber'], dtype=np.float32)
            ch_events['weight'] = ch_events['weight_nominal'] # default weight is nominal weight
        self.current_variation = ('nominal', 0.0)


    def shift_SDM_element(self, element_name, variation):
        if (element_name != 'nominal') and self.is_Ztautau:
            for ch, ch_events in self.data.items():
                new_weight = shift_SDM_element(
                    events = ch_events,
                    element_name = element_name,
                    variation = variation
                )
                ch_events['weight'] = new_weight
        self.current_variation = (element_name, variation)

            





    def finalize(self):
        log.info("DataLoader finalization complete.")


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, format = ">>> [%(levelname)s]: %(message)s")
    config = {
        "tree_name": "t",
        "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251029_Ztautau_singlePionDecay/simana_job_17827112_*_ttree.root",
    }

    loader = DataLoader(config)