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

log = logging.getLogger(__name__)

def filter_event(events: ak.Array, filter_log_dict: dict):
    original_events = copy.deepcopy(events)
    filtered_events = {
        'raw': original_events,
    }

    recpart_pdgid = events['Part_pdgId']
    recpart_abspdgid = abs(recpart_pdgid)
    recpart_charge = events['Part_charge']

    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)
    # no less than two pions (regardless of charge for now) in reco particles
    pass_filter = (ak.sum((recpart_abspdgid == 41), axis=1) >= 2) & pass_filter
    filter_key = 'no less than two reco pions'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    # no other hadronic particles in reco particles
    pass_filter = (ak.sum(
        (recpart_abspdgid == 47) |  # pi0
        (recpart_abspdgid == 42) |  # kaon+
        (recpart_abspdgid == 61) |  # KS
        (recpart_abspdgid == 62) |  # KL
        (recpart_abspdgid == 65) |  # proton
        (recpart_abspdgid == 66) |  # neutron
        (recpart_abspdgid == 81),  # lambda
      axis=1) == 0) & pass_filter
    filter_key = 'no other hadronic particles in reco'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    # no electrons or muons in reco particles
    pass_filter = (ak.sum(
        (recpart_abspdgid == 2) |  # electron
        (recpart_abspdgid == 6),  # muon
      axis=1) == 0) & pass_filter
    filter_key = 'no electrons or muons in reco'    
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    # exactly two reco pions 
    pass_filter = (ak.sum((recpart_abspdgid == 41), axis=1) == 2) & pass_filter
    filter_key = 'exactly two pions'
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    # one pi+ and one pi-
    charge_ary_of_pions = recpart_charge[recpart_abspdgid == 41]
    flag_pi_plus_and_pi_minus = (ak.sum(charge_ary_of_pions == 1, axis=1) == 1 )& (ak.sum(charge_ary_of_pions == -1, axis=1) == 1)
    filter_key = 'one pi+ and one pi-'
    pass_filter = flag_pi_plus_and_pi_minus & pass_filter
    filter_log_dict[filter_key] = filter_log_dict.get(filter_key, 0) + ak.sum(pass_filter)

    filtered_events['pipi'] = original_events[pass_filter]

    return filtered_events, filter_log_dict


class DataLoader:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.tree_name = self.config.get("tree_name", "t")
        self.input_files = self.config.get("input_files", [])
        self.region_of_interest = self.config.get("region_of_interest", "pipi")
        self.is_data = self.config.get("is_data", False)

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
        # if os.path.exists(self.output_dir + "/filtered_data.parquet"):
        if len(glob.glob(self.output_dir + "/filtered___*.parquet")) > 0:
            # ask user if they want to load existing data
            load_existing = input(f"Filtered data file already exists. Do you want to reload and filter data from input files? (y/n): ")
            if load_existing.lower() == 'y':
                log.info("Re-loading and filtering data from input files.")
            else:
                log.info("Loading existing filtered data.")
                for file in glob.glob(self.output_dir + "/filtered___*.parquet"):
                    key = os.path.basename(file).split("___")[-1].replace('.parquet', '')
                    self.data[key] = ak.from_parquet(file)
                _data_loaded = True

        if not _data_loaded:
            self.load_data()
            self.save_data()
            _data_loaded = True

        log.info(f"DataLoader initialization complete. Loaded {len(all_files)} files.")

    

    def load_data(self) -> pd.DataFrame:
        # Identify branches to load
        f = ur.open(self.input_files[0])
        tree = f[self.tree_name]

        common_evt_branches = ["Event_evtNumber", "Event_totalChargedEnergy", "Event_totalEMEnergy", "Event_totalHadronicEnergy", "thrust_Mag", "thrust_x", "thrust_y", "thrust_z", "nGoodPart"]
        gen_part_branches = ["pdgId", "status", "vector_fCoordinates_fX", "vector_fCoordinates_fY", "vector_fCoordinates_fZ", "vector_fCoordinates_fT"]
        gen_part_branches = [f"GenPart_{b}" for b in gen_part_branches]
        
        part_branches = ["charge", "pdgId", "fourMomentum_fCoordinates_fX", "fourMomentum_fCoordinates_fY", "fourMomentum_fCoordinates_fZ", "fourMomentum_fCoordinates_fT", "isGood"]
        part_branches = [f'Part_{b}' for b in part_branches]

        branches_to_load = common_evt_branches + part_branches
        if not self.is_data:
            branches_to_load += gen_part_branches

        os.makedirs(self.output_dir + "/filtered_data_files/", exist_ok=True)
        # Load data from all files
        initial_total_num_events = 0
        for file in self.input_files:
            # path_filtered_single_file = self.output_dir + "/filtered_data_files/" + os.path.basename(file).replace('.root', '_filtered.parquet')
            path_filtered_single_file_prefix = self.output_dir + "/filtered_data_files/" + os.path.basename(file).replace('.root', '')
            if len(glob.glob(path_filtered_single_file_prefix + "*_filtered.parquet")) > 0:
                log.info(f"Filtered data file for {file} already exists at {path_filtered_single_file_prefix}. Loading filtered data.")
                log.warning(f"These files are not included in the filter cutflow statistics!")
                for filtered_file in glob.glob(path_filtered_single_file_prefix + "*.parquet"):
                    key = os.path.basename(filtered_file).split("___")[-1].replace('.parquet', '')
                    evt = ak.from_parquet(filtered_file)
                    evt['evtNumber'] = evt['Event_evtNumber'] + initial_total_num_events
                    initial_total_num_events = initial_total_num_events + evt['initial_total_num_events'][0]
                    self.data.setdefault(key, []).append(evt)
            else:
                log.info(f"Loading data from file: {file}")
                try:
                    f = ur.open(file)
                    tree = f[self.tree_name]
                    # load all events as awkward array 
                    events = tree.arrays(branches_to_load, library="ak")

                    # adjust event index to be unique across files
                    # the original Event_evtNumber starts from 1 for each file
                    if len(events) == 0:
                        continue
                    events['evtNumber'] = events['Event_evtNumber'] + initial_total_num_events
                    events['initial_total_num_events'] = len(events)

                    # select Part_xxx via isGood flag
                    events['Part_isGood'] = events['Part_isGood']==1
                    for part_branch in part_branches:
                        events[part_branch] = events[part_branch][events['Part_isGood']] 

                    # filter events
                    self.filter_results['initial_total_num_events'] += len(events)
                    events_pass_filter, self.filter_results = filter_event(events, self.filter_results)

                    # save filtered events
                    log.info(f"Saving filtered data to {path_filtered_single_file_prefix}.")
                    for key, evt in events_pass_filter.items():
                        path_filtered_single_file = path_filtered_single_file_prefix + f"___{key}.parquet"
                        ak.to_parquet(evt, path_filtered_single_file)

                    # record filtered events into self.data
                    for key, evt in events_pass_filter.items():
                        self.data.setdefault(key, []).append(evt)

                except Exception as e:
                    log.error(f"Error reading file {file} or tree {self.tree_name}: {e}")
                    continue

        # Concatenate data from all files
        for key in self.data:
            self.data[key] = ak.concatenate(self.data[key], axis=0)
            self.data[key]['initial_total_num_events'] = initial_total_num_events

        # Log filter results
        if self.filter_results['initial_total_num_events'] > 0:
            for key, value in self.filter_results.items():
                log.info(f"Filter result - {key}: {value}. Filter efficiency: {value / self.filter_results['initial_total_num_events']:.4f}")
            # plot filter results
            cutflow_labels = list(self.filter_results.keys())
            cutflow_values = [self.filter_results[key] for key in cutflow_labels]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_values)
            ax.bar_label(p, labels=[f"{v}" for v in cutflow_values], padding=3)
            ax.set_ylabel('Number of Events')
            ax.set_title('Event Cutflow')
            # rotate x, fontsize to small
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + "/cutflow.pdf")

            cutflow_normalized = [v / self.filter_results['initial_total_num_events'] for v in cutflow_values]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_normalized)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_normalized], padding=3)
            ax.set_ylabel('Efficiency')
            ax.set_title('Event Cutflow Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + "/cutflow_efficiency.pdf")

            cutflow_relative = [cutflow_values[i] / cutflow_values[i-1] if i > 0 else 1.0 for i in range(len(cutflow_values))]
            fig, ax = plt.subplots(dpi=300, figsize=(8,8))
            p = ax.bar(cutflow_labels, cutflow_relative)
            ax.bar_label(p, labels=[f"{v:.4f}" for v in cutflow_relative], padding=3)
            ax.set_ylabel('Relative Efficiency')
            ax.set_title('Event Cutflow Relative Efficiency')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            fig.tight_layout()
            fig.savefig(self.output_dir + "/cutflow_relative_efficiency.pdf")

        return self.data

    
    def save_data(self):
        output_file_prefix = self.output_dir + "/filtered"
        for key, evt in self.data.items():
            output_file = output_file_prefix + f"___{key}.parquet"
            log.info(f"Saving data for channel {key} to {output_file}.")
            ak.to_parquet(evt, output_file)

        log.info(f"Data saved to {output_file}.")

        # structured_output_file = self.output_dir + f"/filtered_{self.region_of_interest}_structured.npy"
        # np.save(structured_output_file, self.structured_data)
        # # To load the structured data, use: np.load(structured_output_file, allow_pickle=True).item()
        # log.info(f"Structured data saved to {structured_output_file}.")


    def run(self, dl):
        pass


    def finalize(self):
        log.info("DataLoader finalization complete.")


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, format = ">>> [%(levelname)s]: %(message)s")
    config = {
        "tree_name": "t",
        "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251029_Ztautau_singlePionDecay/simana_job_17827112_*_ttree.root",
        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/simana_job_*_ttree.root",
        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/simana_job_17841601_*_ttree.root",

        # "input_files": "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251031_Ztautau_singlePi/merged/simana_ttree.root",
    }

    loader = DataLoader(config)

    # import numpy as np
    # import pandas as pd
    # # test reading output file
    # df = pd.read_hdf("filtered_data.h5")
    # print(df)
    # ary = np.load("filtered_data_structured.npy", allow_pickle=True).item()
    # print(ary)
