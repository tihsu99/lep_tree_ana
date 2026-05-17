import numpy as np
import json
import matplotlib.pyplot as plt
import logging
import vector
import glob
import os
import awkward as ak
import copy
import re
from utils.common_functions import load_events_from_parquet
from quantum.observables_builder import shift_SDM_element, get_observable_names

log = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config, output_dir):
        self.config = config
        self.norm_factor = config.get("norm_factor", 1.0)
        self.name = self.config.get("name", "")
        self.load_regions = self.config.get("load_regions", [])
        self.load_regions = list(dict.fromkeys(self.load_regions)) # remove duplicates while preserving order
        self.processed_data_dir = self.config.get("processed_data_dir", "")
        self.is_data = self.config.get("is_data", False)
        self.initial_total_num_events = 0
        self.luminosity = self.config.get("luminosity", 0)
        if self.luminosity == 0 and not self.is_data:
            log.warning("Luminosity is set to 0 for MC sample. Please set luminosity in config for proper normalization.")
        self.is_Ztautau = "Ztautau" in self.name
        self.is_signal = self.is_Ztautau

        self.data = {}
        self.load_data()

        log.info(f"DataLoader initialization complete. Loaded {len(self.data)} regions.")


    @staticmethod
    def load_processed_data(data_dir, sample_name, region_name='raw', is_data=False, is_trainset=False):
        sample_dirs = DataLoader.get_processed_sample_dirs(data_dir, sample_name)
        # split Ztautau into train and test set 
        if sample_name=='Ztautau':
            if is_trainset:
                sample_dirs = sample_dirs[1:]
                print(f"Using train set for sample {sample_name} from {sample_dirs}")
            else:
                sample_dirs = sample_dirs[:1]
                # sample_dirs = sample_dirs[1:]
                print(f"Using test set for sample {sample_name} from {sample_dirs}")

        
        files = [
            os.path.join(sample_dir, f"filtered___{region_name}.parquet")
            for sample_dir in sample_dirs
        ]
        files = [file for file in files if os.path.exists(file)]

        if not files:
            raise FileNotFoundError(
                f"No processed parquet files found for sample '{sample_name}', region '{region_name}' "
                f"under {data_dir}."
            )
        
        if len(files) == 1:
            events = load_events_from_parquet(files[0])
        else:
            events_list = [load_events_from_parquet(file) for file in files]
            events = ak.concatenate(events_list, axis=0)

        # Reweight events based on cutflow information
        initial_total_num_events = 0
        total_weights = 0
        for d in sample_dirs:
            cutflow_file = glob.glob(os.path.join(d, "cutflow_*.json"))[0]
            with open(cutflow_file, "r") as f:
                cutflow = json.load(f)
                assert cutflow[0]['cut'] == 'initial_total_num_events', f"First cut in cutflow should be 'initial_total_num_events' but got {cutflow[0]['cut']} in {cutflow_file}"
                initial_total_num_events += cutflow[0]['events']
                if (total_weights != 0) and (not is_data):
                    assert abs(cutflow[0]['weighted_events'] - total_weights) < 1e-6, f"Weighted events in cutflow do not match across samples for {sample_name}. Please check cutflow files. {total_weights} vs {cutflow[0]['weighted_events']}"
                total_weights = cutflow[0]['weighted_events']

        # # split Ztautau into train and test set 
        # if sample_name=='Ztautau':
        #     half_num_events = len(events) // 2
        #     initial_total_num_events = initial_total_num_events // 2
        #     if is_trainset:
        #         events = events[:half_num_events]
        #         print(f"Using train set for sample {sample_name} from the first half of events: {len(events)} events from {files}")
        #         # sample_dirs = sample_dirs[:-1]
        #         # print(f"Using train set for sample {sample_name} from {len(sample_dirs)} slices: {sample_dirs}")
        #     else:
        #         events = events[half_num_events:]
        #         print(f"Using test set for sample {sample_name} from the second half of events: {len(events)} events from {files}")
        #         # sample_dirs = [sample_dirs[-1]]
        #         # print(f"Using test set for sample {sample_name} from {len(sample_dirs)} slices: {sample_dirs}")
        
        events['initial_total_num_events'] = initial_total_num_events
        weight = 1.0 if is_data else total_weights / initial_total_num_events if initial_total_num_events > 0 else 1.0
        events['weight_nominal'] = weight
        events['weight'] = events['weight_nominal'] # default weight is nominal weight

        # # initilize weight sf
        # for var in get_observable_names():
        #     if not 'cos' in var: continue
        #     if not f'{var}_reweight_sf' in events.fields:
        #         events[f'{var}_reweight_sf'] = ak.ones_like(events['Event_evtNumber'], dtype=np.float32)

        # # test ideal neutrino reconstruction
        # if 'Ztautau' in sample_name:
        #     for obs in get_observable_names():
        #         events[obs] = events[f"truth_{obs}"]
        #     events['flags_valid'] = ak.ones_like(events['Event_evtNumber'], dtype=np.int32)
        return events, initial_total_num_events


    @staticmethod
    def get_processed_sample_dirs(data_dir, sample_name):
        sample_pattern = re.compile(rf"^{re.escape(sample_name)}(?:_(\d+))?$")
        matched_dirs = []

        for path in glob.glob(os.path.join(data_dir, "*")):
            if not os.path.isdir(path):
                continue
            match = sample_pattern.match(os.path.basename(path))
            if match is None:
                continue
            slice_idx = int(match.group(1)) if match.group(1) is not None else -1
            matched_dirs.append((slice_idx, path))

        if not matched_dirs:
            raise FileNotFoundError(f"No processed sample directories found for '{sample_name}' under {data_dir}.")

        return [path for _, path in sorted(matched_dirs)]


    def load_data(self):
        if not self.processed_data_dir:
            raise ValueError("processed_data_dir must be set in the DataLoader config.")

        log.info(f"Loading processed data for sample {self.name} from {self.processed_data_dir}")
        for region in self.load_regions:
            events, self.initial_total_num_events = self.load_processed_data(self.processed_data_dir, self.name, region)
            self.data[region] = events

        return self.data


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