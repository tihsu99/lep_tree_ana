import numpy as np
import pandas as pd
import uproot as ur
import logging
import vector
import glob
import os
import awkward as ak
import copy

log = logging.getLogger(__name__)

def filter_event(events: ak.Array) -> dict:
    original_events = copy.deepcopy(events)
    filtered_events = {}

    genpart_pdgid = events['GenPart_pdgId']
    genpart_abspdgid = abs(genpart_pdgid)
    genpart_status = events['GenPart_status']
    is_finalstatus = genpart_status==1

    ##############################
    # tau tau > pi+ pi- v v events
    ##############################
    pass_filter = ak.ones_like(events['evtNumber'], dtype=bool)
    # no kaon, lambda and Xi0: no status=4 particles
    pass_filter = ~ak.any((genpart_status == 4), axis=1) & pass_filter
    # exactly one pi+ and one pi- in final status particles
    pass_filter = (ak.sum((genpart_pdgid == 211) & is_finalstatus, axis=1) == 1) & pass_filter
    pass_filter = (ak.sum((genpart_pdgid == -211) & is_finalstatus, axis=1) == 1) & pass_filter

    # no neutral pions, no short-lived particles, no kaons, eta, omega, neutrinos other than nu_tau
    num_short_lived = ak.sum((genpart_status == 4), axis=1)
    num_unwanted = \
        ak.sum((genpart_abspdgid == 111), axis=1) + \
        ak.sum((genpart_abspdgid == 321), axis=1) + \
        ak.sum((genpart_abspdgid == 221), axis=1) + \
        ak.sum((genpart_abspdgid == 223), axis=1) + \
        ak.sum((genpart_abspdgid == 14), axis=1)  + \
        ak.sum((genpart_abspdgid == 12), axis=1)      
    pass_filter = (num_short_lived == 0) & (num_unwanted == 0) & pass_filter
    filtered_events['single_pion_decay'] = original_events[pass_filter]

    return filtered_events


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.tree_name = self.config.get("tree_name", "t")
        self.input_files = self.config.get("input_files", [])

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
        self.structured_data = {}
        self.filter_results = {
            'initial_total_num_events': 0,
        }

        _data_loaded = False
        if os.path.exists(self.config.get("output_dir", "./") + "/filtered_data.parquet"):
            # ask user if they want to load existing data
            load_existing = input(f"Filtered data file already exists. Do you want to reload and filter data from input files? (y/n): ")
            if load_existing.lower() == 'y':
                log.info("Re-loading and filtering data from input files.")
            else:
                log.info("Loading existing filtered data.")
                loaded = ak.from_parquet(self.config.get("output_dir", "./") + "/filtered_data.parquet")
                for key in loaded.fields:
                    self.data[key] = loaded[key]
                self.structured_data = np.load(self.config.get("output_dir", "./") + "/filtered_data_structured.npy", allow_pickle=True).item()
                _data_loaded = True

        if not _data_loaded:
            self.load_data()
            self.save_data()
            _data_loaded = True

        # vectorize the data
        def vectorize_p4(key):
            return vector.zip({
                "px": self.structured_data[key][:, 0],
                "py": self.structured_data[key][:, 1],
                "pz": self.structured_data[key][:, 2],
                "E": self.structured_data[key][:, 3],
            })
        self.vectored_data = {
            key.removesuffix("_p4"): vectorize_p4(key)
            for key in self.structured_data
        }

        log.info(f"DataLoader initialization complete. Loaded {len(all_files)} files.")

    

    def load_data(self) -> pd.DataFrame:
        # Identify branches to load
        f = ur.open(self.input_files[0])
        tree = f[self.tree_name]

        common_evt_branches = ["Event_evtNumber"]
        gen_part_branches = ["pdgId", "status", "vector_fCoordinates_fX", "vector_fCoordinates_fY", "vector_fCoordinates_fZ", "vector_fCoordinates_fT"]
        gen_part_branches = [f"GenPart_{b}" for b in gen_part_branches]

        part_branches = ["pdgId", "fourMomentum_fCoordinates_fX", "fourMomentum_fCoordinates_fY", "fourMomentum_fCoordinates_fZ", "fourMomentum_fCoordinates_fT", ]
        part_branches = [f'Part_{b}' for b in part_branches]

        branches_to_load = common_evt_branches + gen_part_branches + part_branches

        os.makedirs(self.config.get("output_dir", "./") + "/filtered_data_files/", exist_ok=True)
        # Load data from all files
        initial_total_num_events = 0
        for file in self.input_files:
            path_filtered_single_file = self.config.get("output_dir", "./") + "/filtered_data_files/" + os.path.basename(file).replace('.root', '_filtered.parquet')
            if os.path.exists(path_filtered_single_file):
                log.info(f"Filtered data file for {file} already exists at {path_filtered_single_file}. Loading filtered data.")
                filtered_events = ak.from_parquet(path_filtered_single_file)
                for key in filtered_events.fields:
                    evt = filtered_events[key]
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

                    # filter events
                    self.filter_results['initial_total_num_events'] += initial_total_num_events
                    events_pass_filter = filter_event(events)
                    for key, evt in events_pass_filter.items():
                        self.filter_results[key] = self.filter_results.get(key, 0) + len(evt)

                    # save filtered events
                    log.info(f"Saving filtered data from file: {file}")
                    ak.to_parquet( ak.zip(events_pass_filter), path_filtered_single_file)

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

        # get structured data
        self.structured_data = self.structure_data()

        return self.data

    
    def structure_data(self):
        # structured data as input of postanalysis
        def get_p4(events, flag, prefix='GenPart_vector'):
            p4 = np.zeros((len(events), 4))
            # # make sure there is only one entry per event
            # assert all(grouped.size() == 1), "Multiple entries found for events in get_p4."
            # if there are multiple entries, take the last one
            p4[:, 0] = (events[f'{prefix}_fCoordinates_fX'][flag][...,-1])
            p4[:, 1] = (events[f'{prefix}_fCoordinates_fY'][flag][...,-1])
            p4[:, 2] = (events[f'{prefix}_fCoordinates_fZ'][flag][...,-1])
            p4[:, 3] = (events[f'{prefix}_fCoordinates_fT'][flag][...,-1])
            return p4
        # interpretation of status code: https://github.com/jingyucms/Delphi-Sim-Pipeline/blob/main/pythia8_generate.cpp#L17-L43 and https://pythia.org/latest-manual/ParticleProperties.html

        for channel, events in self.data.items():
            # truth info
            truth_flag_intermediate_state = (events['GenPart_status']==21)
            truth_flag_tau1 = ((events['GenPart_pdgId']==-15) & truth_flag_intermediate_state)
            truth_flag_tau2 = ((events['GenPart_pdgId']==15) & truth_flag_intermediate_state)
            truth_flag_Z = ((events['GenPart_pdgId']==23) & truth_flag_intermediate_state)

            truth_flag_final_status = (events['GenPart_status']==1)
            truth_flag_vischild_tau1 = ((events['GenPart_pdgId']==211) & truth_flag_final_status)
            truth_flag_vischild_tau2 = ((events['GenPart_pdgId']==-211) & truth_flag_final_status)
            truth_flag_nu1 = ((events['GenPart_pdgId']==16) & truth_flag_final_status)
            truth_flag_nu2 = ((events['GenPart_pdgId']==-16) & truth_flag_final_status)

            # reco_flag_pip = ((events['Part_pdgId']==41))
            # reco_flag_pim = ((events['Part_pdgId']==-41))
            return {
                f'{channel}/TRUTH/tau1_p4': get_p4(events, truth_flag_tau1),
                f'{channel}/TRUTH/tau2_p4': get_p4(events, truth_flag_tau2),
                f'{channel}/TRUTH/vischild_tau1_p4': get_p4(events, truth_flag_vischild_tau1),
                f'{channel}/TRUTH/vischild_tau2_p4': get_p4(events, truth_flag_vischild_tau2),
                f'{channel}/TRUTH/Z_p4': get_p4(events, truth_flag_Z),
                f'{channel}/TRUTH/nu_tau1_p4': get_p4(events, truth_flag_nu1),
                f'{channel}/TRUTH/nu_tau2_p4': get_p4(events, truth_flag_nu2),
                # f'{channel}/RECO/piplus_p4': get_p4(events, reco_flag_pip, prefix='Part_fourMomentum'),
                # f'{channel}/RECO/piminus_p4': get_p4(events, reco_flag_pim, prefix='Part_fourMomentum'),
            }
    
    def save_data(self):
        output_file = self.config.get("output_dir", "./") + "/filtered_data.parquet"
        ak.to_parquet(ak.zip(self.data), output_file)
        log.info(f"Data saved to {output_file}.")
        # test loading saved data
        loaded_test = ak.from_parquet(output_file)
        for key in loaded_test.fields:
            print(f"Loaded test field {key} has {len(loaded_test[key])} events.")
            print(loaded_test[key])

        # ...
        structured_output_file = output_file.replace('.h5', '_structured.npy')
        np.save(structured_output_file, self.structured_data)
        # To load the structured data, use: np.load(structured_output_file, allow_pickle=True).item()
        log.info(f"Structured data saved to {structured_output_file}.")


    def filter_data(self, df_dict: dict) -> dict:
        random_df = next(iter(df_dict.values()))
        self.filter_results['initial_total_num_events'] += len(random_df.groupby(level=0))
        pass_filter = pd.Series([True] * len(random_df), index=random_df.index)
        
        # Apply filter to GenPart dataframe
        assert 'GenPart' in df_dict, "GenPart dataframe not found in input dictionary."
        pass_genpart_filter = df_dict['GenPart'].groupby(level=0).apply(filter_event)
        self.filter_results['single pion filter'] = self.filter_results.get('single pion filter', 0) + sum(pass_genpart_filter)
        pass_filter &= pass_genpart_filter

        # get idx of events passing the filter
        idx_to_keep = pass_filter[pass_filter].index.unique()

        return {
            key: df.loc[idx_to_keep]
            for key, df in df_dict.items()
        }


    def run(self, df: pd.DataFrame):
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

    import numpy as np
    import pandas as pd
    # test reading output file
    df = pd.read_hdf("filtered_data.h5")
    print(df)
    ary = np.load("filtered_data_structured.npy", allow_pickle=True).item()
    print(ary)

    # input_file = "/eos/user/c/cmo/project/ZtautauLep/simulation/run/251029_Ztautau_singlePionDecay/simana_job_17827112_47_ttree.root"
    # tree_name = "t"

    # f = ur.open(input_file)
    # tree = f[tree_name]

    # gen_part_branches = []
    # for b in tree.keys():
    #     try:
    #         btype = tree[b].typename
    #         print(b, btype)
    #         if "[]" in btype or "ROOT::Math" in btype or "/" in b:
    #             continue
    #         elif "GenPart" in b:
    #             gen_part_branches.append(b)
    #     except:
    #         print(f"Could not interpret branch {b}. Skipping.")
    #         continue


    # print(gen_part_branches)
    # gen_part_df = tree.arrays(gen_part_branches, library="pd")
    # print(gen_part_df)

    # flag_filtered = gen_part_df.groupby(level=0).apply(filter_event)

    # print(flag_filtered)
    # print(flag_filtered[flag_filtered].index)
    # print(f"Filter efficiency: {sum(flag_filtered) / len(flag_filtered):.4f}")
