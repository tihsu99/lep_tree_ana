import numpy as np
import pandas as pd
import uproot as ur
import logging
import vector
import glob
import os

log = logging.getLogger(__name__)

def filter_event(row):
    # no kaon, lambda and Xi0
    if not all(status != 4 for status in row['GenPart_status']):
        return False
    flag_final_products = row['GenPart_status'] == 1
    abs_pdg_ids = abs(row['GenPart_pdgId'])
    # ensure single pion decay: exactly two charged pions, no neutral pions, no short-lived particles, no kaons, eta, omega, neutrinos other than nu_tau
    num_pipm = sum(abs_pdg_ids[flag_final_products] == 211)
    num_short_lived = sum(row['GenPart_status'] == 4)
    # in turn: 111 (neutral pions), 321 (kaons), 221 (eta), 223 (omega), 14 (muon neutrinos), 12 (electron neutrinos)
    num_unwanted = \
        sum(abs_pdg_ids == 111) + \
        sum(abs_pdg_ids == 321) + \
        sum(abs_pdg_ids == 221) + \
        sum(abs_pdg_ids == 223) + \
        sum(abs_pdg_ids == 14)  + \
        sum(abs_pdg_ids == 12)  
    if num_pipm != 2 or num_short_lived > 0 or num_unwanted > 0:
        return False
    return True

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
        if os.path.exists(self.config.get("output_dir", "./") + "/filtered_data.h5"):
            # ask user if they want to load existing data
            load_existing = input(f"Filtered data file already exists. Do you want to reload and filter data from input files? (y/n): ")
            if load_existing.lower() == 'y':
                log.info("Re-loading and filtering data from input files.")
            else:
                log.info("Loading existing filtered data.")
                self.data = pd.read_hdf(self.config.get("output_dir", "./") + "/filtered_data.h5", key='GenPart')
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

        log.info(f"DataLoader initialization complete. Loaded {self.data['GenPart'].index.nunique()} events from {len(self.input_files)} files.")

    

    def load_data(self) -> pd.DataFrame:
        # Identify branches to load
        f = ur.open(self.input_files[0])
        tree = f[self.tree_name]

        common_evt_branches = ["Event_evtNumber"]
        gen_part_branches = ["pdgId", "status", "vector_fCoordinates_fX", "vector_fCoordinates_fY", "vector_fCoordinates_fZ", "vector_fCoordinates_fT"]
        gen_part_branches = [f"GenPart_{b}" for b in gen_part_branches]
        # for b in tree.keys():
        #     try:
        #         btype = tree[b].typename
        #         if "[]" in btype or "ROOT::Math" in btype or "/" in b:
        #             log.debug(f"Skipping branch {b} of type {btype}.")
        #             continue
        #         elif "GenPart" in b:
        #             log.info(f"Loading branch {b} of type {btype} for GenPart.")
        #             gen_part_branches.append(b)
        #     except:
        #         log.warning(f"Could not interpret branch {b}. Skipping.")
        #         continue
        
        # Load data from all files
        initial_total_num_events = 0
        for file in self.input_files:
            path_filtered_single_file = self.config.get("output_dir", "./") + "/filtered_data_files/" + os.path.basename(file).replace('.root', '_filtered.h5')
            if os.path.exists(path_filtered_single_file):
                log.info(f"Filtered data file for {file} already exists at {path_filtered_single_file}. Loading filtered data.")
                with pd.HDFStore(path_filtered_single_file, mode='r') as store:
                    for key in store.keys():
                        key_clean = key.lstrip('/')
                        df = store.get(key_clean)
                        df['evtNumber'] = df['Event_evtNumber'] + initial_total_num_events
                        # df['evtNumber'] = df['Event_evtNumber']
                        df = df.set_index('evtNumber', drop=False)
                        initial_total_num_events = initial_total_num_events + df['initial_total_num_events'].iloc[0]
                        self.data.setdefault(key_clean, []).append(df)
            else:
                continue
                log.info(f"Loading data from file: {file}")
                try:
                    f = ur.open(file)
                    tree = f[self.tree_name]
                    df_genpart = tree.arrays(common_evt_branches + gen_part_branches, library="pd")
                    df_genpart['initial_total_num_events'] = len(df_genpart.groupby(level=0))
                    # adjust event index to be unique across files
                    # the original Event_evtNumber starts from 1 for each file
                    df_genpart['evtNumber'] = df_genpart['Event_evtNumber'] + initial_total_num_events
                    df_genpart = df_genpart.set_index('evtNumber', drop=False)
                    initial_total_num_events = df_genpart['evtNumber'].max()
                except Exception as e:
                    log.error(f"Error reading file {file} or tree {self.tree_name}: {e}")
                    continue

                # filter events
                data_pass_filter = self.filter_data({
                    'GenPart': df_genpart
                    })

                # save filtered results of each file into a separate HDF5 file
                log.info(f"Saving filtered data from file: {file}")
                os.makedirs(os.path.dirname(path_filtered_single_file), exist_ok=True)
                with pd.HDFStore(path_filtered_single_file, mode='w') as store:
                    for key, df in data_pass_filter.items():
                        store.put(key, df)
                log.info(f"Filtered data from file saved to: {path_filtered_single_file}")
                
                # store filtered data
                for key in data_pass_filter:
                    self.data.setdefault(key, []).append(data_pass_filter[key])

        # Concatenate data from all files
        for key in self.data:
            self.data[key] = pd.concat(self.data[key], axis=0)
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
        # interpretation of status code: https://github.com/jingyucms/Delphi-Sim-Pipeline/blob/main/pythia8_generate.cpp#L17-L43 and https://pythia.org/latest-manual/ParticleProperties.html
        truth_particles = self.data['GenPart']
        flag_intermediate_state = (truth_particles['GenPart_status']==21)
        flag_tau1 = ((truth_particles['GenPart_pdgId']==-15) & flag_intermediate_state)
        flag_tau2 = ((truth_particles['GenPart_pdgId']==15) & flag_intermediate_state)
        flag_Z = ((truth_particles['GenPart_pdgId']==23) & flag_intermediate_state)

        flag_final_status = (truth_particles['GenPart_status']==1)
        flag_vischild_tau1 = ((truth_particles['GenPart_pdgId']==211) & flag_final_status)
        flag_vischild_tau2 = ((truth_particles['GenPart_pdgId']==-211) & flag_final_status)

        def get_p4(df, flag):
            p4 = np.zeros((len(df[flag].groupby(level=0)), 4))
            grouped = df[flag].groupby(level=0)
            # # make sure there is only one entry per event
            # assert all(grouped.size() == 1), "Multiple entries found for events in get_p4."
            # if there are multiple entries, take the last one
            p4[:, 0] = grouped['GenPart_vector_fCoordinates_fX'].last().values
            p4[:, 1] = grouped['GenPart_vector_fCoordinates_fY'].last().values
            p4[:, 2] = grouped['GenPart_vector_fCoordinates_fZ'].last().values
            p4[:, 3] = grouped['GenPart_vector_fCoordinates_fT'].last().values
            return p4

        return {
                'TRUTH/tau1_p4': get_p4(truth_particles, ((truth_particles['GenPart_pdgId']==-15) & flag_intermediate_state)),
                'TRUTH/tau2_p4': get_p4(truth_particles, ((truth_particles['GenPart_pdgId']==15) & flag_intermediate_state)),
                'TRUTH/vischild_tau1_p4': get_p4(truth_particles, ((truth_particles['GenPart_pdgId']==211) & flag_final_status)),
                'TRUTH/vischild_tau2_p4': get_p4(truth_particles, ((truth_particles['GenPart_pdgId']==-211) & flag_final_status)),
                'TRUTH/nu_tau1_p4': get_p4(truth_particles, (truth_particles['GenPart_pdgId']==-16) & flag_final_status),
                'TRUTH/nu_tau2_p4': get_p4(truth_particles, (truth_particles['GenPart_pdgId']==16) & flag_final_status),
                'TRUTH/Z_p4': get_p4(truth_particles, ((truth_particles['GenPart_pdgId']==23) & flag_intermediate_state)),
            }
    
    def save_data(self):
        output_file = self.config.get("output_dir", "./") + "/filtered_data.h5"
        with pd.HDFStore(output_file, mode='w') as store:
            for key, df in self.data.items():
                store.put(key, df)
        # To load the data, use: pd.read_hdf(output_file, key='GenPart')
        log.info(f"Data saved to {output_file}.")

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
